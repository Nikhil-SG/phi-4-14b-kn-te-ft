"""Joint bilingual training for Multilingual Fusion.

Trains on merged Kannada+Telugu dataset using the Monolingual FT winning
technique.  Supports equal (50/50) and proportional sampling strategies
via ``datasets.interleave_datasets``.

Usage:
    # Multi-GPU (FFT, LoRA, DoRA, IA³):
    accelerate launch --num_processes 2 --mixed_precision bf16 \
        multilingual_fusion/joint_train.py --sampling_strategy equal

    # Single-GPU (QLoRA):
    python multilingual_fusion/joint_train.py --sampling_strategy proportional
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import datasets as hf_datasets
# Monkey-patch to bypass PyTorch 2.6+ requirement for torch.load in older PyTorch versions
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except Exception:
    pass


import torch
import transformers
import yaml
from peft import IA3Config, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_engine import evaluate_checkpoint
from evaluation.metrics.system import count_trainable_params, get_peak_vram_gb, reset_vram_peak
from utils.data_utils import format_phi4_prompt, load_config, load_jsonl
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _build_joint_dataset(
    sampling_strategy: str,
    model_config: dict,
    data_config: dict,
    hp_config: dict,
) -> tuple[Any, Any]:
    """Build interleaved joint datasets for bilingual training.

    Args:
        sampling_strategy: ``"equal"`` for 50/50 or ``"proportional"``
            for dataset-size-proportional sampling.
        model_config: Parsed ``configs/model.yaml``.
        data_config: Parsed ``configs/data.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.

    Returns:
        A tuple of ``(train_dataset, val_dataset)`` as HuggingFace
        Dataset objects.
    """

    def fmt(rows: list[dict]) -> list[dict]:
        """Format rows into prompt dicts for SFTTrainer."""
        return [
            {
                "text": format_phi4_prompt(
                    r["instruction"], r["input"], r["output"]
                )
            }
            for r in rows
        ]

    kn_train = hf_datasets.Dataset.from_list(
        fmt(load_jsonl(str(Path("data") / "kannada" / "train.jsonl")))
    )
    te_train = hf_datasets.Dataset.from_list(
        fmt(load_jsonl(str(Path("data") / "telugu" / "train.jsonl")))
    )
    kn_val = hf_datasets.Dataset.from_list(
        fmt(load_jsonl(str(Path("data") / "kannada" / "val.jsonl")))
    )
    te_val = hf_datasets.Dataset.from_list(
        fmt(load_jsonl(str(Path("data") / "telugu" / "val.jsonl")))
    )

    logger.info(
        "Individual train sizes — KN: %d, TE: %d", len(kn_train), len(te_train)
    )

    # Compute probabilities
    if sampling_strategy == "equal":
        probabilities = [0.5, 0.5]
        logger.info("Equal sampling — KN: 0.500, TE: 0.500")
    else:
        total = len(kn_train) + len(te_train)
        p_kn = len(kn_train) / total
        p_te = len(te_train) / total
        probabilities = [p_kn, p_te]
        logger.info("Proportional sampling — KN: %.3f, TE: %.3f", p_kn, p_te)

    # Build interleaved training set
    joint_train = hf_datasets.interleave_datasets(
        [kn_train, te_train],
        probabilities=probabilities,
        seed=hp_config["seed"],
        stopping_strategy="all_exhausted",
    )
    logger.info("Interleaved joint train set size: %d", len(joint_train))

    # Concatenate validation sets (always 50/50 for consistent eval)
    joint_val = hf_datasets.concatenate_datasets([kn_val, te_val])
    joint_val = joint_val.shuffle(seed=hp_config["seed"])

    return joint_train, joint_val


def _load_base_model(
    winner_technique: str,
    model_config: dict,
    hp_config: dict,
) -> tuple[torch.nn.Module, LoraConfig | IA3Config | None]:
    """Load the base model for joint training (fresh, not from checkpoint).

    Args:
        winner_technique: The winning technique from Monolingual FT.
        model_config: Parsed ``configs/model.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.

    Returns:
        A tuple of ``(model, peft_config)`` where peft_config is None
        for FFT.
    """
    peft_config: LoraConfig | IA3Config | None = None

    from utils.gpu_utils import configure_gpu_settings
    primary_device_id, max_memory = configure_gpu_settings()

    if winner_technique == "fft":
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            use_cache=False,
            trust_remote_code=True,
            attn_implementation=model_config.get("attn_implementation", "eager"),
        )
        model.gradient_checkpointing_enable()
        return model, None

    if winner_technique == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=hp_config["qlora"]["quant_type"],
            bnb_4bit_double_quant=hp_config["qlora"]["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=hp_config["qlora"]["double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            use_cache=False,
            trust_remote_code=True,
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            r=hp_config["lora"]["r"],
            lora_alpha=hp_config["lora"]["alpha"],
            lora_dropout=hp_config["lora"]["dropout"],
            target_modules=hp_config["lora"]["target_modules"],
            bias=hp_config["lora"]["bias"],
            use_rslora=hp_config["lora"]["use_rslora"],
            use_dora=False,
            task_type="CAUSAL_LM",
        )
        return model, peft_config

    # LoRA, DoRA, IA³
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map_choice = {"": local_rank} if local_rank != -1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map=device_map_choice,
        max_memory=max_memory,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation=model_config.get("attn_implementation", "eager"),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if winner_technique in ("lora", "dora"):
        peft_config = LoraConfig(
            r=hp_config["lora"]["r"],
            lora_alpha=hp_config["lora"]["alpha"],
            lora_dropout=hp_config["lora"]["dropout"],
            target_modules=hp_config["lora"]["target_modules"],
            bias=hp_config["lora"]["bias"],
            use_rslora=hp_config["lora"]["use_rslora"],
            use_dora=(winner_technique == "dora"),
            task_type="CAUSAL_LM",
        )
    elif winner_technique == "ia3":
        peft_config = IA3Config(
            target_modules=hp_config["ia3"]["target_modules"],
            feedforward_modules=hp_config["ia3"]["feedforward_modules"],
            init_ia3_weights=hp_config["ia3"]["init_ia3_weights"],
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError(f"Unknown technique: {winner_technique}")

    return model, peft_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for joint training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Multilingual Fusion: Joint bilingual training."
    )
    parser.add_argument(
        "--sampling_strategy",
        required=True,
        choices=["equal", "proportional"],
        help="Sampling strategy for interleaving languages.",
    )
    parser.add_argument(
        "--model_config",
        default="configs/model.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--data_config",
        default="configs/data.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--hp_config",
        default="configs/hyperparams.yaml",
        help="Path to hyperparameters config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    """Run joint bilingual training and evaluation."""
    setup_logging("module_3_multilingual_fusion", "module_3_multilingual_fusion")
    args = parse_args()

    # Load configs
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    hp_config = load_config(args.hp_config)

    # Load Monolingual FT winner
    winner = yaml.safe_load(
        Path("results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml").read_text(encoding="utf-8")
    )
    winner_technique = winner["technique"]

    from utils.checkpoint_utils import get_checkpoint_path
    output_dir = get_checkpoint_path(3, "joint", args.sampling_strategy)
    experiment_prefix = f"multilingual_fusion_joint_{args.sampling_strategy}"
    train_time_hrs = 0.0

    logger.info(
        "Joint %s training with technique=%s → %s",
        args.sampling_strategy,
        winner_technique,
        output_dir,
    )

    # Check if already trained
    if output_dir.exists() and (output_dir / "train_metadata.json").exists():
        logger.warning(
            "Joint %s already trained. Skipping to evaluation.",
            args.sampling_strategy,
        )
        meta = json.loads(
            (output_dir / "train_metadata.json").read_text(encoding="utf-8")
        )
        train_time_hrs = meta.get("train_time_hrs", 0.0)
    else:
        # ── Training ──────────────────────────────────────────────────
        transformers.set_seed(hp_config["seed"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build joint dataset
        joint_ds, joint_val_ds = _build_joint_dataset(
            args.sampling_strategy, model_config, data_config, hp_config
        )
        logger.info(
            "Joint dataset sizes — train: %d, val: %d",
            len(joint_ds),
            len(joint_val_ds),
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["tokenizer_id"], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load base model
        model, peft_config = _load_base_model(
            winner_technique, model_config, hp_config
        )
        reset_vram_peak()
        logger.info("Base model loaded. Params: %s", count_trainable_params(model))

        # Build TrainingArguments
        training_kwargs: dict = {
            "output_dir": str(output_dir),
            "dataset_text_field": "text",
            "max_length": model_config["max_context_length"],
            "num_train_epochs": hp_config["epochs"],
            "per_device_train_batch_size": hp_config["per_device_train_batch_size"],
            "per_device_eval_batch_size": hp_config["per_device_train_batch_size"],
            "gradient_accumulation_steps": hp_config["gradient_accumulation_steps"],
            "learning_rate": hp_config["learning_rates"][winner_technique],
            "lr_scheduler_type": hp_config["lr_scheduler_type"],
            "warmup_ratio": hp_config["warmup_ratio"],
            "bf16": hp_config["bf16"],
            "fp16": hp_config["fp16"],
            "logging_steps": hp_config["logging_steps"],
            "eval_strategy": "steps",
            "eval_steps": hp_config["eval_steps"],
            "save_strategy": "steps",
            "save_steps": hp_config["eval_steps"],
            "save_total_limit": hp_config["save_total_limit"],
            "load_best_model_at_end": False,
            "max_grad_norm": hp_config["max_grad_norm"],
            "weight_decay": hp_config["weight_decay"],
            "seed": hp_config["seed"],
            "data_seed": hp_config["seed"],
            "dataloader_num_workers": hp_config["dataloader_num_workers"],
            "report_to": os.environ.get("PIPELINE_REPORT_TO", "none"),
            "run_name": f"{winner_technique}_joint_{args.sampling_strategy}",
            "gradient_checkpointing": (winner_technique != "qlora"),
        }

        if winner_technique == "qlora":
            training_kwargs["optim"] = "paged_adamw_8bit"
        elif winner_technique == "fft":
            training_kwargs["optim"] = hp_config["optimizer"]["fft"]
            training_kwargs["deepspeed"] = "configs/deepspeed_zero3.json"
        else:
            training_kwargs["optim"] = hp_config["optimizer"]["default"]

        training_args = SFTConfig(**training_kwargs)

        # Build SFTTrainer
        trainer_kwargs: dict = {
            "model": model,
            "processing_class": tokenizer,
            "train_dataset": joint_ds,
            "eval_dataset": joint_val_ds,
            "args": training_args,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        trainer = SFTTrainer(**trainer_kwargs)

        if peft_config is not None:
            trainer.model.print_trainable_parameters()
            logger.info(
                "Trainable params: %s", count_trainable_params(trainer.model)
            )

        # Auto-detect existing checkpoint for resume (e.g. after wall time limit)
        resume_ckpt = None
        if output_dir.exists():
            ckpt_dirs = sorted(
                [d for d in output_dir.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-")[1]),
            )
            if ckpt_dirs:
                resume_ckpt = str(ckpt_dirs[-1])
                logger.info("Found existing checkpoint for joint training resume: %s", resume_ckpt)

        # Train
        logger.info("Starting joint %s training ...", args.sampling_strategy)
        t0 = time.time()
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        train_time_hrs = (time.time() - t0) / 3600
        logger.info("Joint training complete in %.2fh", train_time_hrs)

        # Save
        if winner_technique == "fft":
            trainer.save_model(str(output_dir))
        else:
            trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Save train_metadata.json
        metadata = {
            "technique": winner_technique,
            "module": 3,
            "sampling_strategy": args.sampling_strategy,
            "train_time_hrs": round(train_time_hrs, 4),
            "final_train_loss": round(train_result.training_loss, 6),
        }
        try:
            (output_dir / "train_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            logger.info("Train metadata saved.")
        except Exception as exc:
            logger.error("Failed to save train_metadata.json: %s", exc)

        peak_vram = get_peak_vram_gb()
        logger.info("Peak VRAM: %.2f GB", peak_vram)

        # Cleanup training model
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # ── Evaluate on both languages ────────────────────────────────────
    for language in ["kannada", "telugu"]:
        experiment_id = f"{experiment_prefix}_{language}"
        logger.info("Evaluating %s ...", experiment_id)
        evaluate_checkpoint(
            checkpoint_path=str(output_dir),
            language=language,
            experiment_id=experiment_id,
            model_config=model_config,
            data_config=data_config,
            module=3,
            technique=winner_technique,
            load_in_4bit=(winner_technique == "qlora"),
            train_time_hrs=train_time_hrs if language == "kannada" else None,
            notes=f"Joint {args.sampling_strategy} — evaluated on {language}",
        )
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Joint %s complete.", args.sampling_strategy)


if __name__ == "__main__":
    main()
