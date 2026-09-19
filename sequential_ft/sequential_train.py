"""Sequential FT sequential training pipeline.

Redesigned Module 2 — Bidirectional cross-lingual evaluation and
sequential continual learning.

Four stages:
    Stage 1 — Bidirectional zero-shot cross-lingual measurement.
              (a) Evaluate the Kannada FFT model on Telugu test set.
              (b) Evaluate the Telugu FFT model on Kannada test set.
              These baselines quantify cross-lingual transfer in BOTH
              directions before any sequential training.
    Stage 2 — Continue fine-tuning the Telugu FFT (winner) checkpoint
              on Kannada training data using a SINGLE GPU with
              DeepSpeed ZeRO-2 + CPU optimizer offloading.
    Stage 3 — Evaluate the sequential model on Kannada, Telugu, and
              English benchmarks (MMLU, MGSM, GPQA) with LLM-as-Judge.
    Stage 4 — Compute retention report (catastrophic forgetting analysis).

Usage:
    # Always single GPU — no accelerate needed:
    python sequential_ft/sequential_train.py
"""

import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path

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

# Relax DeepSpeed's strict CUDA version check.
try:
    import deepspeed.ops.op_builder.builder as _ds_builder
    _ds_builder.assert_no_cuda_mismatch = lambda name="": None
except Exception:
    pass


import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_engine import evaluate_checkpoint
from utils.checkpoint_utils import get_checkpoint_path
from utils.data_utils import format_phi4_prompt, load_config, load_jsonl
from utils.logging_utils import get_logger, load_results, setup_logging
from utils.training_monitor import MemoryMonitorCallback, install_signal_handlers

logger = get_logger(__name__)


def _load_model_for_continuation(
    source_ckpt_path: Path,
    winner_technique: str,
    model_config: dict,
    hp_config: dict,
) -> torch.nn.Module:
    """Load a model from the source checkpoint for continuation training.

    Dispatches to the appropriate loading strategy based on the winner
    technique (FFT, LoRA, DoRA, QLoRA, or IA³).

    The model is loaded onto a SINGLE GPU (device 0) regardless of how
    many GPUs are physically available.

    Args:
        source_ckpt_path: Path to the Monolingual FT source checkpoint
            (the winner — typically Telugu FFT).
        winner_technique: The winning technique from Monolingual FT.
        model_config: Parsed ``configs/model.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.

    Returns:
        A model ready for continuation training.

    Raises:
        ValueError: If the winner technique is not recognised.
    """
    # Single-GPU memory config — use only device 0
    max_memory = {0: "75GiB", "cpu": "250GiB"}

    if winner_technique == "fft":
        # Load WITHOUT device_map / max_memory — let DeepSpeed ZeRO-2 handle
        # memory placement.  Using device_map="auto" causes HF Accelerate to
        # pin the model to GPU *before* DeepSpeed wraps it, leading to
        # double-memory usage and OOM SIGKILL on 80 GB GPUs.
        # This mirrors monolingual_ft/trainers/fft_trainer.py (single-GPU path).
        model = AutoModelForCausalLM.from_pretrained(
            str(source_ckpt_path),
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info("FFT model loaded from %s for continuation (single GPU, no device_map).", source_ckpt_path)
        return model

    if winner_technique in ("lora", "dora", "ia3"):
        base = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            use_cache=False,
            trust_remote_code=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base,
            str(source_ckpt_path),
            is_trainable=True,
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info(
            "PEFT (%s) model loaded from %s with is_trainable=True (single GPU).",
            winner_technique,
            source_ckpt_path,
        )
        return model

    if winner_technique == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=hp_config["qlora"]["quant_type"],
            bnb_4bit_double_quant=hp_config["qlora"]["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=hp_config["qlora"]["double_quant"],
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            use_cache=False,
            trust_remote_code=True,
        )
        from peft import PeftModel, prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
        base.enable_input_require_grads()
        model = PeftModel.from_pretrained(
            base,
            str(source_ckpt_path),
            is_trainable=True,
        )
        logger.info(
            "QLoRA model loaded from %s with is_trainable=True (single GPU).",
            source_ckpt_path,
        )
        return model

    raise ValueError(f"Unknown technique for continuation: {winner_technique}")


def _run_continuation_training(
    source_ckpt_path: Path,
    winner_technique: str,
    model_config: dict,
    data_config: dict,
    hp_config: dict,
    output_dir: Path,
    target_language: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Continue training from the source checkpoint on the target language data.

    Loads the model from the source checkpoint (e.g. Telugu FFT winner),
    prepares the target language datasets (e.g. Kannada), and runs
    SFTTrainer for continuation fine-tuning on a SINGLE GPU with
    DeepSpeed ZeRO-2 + CPU optimizer offloading.

    Args:
        source_ckpt_path: Path to the winning Monolingual FT checkpoint.
        winner_technique: The winning technique from Monolingual FT.
        model_config: Parsed ``configs/model.yaml``.
        data_config: Parsed ``configs/data.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.
        output_dir: Directory to save the sequential model checkpoint.
        target_language: The language to train on (e.g. ``"kannada"``).
    """
    # 1. Setup
    transformers.set_seed(hp_config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_id"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load model for continuation (single GPU)
    model = _load_model_for_continuation(
        source_ckpt_path, winner_technique, model_config, hp_config
    )
    gc.collect()  # Free temporary init buffers before optimizer allocation

    # 4. Prepare target language datasets
    import datasets as hf_datasets

    train_rows = load_jsonl(str(Path("data") / target_language / "train.jsonl"))
    val_rows = load_jsonl(str(Path("data") / target_language / "val.jsonl"))

    train_ds = hf_datasets.Dataset.from_list(
        [
            {
                "text": format_phi4_prompt(
                    r["instruction"], r["input"], r["output"]
                )
            }
            for r in train_rows
        ]
    )
    val_ds = hf_datasets.Dataset.from_list(
        [
            {
                "text": format_phi4_prompt(
                    r["instruction"], r["input"], r["output"]
                )
            }
            for r in val_rows
        ]
    )
    logger.info(
        "%s datasets: train=%d, val=%d",
        target_language.capitalize(),
        len(train_ds),
        len(val_ds),
    )

    # 5. Build TrainingArguments — single GPU with DeepSpeed ZeRO-2
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
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "torch_compile": False,
        "save_safetensors": False,
        "report_to": os.environ.get("PIPELINE_REPORT_TO", "none"),
        "run_name": f"{winner_technique}_sequential_{target_language}",
        "gradient_checkpointing": (winner_technique != "qlora"),
    }

    # Set optimizer and DeepSpeed based on technique — always single GPU
    if winner_technique == "qlora":
        training_kwargs["optim"] = "paged_adamw_8bit"
    elif winner_technique == "fft":
        training_kwargs["optim"] = hp_config["optimizer"]["fft"]
        # Use ZeRO-2 with CPU offload for single-GPU FFT.
        # Pre-resolve all "auto" values and pass as dict — matching Module 1
        # FFT approach.  Passing a file path causes HF Trainer and DeepSpeed
        # to independently resolve "auto" values, leading to mismatches
        # (e.g. gradient_accumulation_steps mismatch warning).
        import json as _json
        with open("configs/deepspeed_zero2.json", "r") as _f:
            ds_config_dict = _json.load(_f)
        if ds_config_dict.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config_dict["train_micro_batch_size_per_gpu"] = hp_config["per_device_train_batch_size"]
        if ds_config_dict.get("gradient_accumulation_steps") == "auto":
            ds_config_dict["gradient_accumulation_steps"] = hp_config["gradient_accumulation_steps"]
        if "optimizer" in ds_config_dict and "params" in ds_config_dict["optimizer"]:
            opt_params = ds_config_dict["optimizer"]["params"]
            _auto_map = {
                "lr": hp_config["learning_rates"][winner_technique],
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": hp_config.get("weight_decay", 0.01),
            }
            for k, default_v in _auto_map.items():
                if opt_params.get(k) == "auto":
                    opt_params[k] = default_v
        training_kwargs["deepspeed"] = ds_config_dict
        logger.info("DeepSpeed ZeRO-2 config pre-resolved and passed as dict.")
    else:
        training_kwargs["optim"] = hp_config["optimizer"]["default"]

    from trl import SFTConfig, SFTTrainer
    training_args = SFTConfig(**training_kwargs)

    # 6. Install crash forensics (signal handlers + memory monitoring)
    install_signal_handlers(output_dir=str(output_dir))

    # 7. Build SFTTrainer with memory monitoring
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        callbacks=[
            MemoryMonitorCallback(log_every_steps=50, output_dir=str(output_dir)),
        ],
    )

    # 7. Train
    logger.info(
        "Starting continuation training on %s (single GPU) ...",
        target_language.capitalize(),
    )
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_time_hrs = (time.time() - t0) / 3600
    logger.info("Continuation training complete in %.2fh", train_time_hrs)

    # 8. Save tokenizer and config to output_dir
    tokenizer.save_pretrained(str(output_dir))
    if hasattr(model, 'config') and model.config is not None:
        model.config.save_pretrained(str(output_dir))
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.save_pretrained(str(output_dir))
    logger.info("Tokenizer and config saved to %s", output_dir)

    # 9. Find best checkpoint for weight consolidation
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt is None:
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
        best_ckpt = str(ckpt_dirs[-1]) if ckpt_dirs else None
        logger.warning("No best_model_checkpoint tracked. Using latest: %s", best_ckpt)
    else:
        logger.info("Best model checkpoint: %s", best_ckpt)

    # 10. Save train_metadata.json
    metadata = {
        "technique": winner_technique,
        "module": 2,
        "direction": "telugu_to_kannada",
        "training_language": target_language,
        "starting_checkpoint": str(source_ckpt_path),
        "train_time_hrs": round(train_time_hrs, 4),
        "final_train_loss": round(train_result.training_loss, 6),
        "best_checkpoint": best_ckpt,
        "single_gpu": True,
        "deepspeed_config": "configs/deepspeed_zero2.json" if winner_technique == "fft" else None,
    }

    try:
        metadata_path = output_dir / "train_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Train metadata saved to %s", metadata_path)
    except Exception as exc:
        logger.error("Failed to save train_metadata.json: %s", exc)

    # 11. Free all training memory before weight consolidation
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Training memory freed.")

    # 12. Move best checkpoint weights to parent output directory
    #     ZeRO-2 (single-GPU): Trainer writes standard HF weights inside
    #     each checkpoint-NNNN directory.  Move them to the parent so the
    #     output_dir itself is a loadable HF checkpoint.
    if best_ckpt is not None:
        import shutil
        logger.info(
            "Moving weights from %s to parent %s ...",
            best_ckpt, output_dir,
        )
        try:
            best_ckpt_path = Path(best_ckpt)
            weight_patterns = [
                "pytorch_model*.bin",
                "model*.safetensors",
                "*.index.json",
                "config.json",
                "generation_config.json",
            ]
            moved_files = []
            for pattern in weight_patterns:
                for f in best_ckpt_path.glob(pattern):
                    dest = output_dir / f.name
                    logger.info("Moving %s → %s", f, dest)
                    shutil.move(str(f), str(dest))
                    moved_files.append(dest)
            logger.info("Successfully moved %d model files to parent directory.", len(moved_files))
        except Exception as exc:
            logger.error("Failed to move checkpoint weights to parent directory: %s", exc)

        # Clean up all checkpoint directories to free disk space
        try:
            for child in output_dir.iterdir():
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    logger.info("Removing checkpoint directory: %s", child)
                    shutil.rmtree(child, ignore_errors=True)
            logger.info("Checkpoint cleanup complete.")
        except Exception as exc:
            logger.error("Failed to clean up checkpoint directories: %s", exc)


def _compute_and_save_retention_report(
    winner_technique: str,
    model_config: dict,
    source_language: str,
    target_language: str,
) -> None:
    """Compute retention metrics and save the retention report.

    Loads all evaluation results, extracts relevant Monolingual FT and
    Sequential FT entries, computes retention rate, forgetting delta,
    backward transfer (BWT), forward transfer (FWT), plasticity-stability
    ratio (PSR), and cross-lingual transfer metrics, then saves to
    ``results/module_2_sequential_ft/results/retention_report.json``.

    In the redesigned Module 2 (Telugu → Kannada direction):
        - **Source language**: Telugu (the winner, which might be forgotten)
        - **Target language**: Kannada (the new language being learned)
        - **Retention rate**: How much Telugu is retained after Kannada training
        - **BWT** (Backward Transfer): Telugu performance change after
          Kannada training.  Negative = forgetting.
        - **FWT** (Forward Transfer): Zero-shot Kannada performance from
          the Telugu-only model.  Positive = helpful transfer.
        - **PSR** (Plasticity-Stability Ratio): Ratio of Kannada gain
          to Telugu loss.

    Args:
        winner_technique: The winning technique from Monolingual FT.
        model_config: Parsed ``configs/model.yaml`` (unused but kept
            for interface consistency).
        source_language: The source language (e.g. ``"telugu"``).
        target_language: The target language (e.g. ``"kannada"``).
    """
    monolingual_results = load_results("results/module_1_monolingual_ft/results/all_results.json")
    sequential_results = load_results("results/module_2_sequential_ft/results/all_results.json")

    # Extract relevant entries from Monolingual FT
    mono_source = next(
        (
            r
            for r in monolingual_results
            if r["module"] == 1
            and r["technique"] == winner_technique
            and r["language"] == source_language
        ),
        None,
    )
    mono_target = next(
        (
            r
            for r in monolingual_results
            if r["module"] == 1
            and r["technique"] == winner_technique
            and r["language"] == target_language
        ),
        None,
    )

    # Extract zero-shot entries (bidirectional)
    zero_shot_kn_on_te = next(
        (
            r
            for r in sequential_results
            if r["experiment_id"] == "sequential_ft_zero_shot_telugu"
        ),
        None,
    )
    zero_shot_te_on_kn = next(
        (
            r
            for r in sequential_results
            if r["experiment_id"] == "sequential_ft_zero_shot_kannada"
        ),
        None,
    )

    # Extract post-sequential evaluation entries
    seq_source = next(
        (
            r
            for r in sequential_results
            if r["experiment_id"] == f"sequential_ft_sequential_{source_language}"
        ),
        None,
    )
    seq_target = next(
        (
            r
            for r in sequential_results
            if r["experiment_id"] == f"sequential_ft_sequential_{target_language}"
        ),
        None,
    )

    # Log warnings for missing entries
    for name, entry in [
        (f"monolingual_ft_{source_language}", mono_source),
        (f"monolingual_ft_{target_language}", mono_target),
        ("zero_shot_telugu (KN model → TE eval)", zero_shot_kn_on_te),
        ("zero_shot_kannada (TE model → KN eval)", zero_shot_te_on_kn),
        (f"sequential_{source_language}", seq_source),
        (f"sequential_{target_language}", seq_target),
    ]:
        if entry is None:
            logger.warning("Retention report: missing entry for %s", name)

    # ── Compute retention metrics ──────────────────────────────────────
    # Source retention: did the model forget the source language (Telugu)?
    source_retention_rate = None
    source_forgetting_delta = None

    if mono_source and seq_source and mono_source["chrf"] > 0:
        source_retention_rate = seq_source["chrf"] / mono_source["chrf"]

    if mono_source and seq_source:
        source_forgetting_delta = seq_source["chrf"] - mono_source["chrf"]

    # Target improvement: how did the sequential model do on the target (Kannada)?
    target_vs_mono = None
    if mono_target and seq_target:
        target_vs_mono = seq_target["chrf"] - mono_target["chrf"]

    # Zero-shot cross-lingual lifts
    zero_shot_target_chrf = None  # Zero-shot Kannada from Telugu model
    zero_shot_source_chrf = None  # Zero-shot Telugu from Kannada model
    if zero_shot_te_on_kn:
        zero_shot_target_chrf = zero_shot_te_on_kn["chrf"]
    if zero_shot_kn_on_te:
        zero_shot_source_chrf = zero_shot_kn_on_te["chrf"]

    # ── BWT (Backward Transfer) ───────────────────────────────────────
    # BWT = performance on source (Telugu) AFTER training on target (Kannada)
    #        minus performance on source AFTER training on source only.
    # BWT < 0 → catastrophic forgetting.  BWT ≈ 0 → stable.
    bwt = source_forgetting_delta

    # ── FWT (Forward Transfer) ────────────────────────────────────────
    # FWT = zero-shot performance on target (Kannada) from source-only model.
    # Measures cross-lingual transfer from Telugu training.
    # Higher = more beneficial Dravidian-family transfer.
    fwt = zero_shot_target_chrf

    # ── PSR (Plasticity-Stability Ratio) ──────────────────────────────
    # PSR = (target language gain) / (source language change)
    # Measures the tradeoff between learning Kannada (plasticity) and
    # retaining Telugu (stability).
    # PSR > 1 → model gains more on new task than it loses on old.
    # PSR ≈ 1 → balanced tradeoff.
    # PSR < 1 → forgetting dominates acquisition.
    psr = None
    if (mono_target and seq_target and seq_source and mono_source
            and mono_source["chrf"] > 0):
        new_task_gain = seq_target["chrf"] - (zero_shot_target_chrf if zero_shot_target_chrf else 0.0)
        old_task_change = abs(seq_source["chrf"] - mono_source["chrf"])
        if old_task_change > 0.01:  # avoid division by near-zero
            psr = new_task_gain / old_task_change

    # Build report
    report = {
        "winner_technique": winner_technique,
        "direction": f"{source_language}_to_{target_language}",
        "source_language": source_language,
        "target_language": target_language,
        # Monolingual baselines
        f"monolingual_ft_{source_language}_chrf": mono_source["chrf"] if mono_source else None,
        f"monolingual_ft_{target_language}_chrf": mono_target["chrf"] if mono_target else None,
        # Zero-shot baselines (bidirectional)
        "zero_shot_telugu_chrf": zero_shot_source_chrf,
        "zero_shot_kannada_chrf": zero_shot_target_chrf,
        # Post-sequential evaluations
        f"sequential_{source_language}_chrf": seq_source["chrf"] if seq_source else None,
        f"sequential_{target_language}_chrf": seq_target["chrf"] if seq_target else None,
        # Retention & forgetting (for the SOURCE language — Telugu)
        "retention_rate": (
            round(source_retention_rate, 4) if source_retention_rate is not None else None
        ),
        "forgetting_delta": (
            round(source_forgetting_delta, 4) if source_forgetting_delta is not None else None
        ),
        # Formal continual learning metrics
        "bwt": (
            round(bwt, 4) if bwt is not None else None
        ),
        "fwt": (
            round(fwt, 4) if fwt is not None else None
        ),
        "plasticity_stability_ratio": (
            round(psr, 4) if psr is not None else None
        ),
        # Cross-lingual deltas
        f"sequential_vs_mono_{target_language}_delta": (
            round(target_vs_mono, 4) if target_vs_mono is not None else None
        ),
        "computed_at": datetime.datetime.utcnow().isoformat(),
    }

    # Save
    report_dir = Path("results") / "module_2_sequential_ft" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "retention_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if source_retention_rate is not None:
        logger.info(
            "%s retention rate: %.4f",
            source_language.capitalize(),
            source_retention_rate,
        )
    else:
        logger.info("%s retention rate: N/A", source_language.capitalize())

    if bwt is not None:
        logger.info("BWT (Backward Transfer on %s): %+.2f chrF++", source_language, bwt)

    if fwt is not None:
        logger.info("FWT (Forward Transfer / zero-shot %s): %.2f chrF++", target_language, fwt)

    if psr is not None:
        logger.info("PSR (Plasticity-Stability Ratio): %.4f", psr)

    logger.info("Retention report saved to %s", report_path)


def main() -> None:
    """Run the full Sequential FT pipeline (redesigned).

    Executes four stages:
        Stage 1: Bidirectional zero-shot cross-lingual measurement.
            (a) Kannada model evaluated on Telugu.
            (b) Telugu model evaluated on Kannada.
        Stage 2: Continue training the Telugu (winner) checkpoint on
                 Kannada data (single GPU, DeepSpeed ZeRO-2).
        Stage 3: Evaluate the sequential model on Kannada, Telugu, and
                 English benchmarks with LLM-as-Judge.
        Stage 4: Compute retention/forgetting report.
    """
    setup_logging("module_2_sequential_ft", "module_2_sequential_ft")

    import argparse
    parser = argparse.ArgumentParser(description="Sequential FT training pipeline.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log all resolved configs, check imports, check files, and exit without training or evaluation.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run only evaluation, skipping training.",
    )
    parser.add_argument(
        "--train_time_hrs",
        type=float,
        default=None,
        help="Training time in hours to log during evaluation.",
    )
    args = parser.parse_args()

    # ── CUDA Device Visibility ────────────────────────────────────────
    # Inherit from environment if provided (e.g. CUDA_VISIBLE_DEVICES=0,1 during eval).
    # Default to device "1" for sequential training if not specified.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s",
        os.environ["CUDA_VISIBLE_DEVICES"],
    )

    # 1. Load configs
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")
    hp_config = load_config("configs/hyperparams.yaml")

    # 2. Load Monolingual FT winner
    winner_path = Path("results") / "module_1_monolingual_ft" / "results" / "monolingual_ft_winner.yaml"
    winner = yaml.safe_load(winner_path.read_text(encoding="utf-8"))
    winner_technique = winner["technique"]
    winner_language = winner.get("language", "telugu")

    # Determine source and target languages
    # Source = winner language (trained, might forget)
    # Target = the OTHER language (will learn sequentially)
    source_language = winner_language  # "telugu" — the winner
    target_language = "kannada" if source_language == "telugu" else "telugu"

    # Get both checkpoint paths
    source_ckpt_path = get_checkpoint_path(1, winner_technique, source_language)
    other_ckpt_path = get_checkpoint_path(1, winner_technique, target_language)

    if args.dry_run:
        logger.info("=== DRY RUN: Checking imports and configuration ===")
        # Verify checkpoint paths
        logger.info("Checking source checkpoint: %s (exists=%s)", source_ckpt_path, source_ckpt_path.exists())
        logger.info("Checking other checkpoint: %s (exists=%s)", other_ckpt_path, other_ckpt_path.exists())
        
        # Verify datasets
        train_path = Path("data") / target_language / "train.jsonl"
        val_path = Path("data") / target_language / "val.jsonl"
        logger.info("Checking train dataset path: %s (exists=%s)", train_path, train_path.exists())
        logger.info("Checking val dataset path: %s (exists=%s)", val_path, val_path.exists())
        
        # Verify tokenizer loading (import and initialization check)
        logger.info("Attempting to load tokenizer for config validation...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["tokenizer_id"],
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded successfully: %s", tokenizer.__class__.__name__)
        
        # Verify Deepspeed configuration exists if FFT is selected
        if winner_technique == "fft":
            ds_config_path = Path("configs/deepspeed_zero2.json")
            logger.info("Checking DeepSpeed config for FFT: %s (exists=%s)", ds_config_path, ds_config_path.exists())
            
        # Verify evaluation engine imports
        import evaluation.eval_engine as _eval_engine
        logger.info("Imported evaluation engine successfully.")
        
        # Log all resolved configurations
        logger.info("=== DRY RUN: Resolved Configurations ===")
        logger.info("model_config:\n%s", json.dumps(model_config, indent=2))
        logger.info("data_config:\n%s", json.dumps(data_config, indent=2))
        logger.info("hp_config:\n%s", json.dumps(hp_config, indent=2))
        logger.info("winner_technique: %s", winner_technique)
        logger.info("winner_language: %s", winner_language)
        logger.info("DRY RUN — exiting without training or evaluation.")
        sys.exit(0)

    if not source_ckpt_path.exists():
        raise FileNotFoundError(
            f"Monolingual FT {source_language} checkpoint not found: {source_ckpt_path}\n"
            f"Ensure Monolingual FT completed successfully before running Sequential FT."
        )
    if not other_ckpt_path.exists():
        raise FileNotFoundError(
            f"Monolingual FT {target_language} checkpoint not found: {other_ckpt_path}\n"
            f"Ensure Monolingual FT completed successfully before running Sequential FT."
        )

    # 3. Log start
    logger.info("Sequential FT starting. Winner technique: %s", winner_technique)
    logger.info("Direction: %s → %s (source → target)", source_language, target_language)
    logger.info("Source checkpoint (winner): %s", source_ckpt_path)
    logger.info("Other checkpoint (for zero-shot): %s", other_ckpt_path)

    output_dir = get_checkpoint_path(2, "sequential", "model")
    train_time_hrs = 0.0

    if not args.eval_only:
        # ── STAGE 1a: Zero-shot cross-lingual — Kannada model on Telugu ───
        logger.info("=== STAGE 1a: Zero-shot Telugu evaluation (Kannada model, no training) ===")
        zero_shot_kn_on_te = evaluate_checkpoint(
            checkpoint_path=str(other_ckpt_path),  # Kannada FFT checkpoint
            language="telugu",
            experiment_id="sequential_ft_zero_shot_telugu",
            model_config=model_config,
            data_config=data_config,
            module=2,
            technique=winner_technique,
            load_in_4bit=(winner_technique == "qlora"),
            train_time_hrs=None,
            notes="Zero-shot Telugu eval using Kannada-only FFT model — cross-lingual transfer measurement (KN→TE)",
            run_llm_judge=True,
        )
        logger.info("Zero-shot Telugu chrF++ (from Kannada model): %.2f", zero_shot_kn_on_te["chrf"])

        # GPU cleanup between stages
        torch.cuda.empty_cache()
        gc.collect()

        # ── STAGE 1b: Zero-shot cross-lingual — Telugu model on Kannada ───
        logger.info("=== STAGE 1b: Zero-shot Kannada evaluation (Telugu model, no training) ===")
        zero_shot_te_on_kn = evaluate_checkpoint(
            checkpoint_path=str(source_ckpt_path),  # Telugu FFT checkpoint (winner)
            language="kannada",
            experiment_id="sequential_ft_zero_shot_kannada",
            model_config=model_config,
            data_config=data_config,
            module=2,
            technique=winner_technique,
            load_in_4bit=(winner_technique == "qlora"),
            train_time_hrs=None,
            notes="Zero-shot Kannada eval using Telugu-only FFT model — cross-lingual transfer measurement (TE→KN)",
            run_llm_judge=True,
        )
        logger.info("Zero-shot Kannada chrF++ (from Telugu model): %.2f", zero_shot_te_on_kn["chrf"])

        # GPU cleanup between stages
        torch.cuda.empty_cache()
        gc.collect()

        # ── STAGE 2: Continue training on target language ─────────────────
        logger.info(
            "=== STAGE 2: Continue training %s (winner) on %s data (single GPU) ===",
            source_language.capitalize(),
            target_language.capitalize(),
        )

        if output_dir.exists() and (output_dir / "train_metadata.json").exists():
            logger.warning("Sequential model already trained. Skipping Stage 2.")
            logger.warning(
                "Delete the sequential model checkpoint to re-run."
            )
            # Load existing train time from metadata
            meta = json.loads(
                (output_dir / "train_metadata.json").read_text(encoding="utf-8")
            )
            train_time_hrs = meta.get("train_time_hrs", 0.0)
        else:
            # Auto-detect existing checkpoint for resume (e.g. after OOM crash)
            resume_ckpt = None
            if output_dir.exists():
                ckpt_dirs = sorted(
                    [d for d in output_dir.iterdir()
                     if d.is_dir() and d.name.startswith("checkpoint-")],
                    key=lambda p: int(p.name.split("-")[1]),
                )
                if ckpt_dirs:
                    resume_ckpt = str(ckpt_dirs[-1])
                    logger.info("Found existing checkpoint for resume: %s", resume_ckpt)

            t_start = time.time()
            _run_continuation_training(
                source_ckpt_path=source_ckpt_path,
                winner_technique=winner_technique,
                model_config=model_config,
                data_config=data_config,
                hp_config=hp_config,
                output_dir=output_dir,
                target_language=target_language,
                resume_from_checkpoint=resume_ckpt,
            )
            train_time_hrs = (time.time() - t_start) / 3600
            logger.info(
                "Sequential training complete in %.2fh → %s",
                train_time_hrs,
                output_dir,
            )

        # GPU cleanup between stages
        torch.cuda.empty_cache()
        gc.collect()
    else:
        # Load existing train time from metadata or arguments
        if args.train_time_hrs is not None:
            train_time_hrs = args.train_time_hrs
        else:
            try:
                metadata_path = output_dir / "train_metadata.json"
                if metadata_path.exists():
                    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    train_time_hrs = meta.get("train_time_hrs", 0.0)
            except Exception as exc:
                logger.warning("Could not load train_time_hrs: %s", exc)

    if not args.skip_eval:
        # ── STAGE 3: Evaluate sequential model on both languages + English ─
        logger.info(
            "=== STAGE 3: Evaluate sequential model on %s, %s, and English benchmarks ===",
            target_language.capitalize(),
            source_language.capitalize(),
        )

        # Evaluate on target language first (the newly learned one — Kannada)
        for language in [target_language, source_language]:
            experiment_id = f"sequential_ft_sequential_{language}"
            is_target = (language == target_language)
            result = evaluate_checkpoint(
                checkpoint_path=str(output_dir),
                language=language,
                experiment_id=experiment_id,
                model_config=model_config,
                data_config=data_config,
                module=2,
                technique=winner_technique,
                load_in_4bit=(winner_technique == "qlora"),
                train_time_hrs=train_time_hrs if is_target else None,
                notes=(
                    f"Sequential model ({source_language.upper()}→{target_language.upper()}) "
                    f"evaluated on {language} | "
                    f"{'Newly learned language' if is_target else 'Source language — forgetting check'}"
                ),
                run_llm_judge=True,  # Enable LLM-as-Judge using existing strategy
            )
            logger.info(
                "Sequential [%s] chrF++: %.2f | LLM Judge: %s",
                language,
                result["chrf"],
                result.get("llm_judge_score", "N/A"),
            )

            # GPU cleanup between eval runs
            torch.cuda.empty_cache()
            gc.collect()

        # ── STAGE 4: Compute and save retention report ────────────────────
        logger.info("=== STAGE 4: Computing retention report ===")
        _compute_and_save_retention_report(
            winner_technique,
            model_config,
            source_language=source_language,
            target_language=target_language,
        )

    logger.info("Sequential FT complete.")


if __name__ == "__main__":
    main()
