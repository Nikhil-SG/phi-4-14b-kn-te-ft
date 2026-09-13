"""Full parameter fine-tuning trainer using DeepSpeed ZeRO-3.

Expects to be launched via:
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
        monolingual_ft/train.py --technique fft --language {language}

Usage:
    This module is not run directly. It is imported and called by
    ``monolingual_ft/train.py``.
"""

import gc
import json
import os
import time
from pathlib import Path

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer

from evaluation.metrics.system import count_trainable_params, get_peak_vram_gb, reset_vram_peak
from utils.checkpoint_utils import get_checkpoint_path
from utils.data_utils import format_phi4_prompt, load_jsonl
from utils.logging_utils import get_logger
from utils.training_monitor import MemoryMonitorCallback, install_signal_handlers

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom TQDM callback for epoch-level + step-level progress bars
# ---------------------------------------------------------------------------
class TQDMProgressCallback(TrainerCallback):
    """Dual tqdm progress bars: epoch-level and step-level with live loss."""

    def __init__(self, technique: str, language: str):
        self.technique = technique.upper()
        self.language = language.capitalize()

    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(
            total=int(args.num_train_epochs),
            desc=f"{self.technique} {self.language} [Epochs]",
            unit="epoch",
            position=0,
        )
        self.step_bar = tqdm(
            total=state.max_steps,
            desc=f"{self.technique} {self.language} [Steps]",
            unit="step",
            position=1,
            leave=False,
        )
        self._current_epoch = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_bar.update(1)
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self.step_bar.set_postfix(
                    loss=f"{last['loss']:.4f}",
                    lr=f"{last.get('learning_rate', 0):.2e}",
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        self._current_epoch += 1
        self.epoch_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.step_bar.close()
        self.epoch_bar.close()


# ---------------------------------------------------------------------------
# MLflow setup helper
# ---------------------------------------------------------------------------
def _setup_mlflow(technique: str, language: str, hp_config: dict, model_config: dict):
    """Configure MLflow environment variables for HuggingFace SFTTrainer.

    Returns:
        True if MLflow was successfully configured, False otherwise.
    """
    try:
        import mlflow
        import json

        db_path = Path("mlflow.db").resolve()
        tracking_uri = f"sqlite:///{db_path}"

        # ── Guard: only rank-0 should touch the DB ──────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            # Non-primary processes just set env vars (no DB init)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = "Module1_MonolingualFT"
            tags = {
                "module": "module_1_monolingual_ft",
                "technique": technique,
                "language": language,
            }
            os.environ["MLFLOW_TAGS"] = json.dumps(tags)
            if "MLFLOW_RUN_ID" in os.environ:
                del os.environ["MLFLOW_RUN_ID"]
            return True

        # ── Rank-0: initialise / repair the database ────────────────
        def _init_mlflow(uri: str):
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("Module1_MonolingualFT")

        try:
            _init_mlflow(tracking_uri)
        except Exception as db_err:
            # Corrupted DB (e.g. leftover Alembic tables from a crash).
            # Back up the old file and start fresh.
            if db_path.exists():
                import shutil
                backup = db_path.with_suffix(f".db.bak.{int(time.time())}")
                shutil.move(str(db_path), str(backup))
                logger.warning("Backed up corrupted mlflow.db → %s  (%s)", backup, db_err)
            _init_mlflow(tracking_uri)

        # Set environment variables that MLflowCallback reads
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Module1_MonolingualFT"

        # MLflowCallback decodes MLFLOW_TAGS from a JSON string and logs them automatically
        tags = {
            "module": "module_1_monolingual_ft",
            "technique": technique,
            "language": language
        }
        os.environ["MLFLOW_TAGS"] = json.dumps(tags)

        # Clear any existing MLFLOW_RUN_ID to prevent "run already active" conflicts
        if "MLFLOW_RUN_ID" in os.environ:
            del os.environ["MLFLOW_RUN_ID"]

        logger.info("MLflow configured via environment variables for Trainer integration.")
        return True
    except Exception as exc:
        logger.warning("MLflow setup failed (training will continue without MLflow): %s", exc)
        return False


def run(
    language: str,
    model_config: dict,
    data_config: dict,
    hp_config: dict,
    technique_cfg: dict,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Run full parameter fine-tuning with DeepSpeed ZeRO-3.

    Args:
        language: Target language (``"kannada"`` or ``"telugu"``).
        model_config: Parsed ``configs/model.yaml``.
        data_config: Parsed ``configs/data.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.
        technique_cfg: Parsed ``configs/techniques/fft.yaml``.
        resume_from_checkpoint: Optional path to resume from.

    Returns:
        The output directory path as a string.
    """
    # 1. Set seed
    transformers.set_seed(hp_config["seed"])

    # 2-3. Setup
    output_dir = get_checkpoint_path(1, "fft", language)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("FFT trainer: output_dir=%s", output_dir)

    # 3b. MLflow setup
    mlflow_active = _setup_mlflow("fft", language, hp_config, model_config)

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_id"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Load model — strategy depends on world size:
    #
    #  • Single-GPU (world_size=1): Use ZeRO-2 + CPU optimizer offload.
    #    zero.Init() is a ZeRO-3 primitive that shards parameters *across ranks*.
    #    On one GPU it has no benefit and actively breaks from_pretrained because
    #    HuggingFace creates meta-device tensors inside zero.Init() context and
    #    then tries to materialise them — raising:
    #      NotImplementedError: Cannot copy out of meta tensor; no data!
    #    Fix: load normally (no zero.Init), pass ZeRO-2 config to SFTConfig.
    #
    #  • Multi-GPU (world_size>1): Use ZeRO-3 with zero.Init() as before so
    #    each rank only holds 1/N of the model in GPU memory.
    import json

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Resolve adjusted gradient accumulation steps to maintain effective batch size of 64
    base_grad_accum = hp_config["gradient_accumulation_steps"]
    target_effective_batch = hp_config["per_device_train_batch_size"] * 2 * base_grad_accum
    adjusted_grad_accum = target_effective_batch // (hp_config["per_device_train_batch_size"] * world_size)

    # Choose the right DeepSpeed config based on world size
    if world_size == 1:
        # Single-GPU: use ZeRO-2 (CPU optimizer offload, no parameter sharding)
        zero2_config_path = str(Path(technique_cfg["deepspeed_config"]).parent / "deepspeed_zero2.json")
        if not Path(zero2_config_path).exists():
            # Fallback: derive ZeRO-2 config from ZeRO-3 config dynamically
            with open(technique_cfg["deepspeed_config"], "r") as f:
                ds_config_dict = json.load(f)
            ds_config_dict["zero_optimization"]["stage"] = 2
            # Remove ZeRO-3-specific keys that are invalid for stage 2
            for key in ["sub_group_size", "stage3_gather_16bit_weights_on_model_save"]:
                ds_config_dict["zero_optimization"].pop(key, None)
        else:
            with open(zero2_config_path, "r") as f:
                ds_config_dict = json.load(f)
        logger.info("Single-GPU mode: using ZeRO-2 (no zero.Init, no parameter sharding)")
    else:
        # Multi-GPU: use ZeRO-3 with full parameter sharding
        with open(technique_cfg["deepspeed_config"], "r") as f:
            ds_config_dict = json.load(f)
        logger.info("Multi-GPU mode (world_size=%d): using ZeRO-3 with parameter sharding", world_size)

    # Pre-resolve "auto" scalars so zero.Init / HF Trainer don't crash on them
    if ds_config_dict.get("train_micro_batch_size_per_gpu") == "auto":
        ds_config_dict["train_micro_batch_size_per_gpu"] = hp_config["per_device_train_batch_size"]
    if ds_config_dict.get("gradient_accumulation_steps") == "auto":
        ds_config_dict["gradient_accumulation_steps"] = adjusted_grad_accum
    # Also resolve optimizer param "auto" values (zero.Init requires concrete numbers)
    if "optimizer" in ds_config_dict and "params" in ds_config_dict["optimizer"]:
        opt_params = ds_config_dict["optimizer"]["params"]
        _lr = hp_config["learning_rates"]["fft"]
        _auto_map = {
            "lr": _lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": hp_config.get("weight_decay", 0.01),
        }
        for k, default_v in _auto_map.items():
            if opt_params.get(k) == "auto":
                opt_params[k] = default_v

    if world_size == 1:
        # ── Single-GPU path: plain from_pretrained, no zero.Init ──────────
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True,
            attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        )
        logger.info("Model loaded normally (single-GPU, ZeRO-2 will handle optimizer offload).")
    else:
        # ── Multi-GPU path: shard parameters with zero.Init ───────────────
        import deepspeed as _ds
        with _ds.zero.Init(config_dict_or_path=ds_config_dict):
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_id"],
                torch_dtype=torch.bfloat16,
                use_cache=False,
                trust_remote_code=True,
                attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
            )
        logger.info("Model loaded with ZeRO-3 partitioning across %d GPUs.", world_size)

    model.gradient_checkpointing_enable()
    gc.collect()  # Free temporary init buffers before optimizer allocation
    reset_vram_peak()
    param_info = count_trainable_params(model)
    logger.info("Model ready. world_size=%d, ZeRO stage=%d, Params: %s",
                world_size, ds_config_dict["zero_optimization"]["stage"], param_info)

    # 6. Prepare datasets
    train_rows = load_jsonl(str(Path("data") / language / "train.jsonl"))
    val_rows = load_jsonl(str(Path("data") / language / "val.jsonl"))

    train_ds = datasets.Dataset.from_list(
        [
            {
                "text": format_phi4_prompt(
                    r["instruction"], r["input"], r["output"]
                )
            }
            for r in train_rows
        ]
    )
    val_ds = datasets.Dataset.from_list(
        [
            {
                "text": format_phi4_prompt(
                    r["instruction"], r["input"], r["output"]
                )
            }
            for r in val_rows
        ]
    )
    logger.info("Datasets: train=%d, val=%d", len(train_ds), len(val_ds))

    # 7. Build TrainingArguments
    if adjusted_grad_accum != base_grad_accum:
        logger.info("Adjusted gradient_accumulation_steps: %d → %d (world_size=%d, effective_batch=%d)",
                     base_grad_accum, adjusted_grad_accum, world_size, target_effective_batch)

    # Allow epoch override via FFT_EPOCHS env var (single-GPU runs need fewer
    # epochs to fit within SLURM walltime)
    epochs = int(os.environ.get("FFT_EPOCHS", hp_config["epochs"]))
    if epochs != hp_config["epochs"]:
        logger.info("FFT_EPOCHS override: %d → %d epochs", hp_config["epochs"], epochs)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=model_config["max_context_length"],
        num_train_epochs=epochs,
        per_device_train_batch_size=hp_config["per_device_train_batch_size"],
        per_device_eval_batch_size=hp_config["per_device_train_batch_size"],
        gradient_accumulation_steps=adjusted_grad_accum,
        learning_rate=hp_config["learning_rates"]["fft"],
        lr_scheduler_type=hp_config["lr_scheduler_type"],
        warmup_ratio=hp_config["warmup_ratio"],
        bf16=hp_config["bf16"],
        fp16=hp_config["fp16"],
        logging_steps=hp_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=140,
        save_strategy="steps",
        save_steps=140,
        save_total_limit=hp_config["save_total_limit"],
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
        deepspeed=ds_config_dict,
        max_grad_norm=hp_config["max_grad_norm"],
        weight_decay=hp_config["weight_decay"],
        seed=hp_config["seed"],
        data_seed=hp_config["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_compile=False,
        report_to="mlflow" if (mlflow_active and local_rank == 0) else "none",
        run_name=f"FFT_{language.capitalize()}",
        gradient_checkpointing=True,
    )

    # 8. Install crash forensics (signal handlers + memory monitoring)
    install_signal_handlers(output_dir=str(output_dir))

    # 9. Build SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        callbacks=[
            TQDMProgressCallback("fft", language),
            MemoryMonitorCallback(log_every_steps=50, output_dir=str(output_dir)),
        ],
    )

    # 9. Train
    logger.info("Starting FFT training for %s ...", language)
    t0 = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    t1 = time.time()
    train_time_hrs = (t1 - t0) / 3600
    logger.info("Training complete in %.2fh", train_time_hrs)

    # 10. Save tokenizer, model config, and generation config
    tokenizer.save_pretrained(str(output_dir))
    model.config.save_pretrained(str(output_dir))
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.save_pretrained(str(output_dir))
    logger.info("Tokenizer and config saved to %s", output_dir)

    # 11. Find best checkpoint for weight consolidation
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt is None:
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
        best_ckpt = str(ckpt_dirs[-1]) if ckpt_dirs else None
        logger.warning("No best_model_checkpoint tracked. Using latest: %s", best_ckpt)
    else:
        logger.info("Best model checkpoint: %s", best_ckpt)

    # 12. Save train_metadata.json
    final_loss = train_result.training_loss
    eval_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break

    metadata = {
        "technique": "fft",
        "language": language,
        "train_time_hrs": round(train_time_hrs, 4),
        "final_train_loss": round(final_loss, 6),
        "final_eval_loss": round(eval_loss, 6) if eval_loss is not None else None,
        "best_checkpoint": best_ckpt,
    }

    try:
        metadata_path = output_dir / "train_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Train metadata saved to %s", metadata_path)
    except Exception as exc:
        logger.error("Failed to save train_metadata.json: %s", exc)

    # 13. Log peak VRAM and finalize MLflow (rank 0 only)
    peak_vram = get_peak_vram_gb()
    logger.info("Peak VRAM after FFT training: %.2f GB", peak_vram)

    if mlflow_active and local_rank == 0:
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("Module1_MonolingualFT")
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attribute.start_time DESC"],
                    max_results=1
                )
                if runs:
                    latest_run_id = runs[0].info.run_id
                    client.log_metric(latest_run_id, "train_time_hrs", round(train_time_hrs, 4))
                    client.log_metric(latest_run_id, "peak_vram_gb", peak_vram)
                    logger.info("Logged extra metrics to MLflow run: %s", latest_run_id)
                else:
                    logger.warning("No runs found in experiment to log extra metrics.")
            else:
                logger.warning("Experiment Module1_MonolingualFT not found.")
        except Exception as exc:
            logger.warning("MLflow finalization failed: %s", exc)

    # 14. Synchronize all ranks before cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 15. Free all training memory before (optional) weight consolidation
    optimizer_gb = 14.6e9 * 12 / world_size / 1e9
    logger.info("Freeing training memory (~%.0f GB optimizer states per rank)...", optimizer_gb)
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Training memory freed.")

    # 16. Post-training checkpoint handling:
    #
    #  ZeRO-2 (single-GPU): Trainer already writes standard HuggingFace safetensors /
    #    pytorch_model.bin inside each checkpoint-NNNN directory.  No conversion
    #    is needed — the eval engine can load best_ckpt directly.
    #
    #  ZeRO-3 (multi-GPU): Trainer writes per-rank shard files (zero_pp_rank_*.pt).
    #    These must be consolidated into a single pytorch_model.bin before eval.
    if local_rank == 0 and best_ckpt is not None:
        cleanup_checkpoints = False
        if world_size == 1:
            # ZeRO-2: checkpoint is already a complete, loadable HF checkpoint.
            logger.info(
                "ZeRO-2 checkpoint is already in HF format. Moving weights from %s to parent %s ...",
                best_ckpt, output_dir
            )
            try:
                import shutil
                best_ckpt_path = Path(best_ckpt)
                weight_patterns = [
                    "pytorch_model*.bin",
                    "model*.safetensors",
                    "*.index.json",
                    "config.json",
                    "generation_config.json"
                ]
                moved_files = []
                for pattern in weight_patterns:
                    for f in best_ckpt_path.glob(pattern):
                        dest = output_dir / f.name
                        logger.info("Moving %s → %s", f, dest)
                        shutil.move(str(f), str(dest))
                        moved_files.append(dest)
                logger.info("Successfully moved %d model files to parent directory.", len(moved_files))
                cleanup_checkpoints = True
            except Exception as exc:
                logger.error("Failed to move checkpoint weights to parent directory: %s", exc)
        else:
            # ZeRO-3: consolidate sharded weights into pytorch_model.bin
            # Requires ~58 GB CPU RAM (14.6B × 4 bytes FP32).
            try:
                logger.info(
                    "Consolidating ZeRO-3 shards from %s → %s/pytorch_model.bin ...",
                    best_ckpt, output_dir
                )
                from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
                convert_zero_checkpoint_to_fp32_state_dict(
                    best_ckpt,
                    str(output_dir / "pytorch_model.bin"),
                )
                logger.info("Weight consolidation complete: %s", output_dir / "pytorch_model.bin")
                cleanup_checkpoints = True
            except Exception as exc:
                logger.error("Weight consolidation failed: %s", exc)
                logger.error("Manual consolidation command:")
                logger.error(
                    "  python -c \"from deepspeed.utils.zero_to_fp32 import "
                    "convert_zero_checkpoint_to_fp32_state_dict; "
                    "convert_zero_checkpoint_to_fp32_state_dict('%s', '%s/pytorch_model.bin')\"",
                    best_ckpt, output_dir
                )

        # Remove all checkpoint folders to free up disk space
        if cleanup_checkpoints:
            try:
                import shutil
                for child in output_dir.iterdir():
                    if child.is_dir() and child.name.startswith("checkpoint-"):
                        logger.info("Removing checkpoint directory to free disk space: %s", child)
                        shutil.rmtree(child, ignore_errors=True)
                logger.info("Checkpoint cleanup complete. Evaluators can now safely load from parent directory.")
            except Exception as exc:
                logger.error("Failed to clean up checkpoint directories: %s", exc)

    # 17. Return
    return str(output_dir)
