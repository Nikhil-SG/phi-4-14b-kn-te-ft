"""QLoRA trainer — 4-bit NF4 quantized base model + LoRA adapters.

Intentionally runs on a single GPU (no accelerate launch):
    python monolingual_ft/train.py --technique qlora --language {language}

Usage:
    This module is not run directly. It is imported and called by
    ``monolingual_ft/train.py``.
"""

import json
import os
import time
from pathlib import Path

import datasets
import torch
import transformers
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
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
def _setup_mlflow(language: str, hp_config: dict, model_config: dict):
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
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = "Module1_MonolingualFT"
            tags = {
                "module": "module_1_monolingual_ft",
                "technique": "qlora",
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
            if db_path.exists():
                import shutil
                backup = db_path.with_suffix(f".db.bak.{int(time.time())}")
                shutil.move(str(db_path), str(backup))
                logger.warning("Backed up corrupted mlflow.db → %s  (%s)", backup, db_err)
            _init_mlflow(tracking_uri)

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Module1_MonolingualFT"

        tags = {
            "module": "module_1_monolingual_ft",
            "technique": "qlora",
            "language": language
        }
        os.environ["MLFLOW_TAGS"] = json.dumps(tags)

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
    """Run QLoRA fine-tuning with 4-bit NF4 quantization + LoRA adapters.

    Args:
        language: Target language (``"kannada"`` or ``"telugu"``).
        model_config: Parsed ``configs/model.yaml``.
        data_config: Parsed ``configs/data.yaml``.
        hp_config: Parsed ``configs/hyperparams.yaml``.
        technique_cfg: Parsed ``configs/techniques/qlora.yaml``.
        resume_from_checkpoint: Optional path to resume from.

    Returns:
        The output directory path as a string.
    """
    # 1. Set seed
    transformers.set_seed(hp_config["seed"])

    # 2-3. Setup
    output_dir = get_checkpoint_path(1, "qlora", language)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("QLoRA trainer: output_dir=%s", output_dir)

    # 3b. MLflow setup
    mlflow_active = _setup_mlflow(language, hp_config, model_config)

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_id"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Build quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=hp_config["qlora"]["quant_type"],
        bnb_4bit_double_quant=hp_config["qlora"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=hp_config["qlora"]["double_quant"],
    )

    # 6. Load model with quantization
    from utils.gpu_utils import configure_gpu_settings
    primary_device_id, max_memory = configure_gpu_settings()

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    # prepare_model_for_kbit_training handles gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.enable_input_require_grads()
    reset_vram_peak()
    logger.info("Quantized model loaded. Params: %s", count_trainable_params(model))

    # 7. Build LoRA config
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
    logger.info("LoRA config for QLoRA: %s", peft_config)

    # 8. Prepare datasets
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

    # 9. Build TrainingArguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=model_config["max_context_length"],
        num_train_epochs=hp_config["epochs"],
        per_device_train_batch_size=hp_config["per_device_train_batch_size"],
        per_device_eval_batch_size=hp_config["per_device_train_batch_size"],
        gradient_accumulation_steps=hp_config["gradient_accumulation_steps"],
        learning_rate=hp_config["learning_rates"]["qlora"],
        lr_scheduler_type=hp_config["lr_scheduler_type"],
        warmup_ratio=hp_config["warmup_ratio"],
        bf16=True,
        fp16=False,
        logging_steps=hp_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=hp_config["eval_steps"],
        save_strategy="steps",
        save_steps=hp_config["eval_steps"],
        save_total_limit=hp_config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        optim="paged_adamw_8bit",
        max_grad_norm=hp_config["max_grad_norm"],
        weight_decay=hp_config["weight_decay"],
        seed=hp_config["seed"],
        data_seed=hp_config["seed"],
        dataloader_num_workers=hp_config["dataloader_num_workers"],
        dataloader_pin_memory=True,
        report_to="mlflow" if mlflow_active else "none",
        run_name=f"QLoRA_{language.capitalize()}",
        gradient_checkpointing=False,
    )

    # 10. Install crash forensics
    install_signal_handlers(output_dir=str(output_dir))

    # 11. Build SFTTrainer (pass peft_config — SFTTrainer wraps internally)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        peft_config=peft_config,
        callbacks=[
            TQDMProgressCallback("qlora", language),
            MemoryMonitorCallback(log_every_steps=50, output_dir=str(output_dir)),
        ],
    )

    # Log trainable parameters
    trainer.model.print_trainable_parameters()
    logger.info(
        "Trainable params after QLoRA: %s", count_trainable_params(trainer.model)
    )

    # 11. Train
    logger.info("Starting QLoRA training for %s ...", language)
    t0 = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    t1 = time.time()
    train_time_hrs = (t1 - t0) / 3600
    logger.info("Training complete in %.2fh", train_time_hrs)

    # 12. Synchronize all ranks before cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 13. Save tokenizer, copy best checkpoint, and save metadata on rank 0
    best_ckpt = None
    if local_rank == 0:
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Tokenizer saved to %s", output_dir)

        # Find best checkpoint for weight consolidation
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ckpt is None:
            ckpt_dirs = sorted(output_dir.glob("checkpoint-*"),
                               key=lambda p: int(p.name.split("-")[1]))
            best_ckpt = str(ckpt_dirs[-1]) if ckpt_dirs else None
            logger.warning("No best_model_checkpoint tracked. Using latest: %s", best_ckpt)
        else:
            logger.info("Best model checkpoint: %s", best_ckpt)

        cleanup_checkpoints = False
        if best_ckpt is not None:
            logger.info("Moving adapter weights from %s to parent %s ...", best_ckpt, output_dir)
            try:
                import shutil
                best_ckpt_path = Path(best_ckpt)
                patterns = [
                    "adapter_model*",
                    "adapter_config.json",
                    "config.json",
                    "README.md"
                ]
                moved_files = []
                for pattern in patterns:
                    for f in best_ckpt_path.glob(pattern):
                        dest = output_dir / f.name
                        logger.info("Moving %s → %s", f, dest)
                        shutil.move(str(f), str(dest))
                        moved_files.append(dest)
                logger.info("Successfully moved %d adapter files to parent directory.", len(moved_files))
                cleanup_checkpoints = True
            except Exception as exc:
                logger.error("Failed to move adapter weights to parent directory: %s", exc)

        # Save train_metadata.json
        final_loss = train_result.training_loss
        eval_loss = None
        if trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "eval_loss" in entry:
                    eval_loss = entry["eval_loss"]
                    break

        metadata = {
            "technique": "qlora",
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

    # 14. Log peak VRAM and finalize MLflow
    peak_vram = get_peak_vram_gb()
    logger.info("Peak VRAM after QLoRA training: %.2f GB", peak_vram)

    if mlflow_active:
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            # Find the latest run in the experiment to log extra metrics
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

    return str(output_dir)
