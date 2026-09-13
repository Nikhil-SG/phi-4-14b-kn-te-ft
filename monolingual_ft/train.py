"""Unified CLI entry point for Monolingual FT training.

Dispatches to the appropriate trainer (FFT, LoRA, QLoRA, DoRA, IA³)
based on the ``--technique`` argument.  Each training run is
immediately followed by a full evaluation via the shared eval engine.

Usage:
    # Multi-GPU (FFT, LoRA, DoRA, IA³):
    accelerate launch --num_processes 2 --mixed_precision bf16 \
        monolingual_ft/train.py --technique lora --language kannada

    # Single-GPU (QLoRA):
    python monolingual_ft/train.py --technique qlora --language telugu

    # Config validation (no training):
    python monolingual_ft/train.py --technique lora --language kannada --dry_run
"""
# Redirect Hugging Face cache directories to local workspace folders to prevent global caching
import os
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(_project_root / "models" / ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(_project_root / "data" / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(_project_root / "models" / ".hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Monkey-patch: relax DeepSpeed's strict CUDA version check.
# The HPC has CUDA 12.4, but PyTorch was compiled with CUDA 12.1.
# They are ABI-compatible (same major version 12.x), but DeepSpeed
# demands an exact minor-version match.  This patch allows DeepSpeed's
# native C++ CPUAdam to compile and run, which is 10-50x faster than
# the pure-Python torch_adam fallback.
# ---------------------------------------------------------------------------
try:
    import deepspeed.ops.op_builder.builder as _ds_builder
    _ds_builder.assert_no_cuda_mismatch = lambda name="": None
except Exception:
    pass  # DeepSpeed not installed or different version — ignore


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



from evaluation.eval_engine import evaluate_checkpoint
from utils.checkpoint_utils import checkpoint_exists, get_checkpoint_path
from utils.data_utils import load_config
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Monolingual FT training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Monolingual FT: Fine-tune Phi-4 with a specified technique and language."
    )
    parser.add_argument(
        "--technique",
        required=True,
        choices=["fft", "lora", "qlora", "dora", "ia3"],
        help="Fine-tuning technique to use.",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=["kannada", "telugu"],
        help="Target language.",
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
    parser.add_argument(
        "--technique_config",
        default=None,
        help="Path to technique-specific config YAML. "
        "Defaults to configs/techniques/{technique}.yaml.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log all resolved configs and exit without training.",
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
    return parser.parse_args()


def main() -> None:
    """Dispatch training and evaluation for a single Monolingual FT experiment."""
    setup_logging("module_1_monolingual_ft", "module_1_monolingual_ft")
    args = parse_args()

    # --- Load all configs --------------------------------------------------
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    hp_config = load_config(args.hp_config)
    technique_config_path = args.technique_config or str(
        Path("configs") / "techniques" / f"{args.technique}.yaml"
    )
    technique_cfg = load_config(technique_config_path)

    # --- Dry run -----------------------------------------------------------
    if args.dry_run:
        logger.info("=== DRY RUN: Resolved Configurations ===")
        logger.info("model_config:\n%s", json.dumps(model_config, indent=2))
        logger.info("data_config:\n%s", json.dumps(data_config, indent=2))
        logger.info("hp_config:\n%s", json.dumps(hp_config, indent=2))
        logger.info("technique_cfg:\n%s", json.dumps(technique_cfg, indent=2))
        logger.info("DRY RUN — exiting without training.")
        sys.exit(0)

    # --- Pre-flight checks -------------------------------------------------
    experiment_id = f"monolingual_ft_{args.technique}_{args.language}"
    output_dir = get_checkpoint_path(1, args.technique, args.language)
    logger.info("Starting: %s → %s", experiment_id, output_dir)

    if not args.eval_only:
        if checkpoint_exists(1, args.technique, args.language):
            logger.warning("Checkpoint already exists at %s", output_dir)
            logger.warning("Delete it manually to re-run. Skipping.")
            sys.exit(0)

        # --- Auto-resume: detect interrupted runs with intermediate checkpoints -
        resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt is None and output_dir.exists():
            from utils.checkpoint_utils import cleanup_corrupted_checkpoints, get_latest_checkpoint

            # Remove any partially-written checkpoints (e.g. from mid-save crashes)
            # before looking for a valid resume point
            removed = cleanup_corrupted_checkpoints(output_dir)
            if removed:
                logger.warning(
                    "Cleaned up %d corrupted checkpoint(s): %s",
                    len(removed),
                    [p.name for p in removed],
                )

            latest = get_latest_checkpoint(output_dir)
            if latest is not None:
                resume_ckpt = str(latest)
                logger.info(
                    "Interrupted run detected — resuming from checkpoint: %s",
                    resume_ckpt,
                )

        # --- Dispatch to trainer -----------------------------------------------
        t_start = time.time()

        if args.technique == "fft":
            from monolingual_ft.trainers.fft_trainer import run
        elif args.technique == "qlora":
            from monolingual_ft.trainers.qlora_trainer import run
        else:
            from monolingual_ft.trainers.peft_trainer import run

        run(
            language=args.language,
            model_config=model_config,
            data_config=data_config,
            hp_config=hp_config,
            technique_cfg=technique_cfg,
            resume_from_checkpoint=resume_ckpt,
        )

        t_end = time.time()
        train_time_hrs = (t_end - t_start) / 3600
    else:
        train_time_hrs = args.train_time_hrs

    # --- Evaluate checkpoint -----------------------------------------------
    # In multi-GPU / distributed setup (like FFT under accelerate), only run
    # evaluation on rank 0 to prevent redundant work and out-of-memory errors
    # when loading the 24B judge model.
    if not args.skip_eval:
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            logger.info("Training complete. Pausing 5 seconds to clear memory...")
            time.sleep(5)
            
            # Enable all 3 layers of evaluation (including LLM-as-Judge and Prompt Sensitivity) for all techniques
            run_full_eval = True
            
            logger.info("Starting evaluation for %s (LLM-as-Judge=%s, Prompt Sensitivity=%s) ...", 
                        experiment_id, run_full_eval, run_full_eval)
            evaluate_checkpoint(
                checkpoint_path=str(output_dir),
                language=args.language,
                experiment_id=experiment_id,
                model_config=model_config,
                data_config=data_config,
                module=1,
                technique=args.technique,
                load_in_4bit=(args.technique == "qlora"),
                train_time_hrs=train_time_hrs,
                notes=f"Monolingual SFT {args.technique} on {args.language}",
                run_llm_judge=run_full_eval,
                run_prompt_sensitivity=run_full_eval,
            )
        else:
            logger.info("Process rank %d complete. Skipping evaluation (handled by rank 0).", local_rank)

    if train_time_hrs is not None:
        logger.info(
            "Monolingual FT complete: %s in %.2fh", experiment_id, train_time_hrs
        )
    else:
        logger.info(
            "Monolingual FT complete: %s (evaluation complete)", experiment_id
        )


if __name__ == "__main__":
    main()
