"""Merge Monolingual FT LoRA adapters for Kannada and Telugu.

Implements TIES-Merge and linear averaging as weight-space operations.
No training occurs — merged adapters are saved in PEFT format and
evaluated on both languages.

Usage:
    python multilingual_fusion/adapter_merge.py
"""

import datetime
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_engine import evaluate_checkpoint
from utils.data_utils import load_config
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _load_adapter_state_dict(
    adapter_dir: Path,
) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights as a state dict.

    Args:
        adapter_dir: Path to the adapter directory containing
            ``adapter_model.safetensors`` or ``adapter_model.bin``.

    Returns:
        A dict mapping parameter names to CPU tensors.

    Raises:
        FileNotFoundError: If neither safetensors nor bin file exists.
    """
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device="cpu")
        logger.info(
            "Loaded adapter from %s (%d params)", safetensors_path, len(state)
        )
        return state

    if bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        logger.info(
            "Loaded adapter from %s (%d params)", bin_path, len(state)
        )
        return state

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_dir}. "
        f"Expected adapter_model.safetensors or adapter_model.bin."
    )


def ties_merge(
    adapter_state_dicts: list[dict[str, torch.Tensor]],
    density: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Merge adapter state dicts using the TIES-Merge algorithm.

    Reference: "TIES-Merging: Resolving Interference When Merging Models"
    (Yadav et al., 2023).

    For LoRA adapters, the adapter weights *are* the task vectors
    (initialised near zero; updates captured in A and B matrices).

    Args:
        adapter_state_dicts: List of adapter state dicts to merge.
        density: Fraction of weights to keep during the trim step.
            Default is 0.5 (top 50% by magnitude).

    Returns:
        A merged state dict with the same keys as the union of inputs.
    """
    # Collect all keys across adapters
    all_keys: set[str] = set()
    for sd in adapter_state_dicts:
        all_keys.update(sd.keys())

    merged: dict[str, torch.Tensor] = {}

    for key in sorted(all_keys):
        # Gather tensors for this key (skip adapters missing it)
        tensors = [sd[key] for sd in adapter_state_dicts if key in sd]

        if len(tensors) == 1:
            # Only one adapter has this key — include without modification
            merged[key] = tensors[0].clone()
            continue

        original_dtype = tensors[0].dtype

        # Skip empty tensors
        if tensors[0].numel() == 0:
            merged[key] = tensors[0].clone()
            continue

        # Convert to float32 for math
        float_tensors = [t.float() for t in tensors]

        # Step 1 — Trim: keep top `density` fraction by magnitude
        trimmed: list[torch.Tensor] = []
        for t_k in float_tensors:
            flat = t_k.flatten().abs()
            threshold = torch.quantile(flat, 1.0 - density)
            mask = t_k.abs() >= threshold
            trimmed.append(t_k * mask.float())

        # Step 2 — Elect sign
        sign_sum = torch.zeros_like(float_tensors[0])
        for t in trimmed:
            sign_sum += t
        elected_sign = torch.sign(sign_sum)
        # Break ties positive
        elected_sign[elected_sign == 0] = 1.0

        # Step 3 — Disjoint merge
        masked_list: list[torch.Tensor] = []
        for t_k in trimmed:
            agreement_mask = (torch.sign(t_k) == elected_sign).float()
            masked_list.append(t_k * agreement_mask)

        stacked = torch.stack(masked_list)  # [n_adapters, ...]
        contributor_count = (stacked != 0).float().sum(dim=0).clamp(min=1.0)
        merged_tensor = stacked.sum(dim=0) / contributor_count

        # Convert back to original dtype
        merged[key] = merged_tensor.to(original_dtype)

    logger.info("TIES-Merge complete: %d parameters merged.", len(merged))
    return merged


def linear_merge(
    adapter_state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Merge adapter state dicts using weighted linear averaging.

    Args:
        adapter_state_dicts: List of adapter state dicts to merge.
        weights: Per-adapter weights summing to 1.0. If None, uniform
            averaging is used.

    Returns:
        A merged state dict.

    Raises:
        ValueError: If weights do not sum to 1.0 (±1e-6).
    """
    n = len(adapter_state_dicts)

    if weights is None:
        weights = [1.0 / n] * n

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(
            f"Merge weights must sum to 1.0, got {sum(weights):.6f}"
        )

    # Collect all keys
    all_keys: set[str] = set()
    for sd in adapter_state_dicts:
        all_keys.update(sd.keys())

    merged: dict[str, torch.Tensor] = {}

    for key in sorted(all_keys):
        original_dtype = None
        accumulator: torch.Tensor | None = None

        for i, sd in enumerate(adapter_state_dicts):
            if key not in sd:
                continue

            t = sd[key].float()
            if original_dtype is None:
                original_dtype = sd[key].dtype

            if accumulator is None:
                accumulator = torch.zeros_like(t)

            accumulator += t * weights[i]

        if accumulator is not None and original_dtype is not None:
            merged[key] = accumulator.to(original_dtype)

    logger.info("Linear merge complete: %d parameters merged.", len(merged))
    return merged


def _save_merged_adapter(
    merged_state: dict[str, torch.Tensor],
    source_adapter_dir: Path,
    output_dir: Path,
    model_config: dict,
) -> None:
    """Save a merged adapter in PEFT format.

    Copies ``adapter_config.json`` from the source adapter and saves
    the merged weights as ``adapter_model.safetensors``.

    Args:
        merged_state: Merged adapter state dict.
        source_adapter_dir: Path to a source adapter directory to copy
            ``adapter_config.json`` from.
        output_dir: Directory to save the merged adapter to.
        model_config: Parsed ``configs/model.yaml`` (reserved for
            future use).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter_config.json from source
    src_config = source_adapter_dir / "adapter_config.json"
    dst_config = output_dir / "adapter_config.json"
    shutil.copy2(str(src_config), str(dst_config))

    # Save merged weights
    from safetensors.torch import save_file

    save_file(merged_state, str(output_dir / "adapter_model.safetensors"))

    logger.info("Merged adapter saved to %s", output_dir)


def main() -> None:
    """Run adapter merge experiments (TIES-Merge and linear average).

    Loads Monolingual FT LoRA adapters for Kannada and Telugu, merges them
    using two strategies, saves merged adapters, and evaluates on
    both languages.
    """
    setup_logging("module_3_multilingual_fusion", "module_3_multilingual_fusion")
    # 1. Load configs
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")

    # 2. Verify Monolingual FT LoRA adapter paths
    from utils.checkpoint_utils import get_checkpoint_path
    kn_adapter = get_checkpoint_path(1, "lora", "kannada")
    te_adapter = get_checkpoint_path(1, "lora", "telugu")

    for p in [kn_adapter, te_adapter]:
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Monolingual FT LoRA adapter not found: {p}\n"
                f"Monolingual FT must complete with LoRA technique before adapter merging."
            )

    # 3. Load adapter state dicts
    logger.info("Loading adapter: %s", kn_adapter)
    kn_state = _load_adapter_state_dict(kn_adapter)
    logger.info("Loading adapter: %s", te_adapter)
    te_state = _load_adapter_state_dict(te_adapter)

    # 4. TIES-Merge
    logger.info("Running TIES-Merge (density=0.5) ...")
    ties_state = ties_merge([kn_state, te_state], density=0.5)
    ties_dir = get_checkpoint_path(3, "merged", "ties")
    _save_merged_adapter(ties_state, kn_adapter, ties_dir, model_config)
    logger.info("TIES-Merge complete.")

    # 5. Linear merge
    logger.info("Running linear average merge ...")
    linear_state = linear_merge([kn_state, te_state], weights=[0.5, 0.5])
    linear_dir = get_checkpoint_path(3, "merged", "linear")
    _save_merged_adapter(linear_state, kn_adapter, linear_dir, model_config)
    logger.info("Linear merge complete.")

    # 5.5. Evaluate monolingual adapters on both languages as pre-merge baselines
    for adapter_name, adapter_dir, adapter_lang in [
        ("telugu", te_adapter, "telugu"),
        ("kannada", kn_adapter, "kannada")
    ]:
        for eval_lang in ["kannada", "telugu"]:
            experiment_id = f"multilingual_fusion_monolingual_{adapter_name}_{eval_lang}"
            logger.info("Evaluating monolingual baseline %s ...", experiment_id)
            evaluate_checkpoint(
                checkpoint_path=str(adapter_dir),
                language=eval_lang,
                experiment_id=experiment_id,
                model_config=model_config,
                data_config=data_config,
                module=3,
                technique=f"monolingual_{adapter_name}",
                load_in_4bit=False,
                train_time_hrs=None,
                notes=f"Monolingual {adapter_name.upper()} LoRA baseline evaluated on {eval_lang}",
                run_llm_judge=True,
                run_prompt_sensitivity=True,
            )
            torch.cuda.empty_cache()
            gc.collect()

    # 6. Evaluate both merges on both languages
    for merge_name, merge_dir in [("ties", ties_dir), ("linear", linear_dir)]:
        for language in ["kannada", "telugu"]:
            experiment_id = f"multilingual_fusion_merged_{merge_name}_{language}"
            logger.info("Evaluating %s ...", experiment_id)
            evaluate_checkpoint(
                checkpoint_path=str(merge_dir),
                language=language,
                experiment_id=experiment_id,
                model_config=model_config,
                data_config=data_config,
                module=3,
                technique=f"merge_{merge_name}",
                load_in_4bit=False,
                train_time_hrs=None,
                notes=f"{merge_name.upper()}-merged LoRA (KN⊕TE) evaluated on {language}",
                run_llm_judge=True,
                run_prompt_sensitivity=True,
            )
            torch.cuda.empty_cache()
            gc.collect()

    # 7. Save merge summary
    summary = {
        "source_adapters": [str(kn_adapter), str(te_adapter)],
        "ties_dir": str(ties_dir),
        "linear_dir": str(linear_dir),
        "density": 0.5,
        "computed_at": datetime.datetime.utcnow().isoformat(),
    }
    summary_path = Path("results/module_3_multilingual_fusion/results/merge_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Adapter merge experiments complete.")


if __name__ == "__main__":
    main()
