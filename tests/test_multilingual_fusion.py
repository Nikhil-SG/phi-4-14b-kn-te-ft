"""Pytest validation gate for Multilingual Fusion.

Run after ``run_multilingual_fusion.sh`` completes to verify that joint training
and adapter merge experiments produced valid checkpoints, all 8
evaluation entries are logged, and visualization modules are importable.

Imports: json, pathlib, math, importlib.  No torch, no transformers.

Usage:
    pytest tests/test_multilingual_fusion.py -v
"""

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

# Check if joint training checkpoints were generated/run
has_joint = any(
    Path(f"results/finetuned_models/module_3_multilingual_fusion/joint_{strategy}").exists()
    for strategy in ["equal", "proportional"]
)

if has_joint:
    EXPECTED_MULTILINGUAL_FUSION_IDS = {
        "multilingual_fusion_joint_equal_kannada",
        "multilingual_fusion_joint_equal_telugu",
        "multilingual_fusion_joint_proportional_kannada",
        "multilingual_fusion_joint_proportional_telugu",
        "multilingual_fusion_merged_ties_kannada",
        "multilingual_fusion_merged_ties_telugu",
        "multilingual_fusion_merged_linear_kannada",
        "multilingual_fusion_merged_linear_telugu",
        "multilingual_fusion_monolingual_telugu_telugu",
        "multilingual_fusion_monolingual_telugu_kannada",
        "multilingual_fusion_monolingual_kannada_kannada",
        "multilingual_fusion_monolingual_kannada_telugu",
    }
else:
    EXPECTED_MULTILINGUAL_FUSION_IDS = {
        "multilingual_fusion_merged_ties_kannada",
        "multilingual_fusion_merged_ties_telugu",
        "multilingual_fusion_merged_linear_kannada",
        "multilingual_fusion_merged_linear_telugu",
        "multilingual_fusion_monolingual_telugu_telugu",
        "multilingual_fusion_monolingual_telugu_kannada",
        "multilingual_fusion_monolingual_kannada_kannada",
        "multilingual_fusion_monolingual_kannada_telugu",
    }


# ---------------------------------------------------------------------------
# Test: joint training checkpoints exist
# ---------------------------------------------------------------------------


def test_joint_checkpoints_exist():
    """Verify joint training checkpoint directories exist with metadata if joint SFT was run."""
    if not has_joint:
        pytest.skip("Skipping joint checkpoints check (Option B: Adapter Merge only was run)")

    for strategy in ["equal", "proportional"]:
        ckpt = Path(f"results/finetuned_models/module_3_multilingual_fusion/joint_{strategy}")
        assert ckpt.exists(), f"Missing: {ckpt}"
        assert (ckpt / "train_metadata.json").exists(), (
            f"train_metadata.json missing in {ckpt}"
        )


# ---------------------------------------------------------------------------
# Test: merged adapters exist with correct files
# ---------------------------------------------------------------------------


def test_merged_adapters_exist():
    """Verify merged adapter directories contain required PEFT files."""
    for merge_name in ["ties", "linear"]:
        merge_dir = Path(f"results/finetuned_models/module_3_multilingual_fusion/merged_{merge_name}")
        assert merge_dir.exists(), f"Missing: {merge_dir}"
        assert (merge_dir / "adapter_config.json").exists(), (
            f"adapter_config.json missing in {merge_dir}"
        )
        assert (merge_dir / "adapter_model.safetensors").exists(), (
            f"adapter_model.safetensors missing in {merge_dir}"
        )


# ---------------------------------------------------------------------------
# Test: all 8 Multilingual Fusion results logged
# ---------------------------------------------------------------------------


def test_multilingual_fusion_results_logged():
    """Verify all 8 Multilingual Fusion experiment results are in all_results.json."""
    results = json.loads(
        Path("results/module_3_multilingual_fusion/results/all_results.json").read_text(encoding="utf-8")
    )
    logged_ids = {r["experiment_id"] for r in results}
    missing = EXPECTED_MULTILINGUAL_FUSION_IDS - logged_ids
    assert not missing, f"Missing Multilingual Fusion experiment results: {missing}"


# ---------------------------------------------------------------------------
# Test: no NaN in Multilingual Fusion metrics
# ---------------------------------------------------------------------------


def test_no_nan_in_multilingual_fusion_metrics():
    """Verify key metrics are numeric and non-NaN for all Multilingual Fusion entries."""
    results = json.loads(
        Path("results/module_3_multilingual_fusion/results/all_results.json").read_text(encoding="utf-8")
    )
    multilingual_fusion = [r for r in results if r["module"] == 3]
    assert len(multilingual_fusion) == len(EXPECTED_MULTILINGUAL_FUSION_IDS), (
        f"Expected {len(EXPECTED_MULTILINGUAL_FUSION_IDS)} Multilingual Fusion entries, "
        f"got {len(multilingual_fusion)}"
    )

    for r in multilingual_fusion:
        eid = r["experiment_id"]
        for key in ["chrf", "bleu", "peak_vram_gb", "mmlu_accuracy", "mgsm_en_accuracy", "gpqa_accuracy"]:
            val = r.get(key)
            assert val is not None, f"{eid}: {key} is None"
            assert not math.isnan(float(val)), f"{eid}: {key} is NaN"


# ---------------------------------------------------------------------------
# Test: interference matrix data complete
# ---------------------------------------------------------------------------


def test_interference_matrix_data_complete():
    """Verify all experiment_ids needed by interference_matrix.py are present.

    Ensures the visualization can run without missing data errors.
    """
    def load_combined_results():
        res = []
        for module in ["module_0_prestudy", "module_1_monolingual_ft", "module_2_sequential_ft", "module_3_multilingual_fusion"]:
            p = Path("results") / module / "results" / "all_results.json"
            if p.exists():
                try:
                    res.extend(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return res
    results = load_combined_results()
    logged_ids = {r["experiment_id"] for r in results}

    # Minimum required for a meaningful interference matrix
    required_ids = {
        "prestudy_none_kannada",
        "prestudy_none_telugu",
        "sequential_ft_zero_shot_telugu",
        "sequential_ft_sequential_kannada",
        "sequential_ft_sequential_telugu",
    } | EXPECTED_MULTILINGUAL_FUSION_IDS

    missing = required_ids - logged_ids
    assert not missing, (
        f"Interference matrix will be incomplete. Missing entries: {missing}\n"
        f"Ensure all modules completed successfully."
    )


# ---------------------------------------------------------------------------
# Test: merge summary exists
# ---------------------------------------------------------------------------


def test_merge_summary_exists():
    """Verify the merge summary JSON file exists with required keys."""
    summary_path = Path("results/module_3_multilingual_fusion/results/merge_summary.json")
    assert summary_path.exists(), "results/multilingual_fusion/merge_summary.json not found"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "source_adapters" in summary
    assert "ties_dir" in summary
    assert "linear_dir" in summary


# ---------------------------------------------------------------------------
# Test: visualization scripts are importable
# ---------------------------------------------------------------------------


def test_viz_scripts_importable():
    """Verify visualize/ modules can be imported and have main().

    Does NOT run them (that requires results to be fully populated).
    """
    # Ensure project root is on path
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    scripts = [
        "visualize.pareto_frontier",
        "visualize.interference_matrix",
        "visualize.loss_curves",
        "visualize.fertility_plots",
    ]

    for script in scripts:
        try:
            mod = importlib.import_module(script)
            assert hasattr(mod, "main"), f"{script} missing main() function"
        except ImportError as e:
            pytest.fail(f"Cannot import {script}: {e}")
