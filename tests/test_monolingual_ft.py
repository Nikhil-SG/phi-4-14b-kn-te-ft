"""Pytest validation gate for Monolingual FT.

Run after ``run_all_monolingual_ft.sh`` completes to verify that all 10
fine-tuning experiments produced valid checkpoints, metrics, and
training metadata.

Imports only: json, pathlib, math, yaml.  No torch, no transformers.

Usage:
    pytest tests/test_monolingual_ft.py -v
"""

import json
import math
from pathlib import Path

import yaml

TECHNIQUES = ["fft", "lora", "qlora", "dora", "ia3"]
LANGUAGES = ["kannada", "telugu"]
EXPECTED_IDS = {f"monolingual_ft_{t}_{l}" for t in TECHNIQUES for l in LANGUAGES}


# ---------------------------------------------------------------------------
# Test: all checkpoint directories exist with model files
# ---------------------------------------------------------------------------


def test_all_checkpoints_exist():
    """Verify checkpoint directories exist and contain model artefacts."""
    for technique in TECHNIQUES:
        for language in LANGUAGES:
            ckpt = Path(f"results/finetuned_models/module_1_monolingual_ft/{technique}_{language}")
            assert ckpt.exists(), f"Checkpoint dir missing: {ckpt}"
            has_files = any(
                f.suffix in {".json", ".safetensors", ".bin"}
                for f in ckpt.rglob("*")
            )
            assert has_files, f"Checkpoint dir empty: {ckpt}"


# ---------------------------------------------------------------------------
# Test: all 10 results logged in all_results.json
# ---------------------------------------------------------------------------


def test_all_results_logged():
    """Verify all 10 experiment results are in all_results.json."""
    results_path = Path("results/module_1_monolingual_ft/results/all_results.json")
    assert results_path.exists(), "results/all_results.json not found"

    results = json.loads(results_path.read_text(encoding="utf-8"))
    monolingual_ft = [r for r in results if r["module"] == 1]
    logged_ids = {r["experiment_id"] for r in monolingual_ft}
    missing = EXPECTED_IDS - logged_ids
    assert not missing, f"Missing experiment results: {missing}"


# ---------------------------------------------------------------------------
# Test: no NaN or invalid values in key metrics
# ---------------------------------------------------------------------------


def test_no_nan_in_metrics():
    """Verify key metrics are numeric, finite, and non-NaN."""
    results = json.loads(
        Path("results/module_1_monolingual_ft/results/all_results.json").read_text(encoding="utf-8")
    )
    monolingual_ft = [r for r in results if r["module"] == 1]

    for r in monolingual_ft:
        eid = r["experiment_id"]
        for key in ["chrf", "bleu", "peak_vram_gb", "trainable_params_M", "mmlu_accuracy", "mgsm_en_accuracy", "gpqa_accuracy"]:
            val = r.get(key)
            assert val is not None, f"{eid}: {key} is None"
            assert isinstance(val, (int, float)), (
                f"{eid}: {key} is not numeric (got {type(val).__name__})"
            )
            assert not math.isnan(val), f"{eid}: {key} is NaN"
            assert not math.isinf(val), f"{eid}: {key} is Inf"
        assert r["peak_vram_gb"] > 0, f"{eid}: peak_vram_gb must be > 0"


# ---------------------------------------------------------------------------
# Test: train_metadata.json exists for each run
# ---------------------------------------------------------------------------


def test_train_metadata_exists():
    """Verify training metadata is saved for every experiment."""
    for technique in TECHNIQUES:
        for language in LANGUAGES:
            metadata_path = Path(
                f"results/finetuned_models/module_1_monolingual_ft/{technique}_{language}/train_metadata.json"
            )
            assert metadata_path.exists(), f"Missing: {metadata_path}"

            meta = json.loads(metadata_path.read_text(encoding="utf-8"))

            assert "train_time_hrs" in meta, (
                f"{metadata_path}: missing train_time_hrs"
            )
            assert isinstance(meta["train_time_hrs"], float), (
                f"{metadata_path}: train_time_hrs must be float"
            )
            assert meta["train_time_hrs"] > 0, (
                f"{metadata_path}: train_time_hrs must be positive"
            )
            assert "final_train_loss" in meta, (
                f"{metadata_path}: missing final_train_loss"
            )
            assert math.isfinite(meta["final_train_loss"]), (
                f"{metadata_path}: final_train_loss is not finite"
            )


# ---------------------------------------------------------------------------
# Test: Pareto winner selection (write-through test)
# ---------------------------------------------------------------------------


def test_pareto_winner_written():
    """Select the best technique by chrF++/VRAM ratio and write results.

    This test is write-through: it both validates data integrity and
    produces ``results/monolingual_ft_winner.yaml`` as an artefact.
    """
    results = json.loads(
        Path("results/module_1_monolingual_ft/results/all_results.json").read_text(encoding="utf-8")
    )
    monolingual_ft = [r for r in results if r["module"] == 1]

    assert len({r["chrf"] for r in monolingual_ft}) >= 2, (
        "All chrF++ scores are identical — Pareto frontier degenerate"
    )

    # Compute Pareto score: higher chrF and lower VRAM is better.
    # Simple proxy: chrF / peak_vram_gb (higher is better).
    for r in monolingual_ft:
        r["_pareto_score"] = r["chrf"] / max(r["peak_vram_gb"], 0.01)

    winner = max(monolingual_ft, key=lambda r: r["_pareto_score"])

    winner_data = {
        "technique": winner["technique"],
        "language": winner["language"],
        "experiment_id": winner["experiment_id"],
        "chrf": winner["chrf"],
        "peak_vram_gb": winner["peak_vram_gb"],
        "pareto_score": round(winner["_pareto_score"], 4),
        "rationale": (
            "Selected by chrF++ / peak_vram_gb ratio across Monolingual FT runs"
        ),
    }

    winner_path = Path("results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml")
    winner_path.write_text(
        yaml.dump(winner_data, default_flow_style=False),
        encoding="utf-8",
    )

    assert winner_path.exists(), "results/monolingual_ft_winner.yaml was not written"
    loaded = yaml.safe_load(
        winner_path.read_text(encoding="utf-8")
    )
    assert loaded["technique"] in TECHNIQUES
    assert loaded["language"] in LANGUAGES
