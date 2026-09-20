"""Pytest validation gate for Sequential FT (Redesigned Module 2).

Run after ``sequential_train.py`` completes to verify that:
- Bidirectional zero-shot evaluations are logged (KN→TE and TE→KN)
- Sequential training produced valid checkpoints
- Post-FT evaluations on both languages are logged with LLM Judge
- The retention report is well-formed with correct direction

Imports: json, pathlib, math, yaml.  No torch, no transformers.

Usage:
    pytest tests/test_sequential_ft.py -v
"""

import json
import math
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Test: sequential model checkpoint exists
# ---------------------------------------------------------------------------


def test_sequential_checkpoint_exists():
    """Verify the sequential model checkpoint directory exists with model files."""
    ckpt = Path("results/finetuned_models/module_2_sequential_ft/sequential_model")
    assert ckpt.exists(), "results/sequential_ft/sequential_model/ not found"
    has_files = any(
        f.suffix in {".json", ".safetensors", ".bin"} for f in ckpt.rglob("*")
    )
    assert has_files, "Sequential model checkpoint directory is empty"
    assert (ckpt / "train_metadata.json").exists(), (
        "train_metadata.json missing — training may have crashed"
    )


# ---------------------------------------------------------------------------
# Test: train_metadata.json is valid (updated for Telugu→Kannada direction)
# ---------------------------------------------------------------------------


def test_train_metadata_valid():
    """Verify training metadata has correct Sequential FT schema and values.

    The redesigned Module 2 trains Telugu (winner) → Kannada, so:
    - training_language should be "kannada" (the target)
    - starting_checkpoint should point to the Telugu FFT checkpoint
    - direction should be "telugu_to_kannada"
    """
    metadata_path = Path("results/finetuned_models/module_2_sequential_ft/sequential_model/train_metadata.json")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert meta["module"] == 2
    assert meta["training_language"] == "kannada", (
        f"Expected training_language='kannada', got '{meta['training_language']}'"
    )
    assert meta["direction"] == "telugu_to_kannada", (
        f"Expected direction='telugu_to_kannada', got '{meta.get('direction')}'"
    )
    assert "starting_checkpoint" in meta
    assert "telugu" in meta["starting_checkpoint"].lower() or "fft_telugu" in meta["starting_checkpoint"].lower(), (
        f"Starting checkpoint should reference Telugu: {meta['starting_checkpoint']}"
    )
    assert isinstance(meta["train_time_hrs"], float) and meta["train_time_hrs"] > 0
    assert math.isfinite(meta["final_train_loss"])
    assert meta.get("single_gpu") is True, "Expected single_gpu=True"


# ---------------------------------------------------------------------------
# Test: bidirectional zero-shot measurements logged
# ---------------------------------------------------------------------------


def test_zero_shot_measurement_logged():
    """Verify BOTH zero-shot evaluation entries are logged correctly.

    Stage 1a: Kannada model evaluated on Telugu → sequential_ft_zero_shot_telugu
    Stage 1b: Telugu model evaluated on Kannada → sequential_ft_zero_shot_kannada
    """
    results = json.loads(
        Path("results/module_2_sequential_ft/results/all_results.json").read_text(encoding="utf-8")
    )

    # Check zero-shot Telugu (Kannada model → Telugu eval)
    zero_shot_te = [
        r for r in results if r["experiment_id"] == "sequential_ft_zero_shot_telugu"
    ]
    assert len(zero_shot_te) == 1, (
        f"Expected 1 zero-shot Telugu entry, found {len(zero_shot_te)}"
    )
    zs_te = zero_shot_te[0]
    assert zs_te["module"] == 2
    assert zs_te["language"] == "telugu"
    assert isinstance(zs_te["chrf"], float) and not math.isnan(zs_te["chrf"])
    assert zs_te["train_time_hrs"] is None, (
        "Zero-shot entry should have train_time_hrs=null"
    )

    # Check zero-shot Kannada (Telugu model → Kannada eval)
    zero_shot_kn = [
        r for r in results if r["experiment_id"] == "sequential_ft_zero_shot_kannada"
    ]
    assert len(zero_shot_kn) == 1, (
        f"Expected 1 zero-shot Kannada entry, found {len(zero_shot_kn)}"
    )
    zs_kn = zero_shot_kn[0]
    assert zs_kn["module"] == 2
    assert zs_kn["language"] == "kannada"
    assert isinstance(zs_kn["chrf"], float) and not math.isnan(zs_kn["chrf"])
    assert zs_kn["train_time_hrs"] is None, (
        "Zero-shot entry should have train_time_hrs=null"
    )


# ---------------------------------------------------------------------------
# Test: sequential evaluations logged (Kannada + Telugu + English + LLM Judge)
# ---------------------------------------------------------------------------


def test_sequential_evals_logged():
    """Verify sequential model evaluations for both languages are logged.

    Both entries should have:
    - Translation metrics (chrF, BLEU, etc.)
    - English retention benchmarks (MMLU, MGSM, GPQA)
    - LLM-as-Judge scores
    """
    results = json.loads(
        Path("results/module_2_sequential_ft/results/all_results.json").read_text(encoding="utf-8")
    )

    for lang in ["kannada", "telugu"]:
        eid = f"sequential_ft_sequential_{lang}"
        matches = [r for r in results if r["experiment_id"] == eid]
        assert len(matches) == 1, (
            f"Expected 1 entry for {eid}, found {len(matches)}"
        )

        r = matches[0]
        assert r["module"] == 2
        assert isinstance(r["chrf"], float) and not math.isnan(r["chrf"])
        assert r["peak_vram_gb"] > 0

        # English retention benchmarks must be present
        for key in ["mmlu_accuracy", "mgsm_en_accuracy", "gpqa_accuracy"]:
            val = r.get(key)
            assert val is not None, f"{eid}: {key} is None"
            assert isinstance(val, (int, float)) and not math.isnan(val)

        # LLM-as-Judge score must be present
        judge_score = r.get("llm_judge_score")
        assert judge_score is not None, f"{eid}: llm_judge_score is None"
        assert isinstance(judge_score, (int, float)) and not math.isnan(judge_score)


# ---------------------------------------------------------------------------
# Test: retention report is valid (updated for Telugu→Kannada direction)
# ---------------------------------------------------------------------------


def test_retention_report_valid():
    """Verify the retention report exists and has well-formed values.

    The report should reflect the Telugu→Kannada direction:
    - retention_rate measures Telugu retention after Kannada training
    - direction should be "telugu_to_kannada"
    """
    report_path = Path("results/module_2_sequential_ft/results/retention_report.json")
    assert report_path.exists(), "retention_report.json not found"

    report = json.loads(report_path.read_text(encoding="utf-8"))

    required_keys = {
        "winner_technique",
        "direction",
        "source_language",
        "target_language",
        "monolingual_ft_telugu_chrf",
        "monolingual_ft_kannada_chrf",
        "sequential_telugu_chrf",
        "sequential_kannada_chrf",
        "retention_rate",
        "forgetting_delta",
        "bwt",
        "fwt",
        "computed_at",
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"retention_report.json missing keys: {missing}"

    assert report["direction"] == "telugu_to_kannada", (
        f"Expected direction='telugu_to_kannada', got '{report['direction']}'"
    )
    assert report["source_language"] == "telugu"
    assert report["target_language"] == "kannada"

    if report["retention_rate"] is not None:
        assert 0.0 < report["retention_rate"] <= 1.5, (
            f"Retention rate {report['retention_rate']} outside expected "
            f"range (0, 1.5]"
        )

    if report["forgetting_delta"] is not None:
        assert math.isfinite(report["forgetting_delta"])


# ---------------------------------------------------------------------------
# Test: cross-lingual transfer observable
# ---------------------------------------------------------------------------


def test_cross_lingual_transfer_observable():
    """Sanity check that zero-shot cross-lingual performance is not catastrophic.

    Both zero-shot evaluations should produce non-trivial chrF++ scores,
    indicating that the Dravidian-family cross-lingual transfer exists.
    """
    seq_results_path = Path("results/module_2_sequential_ft/results/all_results.json")
    results = json.loads(seq_results_path.read_text(encoding="utf-8")) if seq_results_path.exists() else []

    prestudy_results_path = Path("results/module_0_prestudy/results/all_results.json")
    prestudy_results = json.loads(prestudy_results_path.read_text(encoding="utf-8")) if prestudy_results_path.exists() else []

    # Check zero-shot Telugu (Kannada model → Telugu)
    baseline_te = next(
        (
            r
            for r in prestudy_results
            if r["module"] == 0 and r["language"] == "telugu"
        ),
        None,
    )
    zero_shot_te = next(
        (
            r
            for r in results
            if r["experiment_id"] == "sequential_ft_zero_shot_telugu"
        ),
        None,
    )

    if baseline_te is not None and zero_shot_te is not None:
        # Allow 10% degradation tolerance (different inference context)
        tolerance = 0.90
        assert zero_shot_te["chrf"] >= baseline_te["chrf"] * tolerance, (
            f"Zero-shot Telugu ({zero_shot_te['chrf']:.2f}) is significantly "
            f"below raw baseline ({baseline_te['chrf']:.2f}). "
            f"Check eval pipeline."
        )

    # Check zero-shot Kannada (Telugu model → Kannada)
    baseline_kn = next(
        (
            r
            for r in prestudy_results
            if r["module"] == 0 and r["language"] == "kannada"
        ),
        None,
    )
    zero_shot_kn = next(
        (
            r
            for r in results
            if r["experiment_id"] == "sequential_ft_zero_shot_kannada"
        ),
        None,
    )

    if baseline_kn is not None and zero_shot_kn is not None:
        tolerance = 0.90
        assert zero_shot_kn["chrf"] >= baseline_kn["chrf"] * tolerance, (
            f"Zero-shot Kannada ({zero_shot_kn['chrf']:.2f}) is significantly "
            f"below raw baseline ({baseline_kn['chrf']:.2f}). "
            f"Check eval pipeline."
        )
