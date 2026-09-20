"""Pytest validation gate for Prestudy.

Run immediately after ``prestudy/base_eval.py`` completes to verify that
all data files, schemas, quality thresholds, and evaluation results
meet the project's requirements.

Imports only stdlib + json + pathlib + math — no torch, no transformers.

Usage:
    pytest tests/test_prestudy.py -v
"""

import hashlib
import json
import math
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Test: data files exist
# ---------------------------------------------------------------------------


def test_data_files_exist():
    """Verify that train/val/test JSONL files exist for both languages."""
    for language in ["kannada", "telugu"]:
        for split in ["train", "val", "test"]:
            p = Path(f"data/{language}/{split}.jsonl")
            assert p.exists(), f"Missing: data/{language}/{split}.jsonl"


# ---------------------------------------------------------------------------
# Test: row counts meet minimums
# ---------------------------------------------------------------------------


def test_row_counts():
    """Verify minimum row counts for each split."""
    for language in ["kannada", "telugu"]:
        train_rows = _load_jsonl(Path(f"data/{language}/train.jsonl"))
        assert len(train_rows) >= 16000, (
            f"{language} train has only {len(train_rows)} rows (need >= 16000)"
        )

        val_rows = _load_jsonl(Path(f"data/{language}/val.jsonl"))
        assert len(val_rows) >= 900, (
            f"{language} val has only {len(val_rows)} rows (need >= 900)"
        )

        test_rows = _load_jsonl(Path(f"data/{language}/test.jsonl"))
        assert len(test_rows) >= 900, (
            f"{language} test has only {len(test_rows)} rows (need >= 900)"
        )


# ---------------------------------------------------------------------------
# Test: schema compliance
# ---------------------------------------------------------------------------


def test_schema_compliance():
    """Verify every row contains required keys with correct types."""
    required_keys = {
        "id",
        "language",
        "instruction",
        "input",
        "output",
        "subset",
        "script_purity",
        "char_count_instruction",
        "char_count_output",
    }

    for language in ["kannada", "telugu"]:
        for split in ["train", "val", "test"]:
            rows = _load_jsonl(Path(f"data/{language}/{split}.jsonl"))
            for row in rows[:10]:
                missing = required_keys - set(row.keys())
                assert not missing, (
                    f"{language}/{split} row missing keys: {missing}"
                )
                assert isinstance(row["script_purity"], float), (
                    f"script_purity must be float, got "
                    f"{type(row['script_purity']).__name__}"
                )
                assert 0.0 <= row["script_purity"] <= 1.0, (
                    f"script_purity {row['script_purity']} out of range [0, 1]"
                )


# ---------------------------------------------------------------------------
# Test: script purity threshold
# ---------------------------------------------------------------------------


def test_script_purity_threshold():
    """Verify mean script purity of training data meets the 0.80 threshold."""
    for language in ["kannada", "telugu"]:
        rows = _load_jsonl(Path(f"data/{language}/train.jsonl"))
        mean_purity = sum(r["script_purity"] for r in rows) / len(rows)
        assert mean_purity >= 0.80, (
            f"{language} mean train script purity {mean_purity:.3f} < 0.80"
        )


# ---------------------------------------------------------------------------
# Test: no train/test split leakage
# ---------------------------------------------------------------------------


def test_no_split_leakage():
    """Verify no duplicate instruction+output pairs between train and test."""
    for language in ["kannada", "telugu"]:

        def row_hash(r: dict) -> str:
            content = (r["instruction"] + r["output"]).encode("utf-8")
            return hashlib.sha256(content).hexdigest()

        train_rows = _load_jsonl(Path(f"data/{language}/train.jsonl"))
        test_rows = _load_jsonl(Path(f"data/{language}/test.jsonl"))

        train_hashes = {row_hash(r) for r in train_rows}
        test_hashes = {row_hash(r) for r in test_rows}

        overlap = train_hashes & test_hashes
        assert not overlap, (
            f"{language} has {len(overlap)} rows in both train and test"
        )


# ---------------------------------------------------------------------------
# Test: fertility results exist and are valid
# ---------------------------------------------------------------------------


def test_fertility_results_exist():
    """Verify fertility analysis results file exists with correct structure."""
    p = Path("results/module_0_prestudy/results/prestudy_fertility.json")
    assert p.exists(), "results/module_0_prestudy/results/prestudy_fertility.json not found"

    data = json.loads(p.read_text(encoding="utf-8"))

    for lang in ["kannada", "telugu", "english"]:
        assert lang in data, f"Missing '{lang}' in fertility results"
        assert "tokens_per_word" in data[lang], (
            f"Missing 'tokens_per_word' for {lang}"
        )
        assert data[lang]["tokens_per_word"]["mean"] > 0, (
            f"tokens_per_word mean for {lang} must be > 0"
        )

    assert data["kannada"]["tokens_per_word"]["mean"] > data["english"]["tokens_per_word"]["mean"], (
        "Kannada should have higher token fertility than English"
    )


# ---------------------------------------------------------------------------
# Test: base evaluation completed
# ---------------------------------------------------------------------------


def test_base_eval_done():
    """Verify base evaluation produced valid results for both languages."""
    results_path = Path("results/module_0_prestudy/results/all_results.json")
    assert results_path.exists(), "results/module_0_prestudy/results/all_results.json not found"

    results = json.loads(results_path.read_text(encoding="utf-8"))
    prestudy = [r for r in results if r["module"] == 0]

    assert len(prestudy) >= 2, (
        f"Expected >= 2 module-0 entries, got {len(prestudy)}"
    )

    languages_evaluated = {r["language"] for r in prestudy}
    assert "kannada" in languages_evaluated, (
        "Missing kannada in module-0 evaluation results"
    )
    assert "telugu" in languages_evaluated, (
        "Missing telugu in module-0 evaluation results"
    )

    for r in prestudy:
        assert isinstance(r["chrf"], float) and not math.isnan(r["chrf"]), (
            f"Invalid chrf in {r['experiment_id']}"
        )
        assert isinstance(r["perplexity"], float), (
            f"Invalid perplexity in {r['experiment_id']}"
        )
