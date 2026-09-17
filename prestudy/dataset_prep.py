"""Download, filter, deduplicate, and split indic-align for Kannada and Telugu.

Loads only the high-hygiene subsets (Dolly_T and OpenAssistant_T) from
ai4bharat/indic-align, samples proportionally from each subset,
and outputs JSONL files to ``data/{language}/`` with train, val, and test
splits conforming to the project's data schema.  Also writes a
``metadata.json`` per language summarising row counts and quality stats.

Usage:
    python prestudy/dataset_prep.py
"""

import ast
import datetime
import hashlib
import json
import random
import sys
from pathlib import Path

import datasets
import numpy

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import compute_script_purity, load_config, save_jsonl
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _sample_proportionally(
    subset_rows: dict[str, list[dict]],
    allowed_subsets: list[str],
    target_n: int,
    seed: int,
) -> list[dict]:
    """Sample target_n rows preserving each subset's natural proportion.

    Args:
        subset_rows: Mapping from subset name to list of row dicts.
        allowed_subsets: List of subset names to include.
        target_n: Total rows to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, shuffled, total length <= target_n.
    """
    total_available = sum(len(subset_rows.get(s, [])) for s in allowed_subsets)
    if total_available == 0:
        raise ValueError(
            f"No rows found for subsets {allowed_subsets}")

    sampled: list[dict] = []
    for subset_name in allowed_subsets:
        rows = subset_rows.get(subset_name, [])
        proportion = len(rows) / total_available
        n_from_subset = min(round(proportion * target_n), len(rows))
        sampled.extend(rows[:n_from_subset])
        logger.info(
            "  Sampled %d from %s (%.1f%% of available)",
            n_from_subset, subset_name, proportion * 100,
        )

    # Final shuffle to interleave subsets
    rng = random.Random(seed)
    rng.shuffle(sampled)
    return sampled[:target_n]


def process_language(language: str, data_config: dict) -> dict:
    """Load ai4bharat/indic-align, filter to high-hygiene subsets only
    (Dolly_T and OpenAssistant_T), sample proportionally from each
    subset, split into train/val/test, and save as JSONL.

    Args:
        language: ``"kannada"`` or ``"telugu"``.
        data_config: Parsed ``configs/data.yaml`` dict.

    Returns:
        A metadata dict summarising the processed dataset, also saved
        to ``data/{language}/metadata.json``.

    Raises:
        ValueError: If the raw dataset columns cannot be mapped or
            no data is retrieved.
    """
    lang_cfg = data_config["datasets"][language]
    lang_code = lang_cfg["language_code"]
    unicode_start = lang_cfg["script_unicode_start"]
    unicode_end = lang_cfg["script_unicode_end"]
    seed = data_config["seed"]
    allowed_subsets = lang_cfg["allowed_subsets"]

    # --- 1. Load raw dataset -----------------------------------------------
    model_cfg = load_config("configs/model.yaml", download_model=False)
    hf_token = model_cfg.get("hf_token", "")

    # Map language names to FLORES-style column names used in indic-align
    lang_to_col = {
        "kannada": "kan_Knda",
        "telugu": "tel_Telu",
    }
    lang_col = lang_to_col.get(language)

    # Collect raw rows per subset for proportional sampling
    subset_rows: dict[str, list[dict]] = {s: [] for s in allowed_subsets}

    for subset in allowed_subsets:
        logger.info("Loading subset %s for %s...", subset, language)
        try:
            ds_subset = datasets.load_dataset(
                lang_cfg["hf_path"],
                name=subset,
                split="train",
                token=hf_token if hf_token else None,
            )

            logger.info(
                "Subset %s: %d rows, columns=%s",
                subset, len(ds_subset), ds_subset.column_names,
            )

            # Extract conversations from the language-specific column
            for row in ds_subset:
                instruction = ""
                output = ""

                if lang_col in row and row[lang_col] is not None:
                    turns = row[lang_col]
                    if isinstance(turns, str):
                        try:
                            turns = ast.literal_eval(turns)
                        except Exception:
                            continue
                    if (isinstance(turns, list)
                            and len(turns) > 0
                            and isinstance(turns[0], list)
                            and len(turns[0]) >= 2):
                        instruction = turns[0][0]
                        output = turns[0][1]

                if instruction and output:
                    subset_rows[subset].append({
                        "instruction": instruction,
                        "output": output,
                        "input": "",
                        "subset": subset,
                    })
        except Exception as e:
            logger.error("Failed to load/parse subset %s: %s", subset, e)

    total_raw = sum(len(v) for v in subset_rows.values())
    if total_raw == 0:
        raise ValueError(f"No data could be retrieved for language: {language}")

    for s in allowed_subsets:
        logger.info("  %s: %d rows extracted", s, len(subset_rows[s]))
    logger.info("Total raw rows for %s: %d", language, total_raw)

    # Combine into a single HF Dataset for filtering
    all_rows = []
    for s in allowed_subsets:
        all_rows.extend(subset_rows[s])
    ds = datasets.Dataset.from_list(all_rows)

    logger.info(
        "Raw dataset: %d rows, columns=%s", len(ds), ds.column_names
    )

    # --- 2. Column mapping (already normalised above) ----------------------
    instr_col = "instruction"
    out_col = "output"
    input_col = "input"

    # --- 3. Filter by character length -------------------------------------
    min_ic = lang_cfg["min_instruction_chars"]
    max_ic = lang_cfg["max_instruction_chars"]
    min_oc = lang_cfg["min_output_chars"]
    max_oc = lang_cfg["max_output_chars"]

    ds = ds.filter(
        lambda x: (
            min_ic <= len(x[instr_col]) <= max_ic
            and min_oc <= len(x[out_col]) <= max_oc
        ),
        desc="Filtering by char length",
    )
    logger.info("After length filter: %d rows", len(ds))

    # --- 4. Filter by script purity >= 0.70 --------------------------------
    ds = ds.filter(
        lambda x: compute_script_purity(
            x[instr_col], unicode_start, unicode_end
        ) >= 0.70,
        desc="Filtering by script purity >= 0.70",
    )
    logger.info("After script purity filter: %d rows", len(ds))

    # --- 5. Deduplicate ----------------------------------------------------
    seen_hashes: set[str] = set()
    keep_indices: list[int] = []

    instructions_list = ds[instr_col]
    outputs_list = ds[out_col]

    for idx, (instruction_text, output_text) in enumerate(zip(instructions_list, outputs_list)):
        content = (instruction_text + output_text).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            keep_indices.append(idx)

    n_duplicates = len(ds) - len(keep_indices)
    logger.info("Duplicates removed: %d (keeping %d)", n_duplicates, len(keep_indices))
    ds = ds.select(keep_indices)

    # --- 6. Check if we have enough data -----------------------------------
    target_train = lang_cfg["target_train_samples"]
    target_val = lang_cfg["target_val_samples"]
    target_test = lang_cfg["target_test_samples"]
    total_needed = target_train + target_val + target_test

    if len(ds) < total_needed:
        logger.warning(
            "%s: only %d rows available (need %d). "
            "Adjusting splits proportionally (90/5/5).",
            language,
            len(ds),
            total_needed,
        )
        available = len(ds)
        target_train = int(available * 0.90)
        target_val = int(available * 0.05)
        target_test = available - target_train - target_val

    # --- 7. Shuffle --------------------------------------------------------
    ds = ds.shuffle(seed=seed)

    # --- 8. Proportional sampling from each subset -------------------------
    # Re-group the filtered+deduped rows by subset for proportional sampling
    filtered_subset_rows: dict[str, list[dict]] = {s: [] for s in allowed_subsets}
    for row in ds:
        s = row.get("subset", "")
        if s in filtered_subset_rows:
            filtered_subset_rows[s].append(dict(row))

    sampled_rows = _sample_proportionally(
        subset_rows=filtered_subset_rows,
        allowed_subsets=allowed_subsets,
        target_n=total_needed,
        seed=seed,
    )

    # --- 9. Split ----------------------------------------------------------
    n_train = min(target_train, len(sampled_rows))
    n_val = min(target_val, len(sampled_rows) - n_train)
    n_test = min(target_test, len(sampled_rows) - n_train - n_val)

    train_slice = sampled_rows[:n_train]
    val_slice = sampled_rows[n_train: n_train + n_val]
    test_slice = sampled_rows[n_train + n_val: n_train + n_val + n_test]

    # --- 10. Build rows with enriched fields -------------------------------
    def build_rows(rows: list[dict], split_name: str) -> list[dict]:
        """Enrich raw dicts with id, purity, and char counts."""
        enriched: list[dict] = []
        for i, row in enumerate(rows):
            instruction_text = row[instr_col]
            output_text = row[out_col]
            input_text = row.get(input_col, "")

            purity = compute_script_purity(
                instruction_text, unicode_start, unicode_end
            )
            enriched.append({
                "id": f"{lang_code}_{split_name}_{i:05d}",
                "language": lang_code,
                "instruction": instruction_text,
                "input": input_text if input_text else "",
                "output": output_text,
                "subset": row.get("subset", ""),
                "script_purity": round(purity, 4),
                "char_count_instruction": len(instruction_text),
                "char_count_output": len(output_text),
            })
        return enriched

    train_rows = build_rows(train_slice, "train")
    val_rows = build_rows(val_slice, "val")
    test_rows = build_rows(test_slice, "test")

    # --- 11. Save JSONL files ----------------------------------------------
    output_dir = Path(data_config["data_output_dir"]) / language
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_rows, str(output_dir / "train.jsonl"))
    save_jsonl(val_rows, str(output_dir / "val.jsonl"))
    save_jsonl(test_rows, str(output_dir / "test.jsonl"))

    logger.info(
        "%s: saved %d train, %d val, %d test to %s",
        language,
        len(train_rows),
        len(val_rows),
        len(test_rows),
        output_dir,
    )

    # --- 12. Build and save metadata ---------------------------------------
    train_purities = [r["script_purity"] for r in train_rows]

    # Per-subset counts in training set
    subset_counts: dict[str, int] = {}
    for s in allowed_subsets:
        subset_counts[s] = sum(1 for r in train_rows if r.get("subset") == s)

    metadata = {
        "language": language,
        "language_code": lang_code,
        "source_hf_path": lang_cfg["hf_path"],
        "allowed_subsets": allowed_subsets,
        "subset_counts_train": subset_counts,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "mean_script_purity_train": round(
            float(numpy.mean(train_purities)) if train_purities else 0.0, 4
        ),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Metadata saved to %s", metadata_path)
    logger.info(
        "%s: mean_purity=%.4f, subset_dist=%s",
        language,
        metadata["mean_script_purity_train"],
        subset_counts,
    )

    return metadata


def main() -> None:
    """Entry point: prepare datasets for all languages.

    Loads configuration, sets random seeds, and processes each language
    defined in ``configs/data.yaml``.
    """
    setup_logging("module_0_prestudy", "module_0_prestudy")
    # Load configs (trigger download of all models/datasets on first run if missing).
    _ = load_config("configs/model.yaml", download_model=True)
    data_config = load_config("configs/data.yaml")

    # Set seeds for reproducibility.
    seed = data_config["seed"]
    random.seed(seed)
    numpy.random.seed(seed)

    for language in ["kannada", "telugu"]:
        logger.info("Processing %s ...", language)
        metadata = process_language(language, data_config)
        logger.info(
            "%s summary: train=%d, val=%d, test=%d, mean_purity=%.4f",
            language,
            metadata["train_count"],
            metadata["val_count"],
            metadata["test_count"],
            metadata["mean_script_purity_train"],
        )

    logger.info("Prestudy dataset preparation complete.")


if __name__ == "__main__":
    main()
