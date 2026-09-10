"""English retention benchmarks — catastrophic forgetting detection.

Measures whether fine-tuning on Kannada/Telugu has degraded the
model's English capabilities by running three standard benchmarks:

- **MMLU** (5-shot): 500 stratified questions across 57 subjects
- **MGSM** (8-shot): 250 multilingual grade-school math (English)
- **GPQA** (0-shot): 448 graduate-level science questions

Results are compared against the base Phi-4 published scores:
    MMLU=84.8, MGSM=80.6, GPQA=56.1

Usage:
    from evaluation.metrics.english_retention import compute_english_retention

    result = compute_english_retention(model, tokenizer, model_config, data_config)
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# MMLU Subject → Category mapping (for per-category reporting)
# ──────────────────────────────────────────────────────────────────────────────
_MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    ],
}

# Reverse lookup: subject → category
_SUBJECT_TO_CATEGORY = {}
for cat, subjects in _MMLU_CATEGORIES.items():
    for subj in subjects:
        _SUBJECT_TO_CATEGORY[subj] = cat


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def compute_english_retention(
    model: Any,
    tokenizer: Any,
    model_config: dict,
    data_config: dict,
    device: str = "cuda",
) -> dict:
    """Run MMLU + MGSM + GPQA English benchmarks on a fine-tuned model.

    Each sub-benchmark is wrapped in try/except so a failure in one
    does not block the others.

    Args:
        model: Fine-tuned causal LM (on GPU, in eval mode).
        tokenizer: The corresponding tokenizer.
        model_config: Parsed ``configs/model.yaml`` dict.
        data_config: Parsed ``configs/data.yaml`` dict.
        device: Torch device string.

    Returns:
        Dict with keys ``mmlu_accuracy``, ``mmlu_category_scores``,
        ``mgsm_en_accuracy``, ``gpqa_accuracy``.  Any sub-benchmark
        that fails returns ``None`` for its keys.
    """
    eng_cfg = data_config.get("english_retention", {})
    hf_token = model_config.get("hf_token", "") or None
    max_len = model_config.get("max_context_length", 1024)

    result = {
        "mmlu_accuracy": None,
        "mmlu_category_scores": None,
        "mgsm_en_accuracy": None,
        "gpqa_accuracy": None,
    }

    # ── MMLU ──────────────────────────────────────────────────────────
    try:
        mmlu_cfg = eng_cfg.get("mmlu", {})
        mmlu_result = _evaluate_mmlu(
            model, tokenizer, max_len, device,
            data_dir=str(Path(data_config.get("data_output_dir", "data")) / "english" / "mmlu"),
            n_samples=mmlu_cfg.get("n_samples", 500),
            n_few_shot=mmlu_cfg.get("n_few_shot", 5),
            hf_token=hf_token,
            hf_path=mmlu_cfg.get("hf_path", "cais/mmlu"),
            config_name=mmlu_cfg.get("config_name", "all"),
        )
        result["mmlu_accuracy"] = mmlu_result["accuracy"]
        result["mmlu_category_scores"] = mmlu_result["category_scores"]
        logger.info(
            "MMLU: %.2f%% overall (%d/%d correct)",
            mmlu_result["accuracy"] * 100,
            mmlu_result["n_correct"],
            mmlu_result["n_total"],
        )
    except Exception as exc:
        logger.error("MMLU evaluation failed: %s", exc, exc_info=True)

    # ── MGSM ─────────────────────────────────────────────────────────
    try:
        mgsm_cfg = eng_cfg.get("mgsm", {})
        mgsm_result = _evaluate_mgsm(
            model, tokenizer, max_len, device,
            data_dir=str(Path(data_config.get("data_output_dir", "data")) / "english" / "mgsm"),
            n_few_shot=mgsm_cfg.get("n_few_shot", 8),
            hf_token=hf_token,
            hf_path=mgsm_cfg.get("hf_path", "juletxara/mgsm"),
            config_name=mgsm_cfg.get("config_name", "en"),
        )
        result["mgsm_en_accuracy"] = mgsm_result["accuracy"]
        logger.info(
            "MGSM (English): %.2f%% (%d/%d correct)",
            mgsm_result["accuracy"] * 100,
            mgsm_result["n_correct"],
            mgsm_result["n_total"],
        )
    except Exception as exc:
        logger.error("MGSM evaluation failed: %s", exc, exc_info=True)

    # ── GPQA ─────────────────────────────────────────────────────────
    try:
        gpqa_cfg = eng_cfg.get("gpqa", {})
        gpqa_result = _evaluate_gpqa(
            model, tokenizer, max_len, device,
            data_dir=str(Path(data_config.get("data_output_dir", "data")) / "english" / "gpqa"),
            n_few_shot=gpqa_cfg.get("n_few_shot", 0),
            hf_token=hf_token,
            hf_path=gpqa_cfg.get("hf_path", "Idavidrein/gpqa"),
            config_name=gpqa_cfg.get("config_name", "gpqa_main"),
        )
        result["gpqa_accuracy"] = gpqa_result["accuracy"]
        logger.info(
            "GPQA: %.2f%% (%d/%d correct)",
            gpqa_result["accuracy"] * 100,
            gpqa_result["n_correct"],
            gpqa_result["n_total"],
        )
    except Exception as exc:
        logger.error("GPQA evaluation failed: %s", exc, exc_info=True)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: generate a short answer from the model
# ──────────────────────────────────────────────────────────────────────────────

def _generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_len: int,
    device: str,
    max_new_tokens: int = 32,
) -> str:
    """Generate a short answer from the model given a prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len - max_new_tokens,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return generated


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Load or download dataset
# ──────────────────────────────────────────────────────────────────────────────

def _load_or_download_dataset(
    data_dir: str,
    hf_path: str,
    config_name: str,
    split: str,
    hf_token: str | None = None,
    cache_filename: str | None = None,
):
    """Load a dataset from local cache (data/english/...) or download it.

    First checks if a local JSONL cache exists in *data_dir*.  If not,
    downloads from HuggingFace Hub and saves a JSONL cache for future
    runs.

    Returns:
        A ``datasets.Dataset`` object.
    """
    import datasets as hf_datasets

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    cache_name = cache_filename or f"{split}.jsonl"
    cache_file = data_path / cache_name

    if cache_file.exists():
        logger.info("Loading cached dataset from %s", cache_file)
        ds = hf_datasets.load_dataset(
            "json", data_files=str(cache_file), split="train"
        )
        return ds

    # Download from HuggingFace Hub
    logger.info(
        "Downloading dataset: %s (config=%s, split=%s)",
        hf_path, config_name, split,
    )
    try:
        ds = hf_datasets.load_dataset(
            hf_path, config_name, split=split, token=hf_token,
        )
    except Exception as exc:
        logger.warning(
            "Primary download failed (%s). Trying without config_name...",
            exc,
        )
        ds = hf_datasets.load_dataset(
            hf_path, split=split, token=hf_token,
        )

    # Cache locally as JSONL
    try:
        ds.to_json(str(cache_file))
        logger.info("Cached dataset to %s (%d rows)", cache_file, len(ds))
    except Exception as exc:
        logger.warning("Could not cache dataset to %s: %s", cache_file, exc)

    return ds


# ──────────────────────────────────────────────────────────────────────────────
# MMLU (5-shot, 500 stratified questions)
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    max_len: int,
    device: str,
    data_dir: str,
    n_samples: int = 500,
    n_few_shot: int = 5,
    hf_token: str | None = None,
    hf_path: str = "cais/mmlu",
    config_name: str = "all",
) -> dict:
    """Evaluate MMLU (5-shot) on a stratified subset of questions.

    Returns dict with ``accuracy``, ``n_correct``, ``n_total``,
    ``category_scores``.
    """
    # Load test split
    test_ds = _load_or_download_dataset(
        data_dir, hf_path, config_name, "test",
        hf_token=hf_token, cache_filename="test.jsonl",
    )

    # Load dev/validation split for few-shot examples
    try:
        dev_ds = _load_or_download_dataset(
            data_dir, hf_path, config_name, "validation",
            hf_token=hf_token, cache_filename="validation.jsonl",
        )
    except Exception:
        try:
            dev_ds = _load_or_download_dataset(
                data_dir, hf_path, config_name, "dev",
                hf_token=hf_token, cache_filename="dev.jsonl",
            )
        except Exception:
            logger.warning("No dev/validation split found for MMLU. Using 0-shot.")
            dev_ds = None
            n_few_shot = 0

    # Detect column names
    cols = test_ds.column_names
    question_col = _detect_col(cols, ["question", "Question", "input"])
    choices_col = _detect_col(cols, ["choices", "options", "Options"])
    answer_col = _detect_col(cols, ["answer", "Answer", "correct_answer", "target"])
    subject_col = _detect_col(cols, ["subject", "Subject", "category"])

    if not question_col or not answer_col:
        raise ValueError(f"Cannot detect MMLU columns. Available: {cols}")

    # Stratified sampling: pick proportionally from each subject
    test_list = list(test_ds)
    if subject_col:
        from collections import defaultdict
        by_subject = defaultdict(list)
        for row in test_list:
            by_subject[row.get(subject_col, "unknown")].append(row)

        n_subjects = len(by_subject)
        per_subject = max(1, n_samples // n_subjects)
        sampled = []
        for subj, rows in sorted(by_subject.items()):
            sampled.extend(rows[:per_subject])
        # If we need more, pad from remaining
        if len(sampled) < n_samples:
            used = set(id(r) for r in sampled)
            for row in test_list:
                if id(row) not in used and len(sampled) < n_samples:
                    sampled.append(row)
        test_list = sampled[:n_samples]
    else:
        test_list = test_list[:n_samples]

    # Build few-shot examples per subject
    dev_by_subject = {}
    if dev_ds is not None and n_few_shot > 0 and subject_col:
        for row in dev_ds:
            subj = row.get(subject_col, "unknown")
            if subj not in dev_by_subject:
                dev_by_subject[subj] = []
            if len(dev_by_subject[subj]) < n_few_shot:
                dev_by_subject[subj].append(row)
    elif dev_ds is not None and n_few_shot > 0:
        dev_by_subject["_all"] = list(dev_ds)[:n_few_shot]

    # Evaluate
    answer_labels = ["A", "B", "C", "D"]
    correct = 0
    category_correct = {}
    category_total = {}

    for i, row in enumerate(test_list):
        subject = row.get(subject_col, "unknown") if subject_col else "unknown"
        category = _SUBJECT_TO_CATEGORY.get(subject, "Other")

        # Build few-shot prefix
        few_shot_text = ""
        if n_few_shot > 0:
            examples = dev_by_subject.get(subject, dev_by_subject.get("_all", []))
            for ex in examples[:n_few_shot]:
                few_shot_text += _format_mmlu_question(
                    ex, question_col, choices_col, answer_col, answer_labels,
                    include_answer=True,
                ) + "\n\n"

        # Build the test question
        question_text = _format_mmlu_question(
            row, question_col, choices_col, answer_col, answer_labels,
            include_answer=False,
        )

        prompt = (
            f"The following are multiple choice questions (with answers) "
            f"about {subject.replace('_', ' ')}.\n\n"
            f"{few_shot_text}{question_text}"
        )

        generated = _generate_answer(model, tokenizer, prompt, max_len, device, max_new_tokens=8)
        gold = _normalize_mmlu_answer(row.get(answer_col, ""), answer_labels)
        predicted = _extract_mcq_answer(generated, answer_labels)

        is_correct = predicted == gold
        if is_correct:
            correct += 1

        category_correct[category] = category_correct.get(category, 0) + (1 if is_correct else 0)
        category_total[category] = category_total.get(category, 0) + 1

        if (i + 1) % 100 == 0:
            logger.info(
                "  MMLU progress: %d/%d (%.1f%% so far)",
                i + 1, len(test_list),
                100 * correct / (i + 1),
            )

    n_total = len(test_list)
    accuracy = correct / n_total if n_total > 0 else 0.0

    category_scores = {}
    for cat in ["STEM", "Humanities", "Social Sciences", "Other"]:
        n_cat = category_total.get(cat, 0)
        n_cat_correct = category_correct.get(cat, 0)
        category_scores[cat] = round(n_cat_correct / n_cat, 4) if n_cat > 0 else None

    return {
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_total": n_total,
        "category_scores": category_scores,
    }


def _format_mmlu_question(
    row: dict,
    question_col: str,
    choices_col: str | None,
    answer_col: str,
    answer_labels: list[str],
    include_answer: bool = False,
) -> str:
    """Format a single MMLU question with answer choices."""
    q = row[question_col]
    text = f"Question: {q}\n"

    if choices_col and row.get(choices_col):
        choices = row[choices_col]
        if isinstance(choices, list):
            for i, c in enumerate(choices):
                if i < len(answer_labels):
                    text += f"{answer_labels[i]}. {c}\n"
    text += "Answer:"

    if include_answer:
        gold = _normalize_mmlu_answer(row.get(answer_col, ""), answer_labels)
        text += f" {gold}"

    return text


def _normalize_mmlu_answer(answer: Any, labels: list[str]) -> str:
    """Normalize an MMLU answer to a letter (A/B/C/D).

    Handles integer indices (0→A, 1→B, ...) and string labels.
    """
    if isinstance(answer, (int, float)):
        idx = int(answer)
        if 0 <= idx < len(labels):
            return labels[idx]
    answer_str = str(answer).strip().upper()
    if answer_str in labels:
        return answer_str
    # Try first character
    if answer_str and answer_str[0] in "ABCD":
        return answer_str[0]
    return answer_str


def _extract_mcq_answer(generated: str, labels: list[str]) -> str:
    """Extract the multiple-choice answer letter from generated text."""
    generated = generated.strip()
    if not generated:
        return ""

    # Check first non-whitespace character
    first_char = generated[0].upper()
    if first_char in labels:
        # Verify it's not part of a longer word
        if len(generated) == 1 or not generated[1].isalpha():
            return first_char

    # Look for patterns like "(A)", "A.", "A:", "Answer: A"
    for label in labels:
        patterns = [
            rf"\b{label}\b",
            rf"\({label}\)",
            rf"{label}\.",
            rf"{label}:",
        ]
        for pat in patterns:
            if re.search(pat, generated[:20], re.IGNORECASE):
                return label

    return generated[0].upper() if generated else ""


# ──────────────────────────────────────────────────────────────────────────────
# MGSM (8-shot, English, chain-of-thought)
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_mgsm(
    model: Any,
    tokenizer: Any,
    max_len: int,
    device: str,
    data_dir: str,
    n_few_shot: int = 8,
    hf_token: str | None = None,
    hf_path: str = "juletxara/mgsm",
    config_name: str = "en",
) -> dict:
    """Evaluate MGSM (8-shot, English) — grade-school math problems.

    Returns dict with ``accuracy``, ``n_correct``, ``n_total``.
    """
    # Load test split
    test_ds = _load_or_download_dataset(
        data_dir, hf_path, config_name, "test",
        hf_token=hf_token, cache_filename="test.jsonl",
    )

    # Load train split for few-shot examples
    few_shot_examples = []
    if n_few_shot > 0:
        try:
            train_ds = _load_or_download_dataset(
                data_dir, hf_path, config_name, "train",
                hf_token=hf_token, cache_filename="train.jsonl",
            )
            few_shot_examples = list(train_ds)[:n_few_shot]
        except Exception:
            logger.warning("No train split for MGSM few-shot. Using 0-shot.")

    # Detect columns
    cols = test_ds.column_names
    question_col = _detect_col(cols, ["question", "Question", "input", "problem"])
    answer_col = _detect_col(cols, [
        "answer_number", "numerical_answer", "answer", "Answer", "target"
    ])

    if not question_col or not answer_col:
        raise ValueError(f"Cannot detect MGSM columns. Available: {cols}")

    # Build few-shot prefix
    few_shot_text = ""
    for ex in few_shot_examples:
        q = ex[question_col]
        a = ex.get(answer_col, "")
        # If answer field has an explanation, use it for chain-of-thought
        answer_text = str(a)
        few_shot_text += f"Q: {q}\nA: The answer is {answer_text}.\n\n"

    # Evaluate
    correct = 0
    test_list = list(test_ds)

    for i, row in enumerate(test_list):
        question = row[question_col]
        gold_answer = _extract_number(str(row.get(answer_col, "")))

        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"{few_shot_text}"
            f"Q: {question}\nA:"
        )

        generated = _generate_answer(
            model, tokenizer, prompt, max_len, device,
            max_new_tokens=128,  # Math problems need longer generation
        )
        predicted = _extract_number(generated)

        if predicted is not None and gold_answer is not None:
            if abs(predicted - gold_answer) < 1e-3:
                correct += 1

        if (i + 1) % 50 == 0:
            logger.info(
                "  MGSM progress: %d/%d (%.1f%% so far)",
                i + 1, len(test_list),
                100 * correct / (i + 1),
            )

    n_total = len(test_list)
    accuracy = correct / n_total if n_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_total": n_total,
    }


def _extract_number(text: str) -> float | None:
    """Extract the last numerical value from text.

    Handles formats like:
    - "The answer is 42."
    - "#### 42"
    - Plain "42"
    - Negative numbers: "-42"
    - Decimals: "3.14"
    - Commas: "1,234"
    """
    if not text:
        return None

    # Try "#### N" pattern first (common in math datasets)
    match = re.search(r"####\s*([-−]?[\d,]+\.?\d*)", text)
    if match:
        return _parse_number(match.group(1))

    # Try "The answer is N" pattern
    match = re.search(
        r"(?:the answer is|answer:|=)\s*([-−]?[\d,]+\.?\d*)",
        text, re.IGNORECASE,
    )
    if match:
        return _parse_number(match.group(1))

    # Find the last number in the text
    numbers = re.findall(r"[-−]?[\d,]+\.?\d*", text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def _parse_number(s: str) -> float | None:
    """Parse a number string, handling commas and Unicode minus."""
    try:
        s = s.replace(",", "").replace("−", "-").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# GPQA (0-shot, graduate-level science)
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_gpqa(
    model: Any,
    tokenizer: Any,
    max_len: int,
    device: str,
    data_dir: str,
    n_few_shot: int = 0,
    hf_token: str | None = None,
    hf_path: str = "Idavidrein/gpqa",
    config_name: str = "gpqa_main",
) -> dict:
    """Evaluate GPQA (0-shot) — graduate-level science MCQ.

    Returns dict with ``accuracy``, ``n_correct``, ``n_total``.
    """
    # Load dataset
    test_ds = _load_or_download_dataset(
        data_dir, hf_path, config_name, "train",  # GPQA only has train split
        hf_token=hf_token, cache_filename="test.jsonl",
    )

    # Detect columns — GPQA has unique column names
    cols = test_ds.column_names
    question_col = _detect_col(cols, [
        "Question", "question", "input", "problem",
    ])
    # GPQA has separate columns for each choice
    answer_col = _detect_col(cols, [
        "answer", "Answer", "correct_answer", "Correct Answer",
    ])

    # GPQA-specific: choices may be in separate columns
    choice_cols = []
    for pattern in [
        "Correct Answer", "Incorrect Answer 1",
        "Incorrect Answer 2", "Incorrect Answer 3",
    ]:
        col = _detect_col(cols, [pattern])
        if col:
            choice_cols.append(col)

    # Also try standard choices column
    choices_col = _detect_col(cols, ["choices", "options", "Options"])

    if not question_col:
        raise ValueError(f"Cannot detect GPQA question column. Available: {cols}")

    answer_labels = ["A", "B", "C", "D"]
    correct = 0
    test_list = list(test_ds)

    for i, row in enumerate(test_list):
        question = row[question_col]

        # Build choices text
        if choice_cols and len(choice_cols) >= 2:
            # GPQA format: separate columns for each choice
            # The correct answer is always in the first column
            choices = [row.get(c, "") for c in choice_cols]
            correct_answer = "A"  # Correct answer is always index 0 in GPQA raw format

            # Shuffle or use as-is (GPQA main is already randomized)
            choices_text = "\n".join(
                f"{answer_labels[j]}. {c}"
                for j, c in enumerate(choices)
                if j < len(answer_labels)
            )
        elif choices_col and row.get(choices_col):
            choices = row[choices_col]
            if isinstance(choices, list):
                choices_text = "\n".join(
                    f"{answer_labels[j]}. {c}"
                    for j, c in enumerate(choices)
                    if j < len(answer_labels)
                )
            else:
                choices_text = str(choices)

            if answer_col:
                correct_answer = _normalize_mmlu_answer(
                    row.get(answer_col, ""), answer_labels
                )
            else:
                correct_answer = "A"
        else:
            # Skip if we can't find choices
            logger.warning("GPQA row %d: no choices found, skipping.", i)
            continue

        if answer_col and answer_col in row:
            correct_answer = _normalize_mmlu_answer(
                row[answer_col], answer_labels
            )

        prompt = (
            f"Answer the following graduate-level science question.\n\n"
            f"Question: {question}\n"
            f"{choices_text}\n"
            f"Answer:"
        )

        generated = _generate_answer(
            model, tokenizer, prompt, max_len, device, max_new_tokens=8,
        )
        predicted = _extract_mcq_answer(generated, answer_labels)

        if predicted == correct_answer:
            correct += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "  GPQA progress: %d/%d (%.1f%% so far)",
                i + 1, len(test_list),
                100 * correct / (i + 1),
            )

    n_total = len(test_list)
    accuracy = correct / n_total if n_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_total": n_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in columns:
            return c
    # Case-insensitive fallback
    lower_map = {col.lower(): col for col in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None
