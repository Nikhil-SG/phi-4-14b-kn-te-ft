"""Prompt Sensitivity Delta — instruction-following robustness metric.

Measures how much a fine-tuned model's accuracy on multiple-choice
questions varies across different prompt templates.  A well-aligned
model should show low variance (small delta) across templates.

Uses IndicMMLU for target-language evaluation, with three prompt
templates: standard, system_role, and few_shot.

Only called when ``run_prompt_sensitivity=True`` (gated — very
expensive due to per-question inference).
"""

import logging
import re
import statistics
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compute_prompt_sensitivity_delta(
    model: Any,
    tokenizer: Any,
    language: str,
    data_config: dict,
    model_config: dict,
    n_questions: int = 100,
) -> dict[str, float] | None:
    """Compute prompt sensitivity delta across three prompt templates.

    Loads IndicMMLU for the target language, evaluates the model on
    ``n_questions`` multiple-choice questions using three different
    prompt templates (standard, system_role, few_shot), and returns
    the standard deviation of accuracy across templates (lower is
    better — more robust to prompt variation).

    Args:
        model: Fine-tuned causal LM (on GPU).
        tokenizer: The corresponding tokenizer.
        language: ``"kannada"`` or ``"telugu"``.
        data_config: Parsed ``configs/data.yaml`` dict.
        model_config: Parsed ``configs/model.yaml`` dict.
        n_questions: Number of IndicMMLU questions to evaluate.

    Returns:
        Dict with ``prompt_sensitivity_delta`` (std dev of accuracy)
        and ``prompt_sensitivity_mean_acc`` (mean accuracy across
        templates), or ``None`` if the dataset is unavailable.
    """
    import datasets as hf_datasets

    ps_cfg = data_config.get("prompt_sensitivity", {})
    hf_path = ps_cfg.get("dataset_hf_path", "data/mmlu-indic")
    fallback_path = ps_cfg.get("fallback_hf_path", "sarvamai/mmlu-indic")

    lang_code = data_config["datasets"][language]["language_code"]
    language_name = "Kannada" if language == "kannada" else "Telugu"

    # ── 1. Load IndicMMLU ─────────────────────────────────────────────
    mmlu_ds = None
    for path in [hf_path, fallback_path]:
        try:
            from pathlib import Path
            if Path(path).exists():
                target_path = path
                if Path(path).is_dir():
                    # Check if there is a language-specific subdirectory (e.g. data/mmlu-indic/kn)
                    local_lang_path = Path(path) / lang_code
                    if local_lang_path.exists() and local_lang_path.is_dir():
                        target_path = str(local_lang_path)
                    
                    try:
                        mmlu_ds = hf_datasets.load_from_disk(target_path)
                    except Exception:
                        files = [str(f) for f in Path(target_path).glob("*.json*")]
                        mmlu_ds = hf_datasets.load_dataset("json", data_files=files)
                else:
                    mmlu_ds = hf_datasets.load_dataset("json", data_files=path)
                
                if isinstance(mmlu_ds, hf_datasets.DatasetDict):
                    mmlu_ds = mmlu_ds["test"] if "test" in mmlu_ds else list(mmlu_ds.values())[0]
                
                if "language" in mmlu_ds.column_names:
                    mmlu_ds = mmlu_ds.filter(
                        lambda x: x["language"] == lang_code
                    )
                logger.info("Loaded IndicMMLU from local path %s: %d questions", target_path, len(mmlu_ds))
                break

            mmlu_ds = hf_datasets.load_dataset(path, lang_code, split="test")
            logger.info(
                "IndicMMLU loaded from %s (%s): %d questions",
                path,
                lang_code,
                len(mmlu_ds),
            )
            break
        except Exception as exc:
            try:
                mmlu_ds = hf_datasets.load_dataset(path, split="test")
                # Filter by language if full dataset loaded
                if "language" in mmlu_ds.column_names:
                    mmlu_ds = mmlu_ds.filter(
                        lambda x: x["language"] == lang_code
                    )
                logger.info(
                    "IndicMMLU loaded from %s (filtered): %d questions",
                    path,
                    len(mmlu_ds),
                )
                break
            except Exception:
                logger.warning("Failed to load IndicMMLU from %s: %s", path, exc)
                continue

    if mmlu_ds is None or len(mmlu_ds) == 0:
        logger.warning(
            "IndicMMLU not available for %s. Skipping prompt sensitivity.",
            language,
        )
        return None

    # Truncate to n_questions
    mmlu_ds = mmlu_ds.select(range(min(n_questions, len(mmlu_ds))))

    # ── 2. Detect column names ────────────────────────────────────────
    cols = mmlu_ds.column_names
    question_col = next(
        (c for c in ["question", "Question", "input", "prompt"] if c in cols),
        None,
    )
    choices_col = next(
        (c for c in ["choices", "options", "Options"] if c in cols), None
    )
    answer_col = next(
        (c for c in ["answer", "Answer", "correct_answer", "label"] if c in cols),
        None,
    )

    if not question_col or not answer_col:
        logger.warning(
            "IndicMMLU columns not recognised: %s. Skipping.", cols
        )
        return None

    # ── 3. Build prompt templates ─────────────────────────────────────
    templates = ps_cfg.get("templates", {})

    def _build_mcq_text(row: dict) -> str:
        """Build the question + choices text from a single row."""
        q = row[question_col]
        if choices_col and row.get(choices_col):
            choices = row[choices_col]
            if isinstance(choices, list):
                labels = ["A", "B", "C", "D", "E", "F"]
                choice_text = "\n".join(
                    f"  {labels[i]}. {c}"
                    for i, c in enumerate(choices)
                    if i < len(labels)
                )
                return f"{q}\n{choice_text}"
        return q

    def _make_prompt(row: dict, template_name: str) -> str:
        """Format a single MCQ question according to template."""
        from utils.data_utils import format_phi4_prompt

        tmpl = templates.get(template_name, {})
        prefix = tmpl.get("prefix", "Answer the following multiple choice question.")
        suffix = tmpl.get("suffix", "Answer:")
        system = tmpl.get("system", None)

        mcq_text = _build_mcq_text(row)
        instruction = f"{prefix}\n\n{mcq_text}\n\n{suffix}"

        if system and template_name == "system_role":
            system_text = system.format(language_name=language_name)
            instruction = f"{system_text}\n\n{instruction}"

        return format_phi4_prompt(
            instruction=instruction,
            input_text="",
            output="",
            include_output=False,
        )

    def _make_few_shot_prompt(row: dict, examples: list[dict]) -> str:
        """Format a few-shot MCQ prompt."""
        from utils.data_utils import format_phi4_prompt

        tmpl = templates.get("few_shot", {})
        prefix = tmpl.get(
            "prefix",
            "Here are example questions with answers, followed by a new question.",
        )
        suffix = tmpl.get("suffix", "Answer:")

        parts = [prefix, ""]
        for ex in examples:
            ex_text = _build_mcq_text(ex)
            ex_answer = str(ex.get(answer_col, ""))
            parts.append(f"Q: {ex_text}")
            parts.append(f"A: {ex_answer}\n")

        mcq_text = _build_mcq_text(row)
        parts.append(f"Q: {mcq_text}")
        parts.append(suffix)

        return format_phi4_prompt(
            instruction="\n".join(parts),
            input_text="",
            output="",
            include_output=False,
        )

    # ── 4. Evaluate each template ─────────────────────────────────────
    max_len = model_config["max_context_length"]
    device = next(model.parameters()).device

    # Pre-compute token IDs for A/B/C/D once (try with and without leading space)
    _mcq_labels = ["A", "B", "C", "D"]
    _label_token_ids: list[int | None] = []
    for _lbl in _mcq_labels:
        # Try variants: " A", "A", "▁A" (SentencePiece prefix)
        _tid = None
        for _variant in [f" {_lbl}", _lbl, f"\u2581{_lbl}"]:
            _ids = tokenizer.encode(_variant, add_special_tokens=False)
            if _ids:
                _tid = _ids[0]
                break
        _label_token_ids.append(_tid)

    def _get_model_answer(prompt: str) -> str:
        """Score each MCQ label (A/B/C/D) by its next-token log-probability.

        Uses log-probability argmax over the four label tokens rather than
        greedy generation. This is the standard approach used in MMLU papers
        and LM-Eval-Harness, and is robust to translation fine-tuning where
        the model may no longer generate bare letter tokens as first output.

        Returns:
            The label letter (''A'', ''B'', ''C'', or ''D'') with the highest
            conditional probability given the prompt.
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len - 1,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # logits at the very last prompt position → P(next token | prompt)
            last_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

        scores = [
            last_logits[tid].item() if tid is not None else float("-inf")
            for tid in _label_token_ids
        ]
        return _mcq_labels[scores.index(max(scores))]

    def _check_answer(generated: str, gold: Any) -> bool:
        """Check if the generated answer matches the gold label.

        Handles both string labels ('A', 'B', etc.) and integer indices
        (0 -> 'A', 1 -> 'B', etc.) from datasets like IndicMMLU.
        """
        labels = ["A", "B", "C", "D", "E", "F"]
        if isinstance(gold, (int, float)):
            idx = int(gold)
            if 0 <= idx < len(labels):
                gold = labels[idx]
        else:
            gold_str = str(gold).strip()
            if gold_str.isdigit():
                idx = int(gold_str)
                if 0 <= idx < len(labels):
                    gold = labels[idx]
            else:
                gold = gold_str.upper()

        gold = str(gold).strip().upper()
        generated = generated.strip()
        # Check if gold label appears at the start of the generated text
        if generated and generated[0].upper() == gold:
            return True
        # Check if gold label is mentioned anywhere in short output
        gold_patterns = [gold, f"({gold})", f"{gold}."]
        for pat in gold_patterns:
            if pat in generated.upper():
                return True
        return False

    template_accuracies: dict[str, float] = {}

    # Standard template
    logger.info("Prompt sensitivity: evaluating 'standard' template...")
    correct = 0
    for row in mmlu_ds:
        prompt = _make_prompt(row, "standard")
        answer = _get_model_answer(prompt)
        if _check_answer(answer, row[answer_col]):
            correct += 1
    template_accuracies["standard"] = correct / len(mmlu_ds)
    logger.info("  standard accuracy: %.4f", template_accuracies["standard"])

    # System role template
    logger.info("Prompt sensitivity: evaluating 'system_role' template...")
    correct = 0
    for row in mmlu_ds:
        prompt = _make_prompt(row, "system_role")
        answer = _get_model_answer(prompt)
        if _check_answer(answer, row[answer_col]):
            correct += 1
    template_accuracies["system_role"] = correct / len(mmlu_ds)
    logger.info("  system_role accuracy: %.4f", template_accuracies["system_role"])

    # Few-shot template
    logger.info("Prompt sensitivity: evaluating 'few_shot' template...")
    n_examples = templates.get("few_shot", {}).get("n_examples", 3)
    correct = 0
    all_rows_list = list(mmlu_ds)
    for idx, row in enumerate(all_rows_list):
        # Use other rows as few-shot examples (avoid using the current row)
        examples = [
            all_rows_list[j]
            for j in range(len(all_rows_list))
            if j != idx
        ][:n_examples]
        prompt = _make_few_shot_prompt(row, examples)
        answer = _get_model_answer(prompt)
        if _check_answer(answer, row[answer_col]):
            correct += 1
    template_accuracies["few_shot"] = correct / len(mmlu_ds)
    logger.info("  few_shot accuracy: %.4f", template_accuracies["few_shot"])

    # ── 5. Compute delta ──────────────────────────────────────────────
    accs = list(template_accuracies.values())
    mean_acc = statistics.mean(accs)
    delta = statistics.stdev(accs) if len(accs) >= 2 else 0.0

    logger.info(
        "Prompt sensitivity: mean_acc=%.4f, delta(std)=%.4f, "
        "per_template=%s",
        mean_acc,
        delta,
        {k: round(v, 4) for k, v in template_accuracies.items()},
    )

    return {
        "prompt_sensitivity_delta":    round(delta, 4),
        "prompt_sensitivity_mean_acc": round(mean_acc, 4),
    }
