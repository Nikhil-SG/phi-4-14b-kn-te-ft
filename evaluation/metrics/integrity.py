"""Language integrity metrics for generated text.

Measures how well generated text stays within the target Dravidian
script, detects code-switching to ASCII Latin, and computes
teacher-forced perplexity on held-out native-language texts.

Usage:
    from evaluation.metrics.integrity import script_purity_rate
    from evaluation.metrics.integrity import code_switch_rate
    from evaluation.metrics.integrity import compute_held_out_perplexity
"""

import math
import re

import torch

from utils.data_utils import compute_script_purity
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def script_purity_rate(
    text: str,
    language: str,
    data_config: dict,
) -> float:
    """Compute script purity of *text* for the given language.

    Looks up the Unicode range from *data_config* and delegates to
    :func:`utils.data_utils.compute_script_purity`.

    Args:
        text: The generated text to evaluate.
        language: Language name (``"kannada"`` or ``"telugu"``).
        data_config: Parsed ``configs/data.yaml`` dict containing
            ``datasets.<language>.script_unicode_start`` and
            ``datasets.<language>.script_unicode_end``.

    Returns:
        Fraction of non-whitespace characters in the target script
        range, as a float in ``[0.0, 1.0]``.
    """
    lang_cfg = data_config["datasets"][language]
    return compute_script_purity(
        text,
        lang_cfg["script_unicode_start"],
        lang_cfg["script_unicode_end"],
    )


def code_switch_rate(text: str) -> float:
    """Compute the fraction of tokens that are purely ASCII Latin letters.

    A high code-switch rate indicates the model is falling back to
    English rather than generating in the target Dravidian script.

    Args:
        text: The generated text to evaluate.

    Returns:
        Fraction of whitespace-split tokens that consist entirely of
        ASCII Latin letters (``[a-zA-Z]+``).  Returns ``0.0`` for
        empty text.
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    ascii_latin_count = sum(
        1 for token in tokens if re.match(r"^[a-zA-Z]+$", token)
    )
    return ascii_latin_count / len(tokens)


def compute_held_out_perplexity(
    model,
    tokenizer,
    texts: list[str],
    model_config: dict,
    device: str = "cuda",
) -> float:
    """Compute teacher-forced perplexity on a list of native-language texts.

    Runs each text through the model with ``labels=input_ids`` to
    obtain the cross-entropy loss, then aggregates across all tokens
    to compute the corpus-level perplexity.

    Args:
        model: A HuggingFace causal LM (already on *device*).
        tokenizer: The corresponding tokenizer.
        texts: List of native-language strings for perplexity evaluation.
        model_config: Parsed ``configs/model.yaml`` dict (used for
            ``max_context_length``).
        device: Torch device string.

    Returns:
        The perplexity as ``exp(avg_cross_entropy)``.  Returns
        ``float('nan')`` if a CUDA OOM error occurs.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    max_length = model_config["max_context_length"]

    try:
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encodings["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss.item()
                n_tokens = input_ids.shape[1]

            total_loss += loss * n_tokens
            total_tokens += n_tokens

    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "CUDA OOM during perplexity computation — returning NaN. "
            "Processed %d tokens before failure.",
            total_tokens,
        )
        torch.cuda.empty_cache()
        return float("nan")

    if total_tokens == 0:
        logger.warning("No tokens processed for perplexity — returning NaN.")
        return float("nan")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def generation_byte_fallback_rate(
    hypotheses: list[str],
    tokenizer,
) -> float:
    """Measure the fraction of byte-fallback tokens in model-generated output.

    Phi-4 (tiktoken-based) represents unknown or rare characters as
    ``<0xNN>`` hex tokens.  A high byte-fallback rate in Dravidian output
    indicates the model is generating characters byte-by-byte rather than
    using learned Dravidian vocabulary — a direct signal of poor script
    acquisition.

    This metric closes the Phase 0 fertility narrative: Phase 0 measured
    how the tokenizer fragments input text; this measures how the
    fine-tuned model fragments its own generated output.  Reduction
    after fine-tuning is a concrete, visually compelling win.

    Args:
        hypotheses: List of model-generated strings (from FLORES
            inference).
        tokenizer: The fine-tuned model's tokenizer
            (``PreTrainedTokenizer``).

    Returns:
        Fraction of generated tokens that are byte-fallback tokens
        (0.0–1.0).
    """
    total_tokens = 0
    fallback_tokens = 0

    for h in hypotheses:
        if not h or not h.strip():
            continue
        try:
            ids = tokenizer.encode(h, add_special_tokens=False)
            token_strs = tokenizer.convert_ids_to_tokens(ids)
            for t in token_strs:
                if t is None:
                    continue
                # tiktoken / Phi-4 byte fallback pattern: <0x41>, <0xE0>, etc.
                if len(t) >= 5 and t.startswith("<0x") and t.endswith(">"):
                    fallback_tokens += 1
            total_tokens += len(ids)
        except Exception:
            continue

    if total_tokens == 0:
        return 0.0
    return round(fallback_tokens / total_tokens, 4)
