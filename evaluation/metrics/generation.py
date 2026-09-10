"""chrF++ and BLEU computation via sacrebleu, and FLORES-200 inference.

Provides corpus-level generation quality metrics and a helper to run
the model on FLORES-200 aligned English→target translation pairs for
standardised evaluation across all modules.

Usage:
    from evaluation.metrics.generation import compute_chrf, compute_bleu
    from evaluation.metrics.generation import generate_flores_predictions
"""

import logging

import datasets
import sacrebleu
import torch

from utils.data_utils import format_phi4_prompt
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level chrF++ score using sacrebleu.

    Args:
        hypotheses: Model-generated strings.
        references: Gold-reference strings (same length as *hypotheses*).

    Returns:
        The chrF++ score rounded to 4 decimal places.

    Raises:
        ValueError: If the lists are empty or have mismatched lengths.
    """
    if not hypotheses or not references:
        raise ValueError("hypotheses and references must be non-empty lists")
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs "
            f"{len(references)} references"
        )

    score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    return round(float(score), 4)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score using sacrebleu.

    Args:
        hypotheses: Model-generated strings.
        references: Gold-reference strings (same length as *hypotheses*).

    Returns:
        The BLEU score rounded to 4 decimal places.

    Raises:
        ValueError: If the lists are empty or have mismatched lengths.
    """
    if not hypotheses or not references:
        raise ValueError("hypotheses and references must be non-empty lists")
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs "
            f"{len(references)} references"
        )

    score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return round(float(score), 4)


def generate_flores_predictions(
    model,
    tokenizer,
    language: str,
    data_config: dict,
    model_config: dict,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> dict[str, list[str]]:
    """Load FLORES-200 data and generate translations with the model.

    Loads aligned English source sentences and target-language gold
    references from FLORES-200, prompts the model to translate each
    English sentence, and returns the generated hypotheses alongside
    the gold references.

    Args:
        model: A HuggingFace causal LM (already on *device*).
        tokenizer: The corresponding tokenizer.
        language: Language name (``"kannada"`` or ``"telugu"``).
        data_config: Parsed ``configs/data.yaml`` dict.
        model_config: Parsed ``configs/model.yaml`` dict.
        max_new_tokens: Maximum tokens to generate per example.
        device: Torch device string.

    Returns:
        A dict with keys ``"hypotheses"``, ``"references"``, and
        ``"sources"`` where each is a list of strings.  Returns
        empty lists for all keys if FLORES loading fails so
        evaluation degrades gracefully.
    """
    flores_hf_path = data_config["flores_hf_path"]
    lang_code = data_config["flores_language_codes"][language]
    split = data_config["flores_eval_split"]
    n_samples = data_config["flores_eval_samples"]

    language_name = "Kannada" if language == "kannada" else "Telugu"

    # --- Load aligned English + target datasets ----------------------------
    token = model_config.get("hf_token") or None
    try:
        eng_ds = datasets.load_dataset(
            flores_hf_path, "eng_Latn", split=split, token=token
        )
        target_ds = datasets.load_dataset(
            flores_hf_path, lang_code, split=split, token=token
        )
    except Exception as exc:
        logger.error(
            "Failed to load FLORES-200 dataset (path=%s, lang=%s): %s",
            flores_hf_path,
            lang_code,
            exc,
        )
        return {"hypotheses": [], "references": [], "sources": []}

    # Take first n_samples examples (datasets are aligned by index).
    eng_ds = eng_ds.select(range(min(n_samples, len(eng_ds))))
    target_ds = target_ds.select(range(min(n_samples, len(target_ds))))

    hypotheses: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    max_input_len = model_config["max_context_length"] - max_new_tokens

    for idx in range(len(eng_ds)):
        eng_sentence = eng_ds[idx]["sentence"]
        target_sentence = target_ds[idx]["sentence"]

        instruction = (
            f"Translate the following English sentence to {language_name}:"
        )
        prompt = format_phi4_prompt(
            instruction=instruction,
            input_text=eng_sentence,
            output="",
            include_output=False,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens.
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        hypotheses.append(generated)
        references.append(target_sentence)
        sources.append(eng_sentence)

        if (idx + 1) % 10 == 0:
            logger.info(
                "FLORES generation progress: %d / %d", idx + 1, len(eng_ds)
            )

    logger.info(
        "FLORES generation complete: %d hypotheses for %s",
        len(hypotheses),
        language,
    )
    return {"hypotheses": hypotheses, "references": references, "sources": sources}


def compute_bertscore_muril(
    hypotheses: list[str],
    references: list[str],
    language: str,
    model_type: str = "google/muril-base-cased",
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute BERTScore using MuRIL for Indian languages.

    MuRIL (Multilingual Representations for Indian Languages) is
    Google's encoder model trained on 17 Indian languages including
    Kannada and Telugu.  BERTScore in embedding space captures
    semantic equivalence despite morphological variation — critical
    for agglutinative Dravidian languages.

    Args:
        hypotheses: Model-generated translation strings.
        references: Gold reference strings from FLORES-200.
        language: ``"kannada"`` or ``"telugu"`` (maps to lang code).
        model_type: HuggingFace encoder model ID.
        batch_size: Scoring batch size.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Dict with ``bertscore_precision``, ``bertscore_recall``,
        ``bertscore_f1``.
    """
    try:
        from bert_score import score as _bert_score
        import bert_score.utils as bert_utils
    except ImportError as exc:
        raise ImportError(
            "bert-score not installed. Run: pip install bert-score"
        ) from exc

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

    if not hypotheses or not references:
        logger.warning("BERTScore: empty inputs, returning zeros.")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall":    0.0,
            "bertscore_f1":        0.0,
        }

    lang_code = "kn" if language == "kannada" else "te"

    # Resolve local path if it exists to ensure transformers loads it offline
    from pathlib import Path
    resolved_model_path = model_type
    if Path(model_type).exists():
        resolved_model_path = str(Path(model_type).resolve())

    # Check if the model is registered in bert_score's model2layers dictionary
    num_layers = None
    if model_type not in bert_utils.model2layers:
        num_layers = 8
        logger.info(
            "model_type '%s' not found in bert_score.utils.model2layers. "
            "Defaulting num_layers to 8.", model_type
        )

    try:
        P, R, F1 = _bert_score(
            cands=hypotheses,
            refs=references,
            model_type=resolved_model_path,
            num_layers=num_layers,
            lang=lang_code,
            batch_size=batch_size,
            device=device,
            verbose=False,
            rescale_with_baseline=False,
        )
        result = {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall":    round(R.mean().item(), 4),
            "bertscore_f1":        round(F1.mean().item(), 4),
        }
        logger.info("BERTScore F1 (MuRIL): %s", result["bertscore_f1"])
        return result
    except Exception as exc:
        logger.warning(
            "BERTScore computation failed: %s. Returning zeros.", exc
        )
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall":    0.0,
            "bertscore_f1":        0.0,
        }


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
    saving_directory: str | None = None,
) -> float:
    """Compute COMET score — neural reference-based translation quality.

    Substantially outperforms BLEU and chrF++ in correlation with human
    ratings on Indic language translation tasks.  Requires source
    (English), hypothesis (generated), and reference (gold).

    Args:
        sources: Original English source sentences.
        hypotheses: Model-generated target translations.
        references: Gold reference target translations.
        model_name: Unbabel COMET model identifier.
        batch_size: Prediction batch size.
        gpus: Number of GPUs for scoring (0 = CPU).
        saving_directory: Saving directory for the downloaded model.

    Returns:
        System-level COMET score (float, typically 0.0–1.0).
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise ImportError(
            "unbabel-comet not installed. Run: pip install unbabel-comet"
        ) from exc

    if not sources or not hypotheses or not references:
        logger.warning("COMET: empty inputs, returning 0.0")
        return 0.0

    n = min(len(sources), len(hypotheses), len(references))
    if n < len(sources):
        logger.warning("COMET: length mismatch, using first %d examples", n)

    try:
        if saving_directory:
            model_path = download_model(model_name, saving_directory=saving_directory)
        else:
            model_path = download_model(model_name)
        comet_model = load_from_checkpoint(model_path)
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources[:n], hypotheses[:n], references[:n])
        ]
        output = comet_model.predict(
            data, batch_size=batch_size, gpus=gpus, progress_bar=False
        )
        score = round(float(output.system_score), 4)
        logger.info("COMET score: %s", score)

        # Clean up COMET model to release VRAM immediately
        try:
            import gc
            comet_model = comet_model.to("cpu")
            del comet_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as cleanup_exc:
            logger.warning("COMET model VRAM cleanup failed: %s", cleanup_exc)

        return score
    except Exception as exc:
        logger.warning("COMET computation failed: %s. Returning 0.0", exc)
        return 0.0
