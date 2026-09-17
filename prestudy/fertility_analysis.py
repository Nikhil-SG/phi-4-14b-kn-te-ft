"""Measure how Phi-4's tokenizer handles Kannada and Telugu versus English.

Computes tokenizer fertility (tokens per word), byte-fallback rate, and
UNK rate for each language, then saves results to
``results/prestudy_fertility.json``.

Usage:
    python prestudy/fertility_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy
from transformers import AutoTokenizer

# Ensure project root is on sys.path for sibling-package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_utils import load_config, load_jsonl
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def analyze_fertility(
    texts: list[str],
    tokenizer,
    label: str,
) -> dict:
    """Analyse tokenizer fertility for a list of texts.

    For each text, computes tokens-per-word ratio, byte-fallback token
    rate (``<0xNN>`` hex tokens from tiktoken-based tokenizers), and
    UNK token rate.

    Args:
        texts: List of text strings to analyse.
        tokenizer: A HuggingFace tokenizer instance.
        label: Human-readable label for this analysis group
            (e.g. ``"kannada"``, ``"english"``).

    Returns:
        A dict with aggregated statistics:

        - ``label``: the provided label.
        - ``n_samples``: number of texts analysed.
        - ``tokens_per_word``: dict with mean/std/min/max.
        - ``byte_fallback_rate``: dict with mean/std.
        - ``unk_rate``: dict with mean/std.

        All floats are rounded to 4 decimal places.
    """
    tpw_list: list[float] = []
    bf_rate_list: list[float] = []
    unk_rate_list: list[float] = []

    unk_token_id = tokenizer.unk_token_id  # May be None for tiktoken-based

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        tokens_per_word = len(token_ids) / max(len(words), 1)
        tpw_list.append(tokens_per_word)

        # Byte-fallback detection: Phi-4 (tiktoken-based) uses <0xNN>
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        byte_fallback_count = sum(
            1
            for ts in token_strings
            if ts.startswith("<0x") and ts.endswith(">")
        )
        bf_rate = byte_fallback_count / max(len(token_ids), 1)
        bf_rate_list.append(bf_rate)

        # UNK token count
        unk_count = token_ids.count(unk_token_id) if unk_token_id is not None else 0
        unk_rate = unk_count / max(len(token_ids), 1)
        unk_rate_list.append(unk_rate)

    tpw_arr = numpy.array(tpw_list)
    bf_arr = numpy.array(bf_rate_list)
    unk_arr = numpy.array(unk_rate_list)

    return {
        "label": label,
        "n_samples": len(texts),
        "tokens_per_word": {
            "mean": round(float(numpy.mean(tpw_arr)), 4),
            "std": round(float(numpy.std(tpw_arr)), 4),
            "min": round(float(numpy.min(tpw_arr)), 4),
            "max": round(float(numpy.max(tpw_arr)), 4),
        },
        "byte_fallback_rate": {
            "mean": round(float(numpy.mean(bf_arr)), 4),
            "std": round(float(numpy.std(bf_arr)), 4),
        },
        "unk_rate": {
            "mean": round(float(numpy.mean(unk_arr)), 4),
            "std": round(float(numpy.std(unk_arr)), 4),
        },
    }


def _load_english_texts(n_samples: int = 500) -> list[str]:
    """Load English texts for the control group.

    Attempts to load from ``tatsu-lab/alpaca`` (HF cache). If that
    fails, generates simple English sentences programmatically.

    Args:
        n_samples: Number of English texts to return.

    Returns:
        A list of English text strings.
    """
    try:
        import datasets as hf_datasets

        ds = hf_datasets.load_dataset("tatsu-lab/alpaca", split="train")
        texts = [
            row["output"]
            for row in ds.select(range(min(n_samples, len(ds))))
            if row.get("output", "").strip()
        ]
        if len(texts) >= n_samples // 2:
            logger.info("Loaded %d English texts from tatsu-lab/alpaca.", len(texts))
            return texts[:n_samples]
    except Exception as exc:
        logger.warning("Could not load alpaca dataset: %s", exc)

    # Fallback: generate simple English sentences programmatically.
    logger.info("Generating %d synthetic English sentences as fallback.", n_samples)
    templates = [
        "The quick brown fox jumps over the lazy dog in the park.",
        "Machine learning models require large amounts of training data.",
        "Natural language processing is a subfield of artificial intelligence.",
        "The weather forecast predicts sunny skies for the weekend ahead.",
        "Students at the university are studying computer science and mathematics.",
        "The library contains thousands of books on various academic subjects.",
        "Research in deep learning has made significant progress in recent years.",
        "The conference will feature presentations on language technology advances.",
        "Data preprocessing is an essential step in any machine learning pipeline.",
        "The team developed a novel approach to cross-lingual text classification.",
    ]
    texts = []
    for i in range(n_samples):
        texts.append(templates[i % len(templates)])
    return texts


def main() -> None:
    """Entry point: run fertility analysis for all languages and English control.

    Loads configurations, initialises the Phi-4 tokenizer, analyses
    each language group, and saves results to
    ``results/module_0_prestudy/results/prestudy_fertility.json``.
    """
    setup_logging("module_0_prestudy", "module_0_prestudy")
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")

    logger.info("Loading Phi-4 tokenizer from %s", model_config["tokenizer_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_id"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    results: dict = {}

    # --- Dravidian languages -----------------------------------------------
    for language in ["kannada", "telugu"]:
        train_path = str(
            Path(data_config["data_output_dir"]) / language / "train.jsonl"
        )
        all_rows = load_jsonl(train_path)
        texts = [row["output"] for row in all_rows[:500]]
        logger.info("Analysing fertility for %s (%d texts)", language, len(texts))
        results[language] = analyze_fertility(texts, tokenizer, language)

    # --- English control ---------------------------------------------------
    english_texts = _load_english_texts(500)
    logger.info("Analysing fertility for english (%d texts)", len(english_texts))
    results["english"] = analyze_fertility(english_texts, tokenizer, "english")

    # --- Save results ------------------------------------------------------
    output_path = Path("results/module_0_prestudy/results/prestudy_fertility.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # --- Print summary table (print() is intentional here) -----------------
    print("\n" + "=" * 65)
    print(f"{'Language':<12} | {'Mean tok/word':>14} | {'Byte-fallback%':>15} | {'UNK%':>6}")
    print("-" * 12 + "-|-" + "-" * 14 + "-|-" + "-" * 15 + "-|-" + "-" * 6)
    for lang_key in ["english", "kannada", "telugu"]:
        entry = results[lang_key]
        tpw = entry["tokens_per_word"]["mean"]
        bf = entry["byte_fallback_rate"]["mean"] * 100
        unk = entry["unk_rate"]["mean"] * 100
        print(f"{lang_key:<12} | {tpw:>14.2f} | {bf:>14.1f}% | {unk:>5.2f}%")
    print("=" * 65 + "\n")

    logger.info("Fertility analysis saved to %s", output_path)


if __name__ == "__main__":
    main()
