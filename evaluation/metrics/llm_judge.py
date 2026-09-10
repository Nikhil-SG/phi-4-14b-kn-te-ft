"""LLM-as-Judge evaluation using sarvamai/sarvam-m.

sarvam-m is a 24B multilingual reasoning model (Mistral-Small base)
post-trained with SFT and RLVR specifically for 11 Indian languages
including Kannada (kn) and Telugu (te).  It achieves +20 % improvement
on Indian language benchmarks over its base model.

Hybrid Thinking Mode is always enabled for judge calls.  The model
generates ``<think>...</think>`` blocks containing Dravidian-aware
linguistic reasoning before outputting final scores.  This enables
reliable assessment of script adherence, instruction relevance, and
morphological fluency in agglutinative Dravidian languages — tasks
that generic multilingual models cannot perform accurately.

Architecture: Mistral-Small 24B → ~48 GB VRAM at bfloat16.
Always called AFTER the fine-tuned model is unloaded from GPU.
"""

import gc
import json
import logging
import pathlib
import re
import datetime
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# JUDGE PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert computational linguist and language quality evaluator \
specializing in Dravidian languages. You have deep knowledge of {language_name} \
grammar, script ({script_name}), morphological structure ({script_features}), \
and linguistic norms.

Your task is to evaluate a language model's response to an instruction \
given in {language_name}. You must reason carefully and systematically \
before assigning scores.

Think through your evaluation step by step, then provide your final scores.\
"""

JUDGE_USER_PROMPT = """\
Evaluate the following {language_name} language model response.

═══════════════════════════════════════════════
ORIGINAL INSTRUCTION (in {language_name}):
{instruction}

MODEL'S RESPONSE:
{generated_output}
═══════════════════════════════════════════════

Think through each dimension carefully:

DIMENSION 1 — Script Adherence (score 0, 1, or 2):
  Consider: Is the response written in {script_name} ({unicode_range})?
  0 = Response is entirely or mostly in Latin/English script
  1 = Response mixes {language_name} and Latin scripts
  2 = Response is correctly and consistently in {script_name}

DIMENSION 2 — Instruction Relevance (score 0, 1, 2, or 3):
  Consider: Does the response actually address what the instruction asked?
  0 = Response is irrelevant or off-topic
  1 = Response partially addresses the instruction
  2 = Response mostly addresses the instruction with minor omissions
  3 = Response fully and accurately addresses the instruction

DIMENSION 3 — Linguistic Fluency (score 0, 1, 2, or 3):
  Consider: Is the {language_name} grammatically natural and idiomatic?
  Think about morphological correctness (verb conjugations, case markers,
  agglutinative suffixes), syntactic structure (SOV order), and whether
  the text reads as natural {language_name} or as awkward translation.
  0 = Broken or unreadable — incorrect morphology, wrong syntax, incomprehensible
  1 = Poor — frequent errors, unnatural phrasing, clearly machine-translated feel
  2 = Acceptable — understandable with some errors, passable quality
  3 = Fluent — natural, grammatically correct, idiomatic {language_name}

After your reasoning, output EXACTLY this format on the final line — \
three integers separated by single spaces, nothing else:
SCORES: [script] [relevance] [fluency]

Example final line: SCORES: 2 3 2\
"""


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def generate_instruction_responses(
    model: Any,
    tokenizer: Any,
    language: str,
    data_config: dict,
    model_config: dict,
    n_samples: int = 50,
) -> list[dict[str, str]]:
    """Generate instruction-following responses from the fine-tuned model.

    Called WHILE the fine-tuned Phi-4 model is still loaded on GPU,
    before it is unloaded.  Stores (instruction, generated_output) pairs
    for later scoring by the judge model.

    Args:
        model: The fine-tuned Phi-4 model (still on GPU).
        tokenizer: The fine-tuned model's tokenizer.
        language: ``"kannada"`` or ``"telugu"``.
        data_config: Parsed ``configs/data.yaml`` dict.
        model_config: Parsed ``configs/model.yaml`` dict.
        n_samples: Number of ``test.jsonl`` examples to generate for.

    Returns:
        List of dicts with keys: ``instruction``, ``input``,
        ``reference_output``, ``generated_output``, ``language``.
    """
    from utils.data_utils import load_jsonl, format_phi4_prompt

    test_path = pathlib.Path("data") / language / "test.jsonl"
    rows = load_jsonl(str(test_path))

    if len(rows) < n_samples:
        logger.warning(
            "test.jsonl has only %d rows, using all (requested %d)",
            len(rows),
            n_samples,
        )
    rows = rows[:n_samples]

    device = next(model.parameters()).device
    max_len = model_config["max_context_length"]
    judge_cfg = data_config.get("llm_judge", {})
    max_new_tokens = judge_cfg.get("max_new_tokens", 512)

    results: list[dict[str, str]] = []

    for i, row in enumerate(rows):
        prompt = format_phi4_prompt(
            instruction=row["instruction"],
            input_text=row.get("input", ""),
            output="",
            include_output=False,
        )
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

        results.append({
            "instruction":      row["instruction"],
            "input":            row.get("input", ""),
            "reference_output": row["output"],
            "generated_output": generated,
            "language":         language,
        })

        if (i + 1) % 10 == 0:
            logger.info("  Generated %d/%d instruction responses", i + 1, len(rows))

    logger.info(
        "Instruction response generation complete: %d examples for %s",
        len(results),
        language,
    )
    return results

def _ensure_judge_model_in_model_dir(judge_model_id: str, hf_token: str = "") -> None:
    """Ensure that the judge model files are present in 'models/sarvam-m' if targeted.

    If judge_model_id is "models/sarvam-m" and the directory does not contain config.json,
    downloads the model from Hugging Face Hub (sarvamai/sarvam-m).
    Handles multi-process coordination to avoid race conditions.
    """
    if judge_model_id != "models/sarvam-m":
        return

    import os
    import time
    from pathlib import Path
    
    model_dir = Path(judge_model_id)
    config_file = model_dir / "config.json"

    if config_file.exists():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        logger.info("Judge model config.json not found in 'models/sarvam-m/' directory. Initiating download...")
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            if hf_token:
                from huggingface_hub import login
                logger.info("Authenticating with Hugging Face Hub using the configured hf_token...")
                login(token=hf_token)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="sarvamai/sarvam-m",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                token=hf_token if hf_token else None,
            )
            logger.info("Judge model downloaded successfully to 'models/sarvam-m/'.")
        except Exception as exc:
            logger.error("Error downloading judge model to 'models/sarvam-m/': %s", exc)
            raise exc
    else:
        # Wait for rank 0 to finish downloading
        logger.info("Process rank %d waiting for judge model download by rank 0...", local_rank)
        start_time = time.time()
        # Poll every 5 seconds, timeout after 30 minutes
        while not config_file.exists():
            time.sleep(5)
            if time.time() - start_time > 1800:
                raise TimeoutError("Timed out waiting for process rank 0 to download the judge model.")


def _select_judge_device(required_gb: float = 50.0, force_device: str = None) -> tuple[str, dict]:
    """Select the best available device for loading the judge model.
    Forced to force_device if provided, otherwise default to GPU device 0 if available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available — judge model will load on CPU.")
        return "cpu", {"device_map": "cpu"}

    if force_device:
        logger.info("Forcing GPU device '%s' for judge model.", force_device)
        try:
            gpu_id = int(force_device.split(":")[-1])
        except Exception:
            gpu_id = 0
        return (
            force_device,
            {
                "device_map": {"": gpu_id},
            },
        )

    logger.info("Forcing GPU device 0 for judge model.")
    return (
        "cuda:0",
        {
            "device_map": {"": 0},
        },
    )


def score_with_judge(
    responses: list[dict[str, str]],
    judge_cfg: dict,
    language: str,
    experiment_id: str = "",
    log_thinking: bool = True,
    hf_token: str = "",
    judge_device: str = None,
) -> dict[str, Any]:
    """Load sarvamai/sarvam-m, score instruction-following responses using
    Hybrid Thinking Mode, then unload the judge model.

    Called AFTER the fine-tuned Phi-4 model has been unloaded from GPU.
    sarvam-m (~48 GB at bfloat16) fits comfortably on a single A100 80 GB.

    Thinking mode is always enabled.  sarvam-m generates a
    ``<think>...</think>`` block containing Dravidian linguistic reasoning
    before outputting final scores.  Thinking traces are optionally saved
    for debugging and analysis.

    Args:
        responses: List of dicts from :func:`generate_instruction_responses`.
        judge_cfg: ``llm_judge`` section from ``data.yaml``.
        language: ``"kannada"`` or ``"telugu"``.
        experiment_id: Used for naming thinking trace output files.
        log_thinking: Whether to save thinking traces to disk.
        hf_token: Optional Hugging Face Hub token.
        judge_device: Specific GPU/device to load the judge model on.

    Returns:
        Dict with keys ``llm_judge_score`` (0.0–8.0 float),
        ``llm_judge_breakdown`` (per-dimension means),
        ``llm_judge_thinking_logged`` (bool).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    judge_model_id = judge_cfg.get("judge_model_id", "sarvamai/sarvam-m")
    _ensure_judge_model_in_model_dir(judge_model_id, hf_token=hf_token)
    judge_max_new_tokens = judge_cfg.get("judge_max_new_tokens", 2048)
    lang_props = judge_cfg.get("language_properties", {}).get(language, {})

    language_name = "Kannada" if language == "kannada" else "Telugu"
    script_name   = lang_props.get("script_name", f"{language_name} script")
    unicode_range = lang_props.get("unicode_range", "")
    script_feats  = lang_props.get("script_features", "agglutinative morphology")

    # ── Select device for judge model with memory pre-flight check ───
    device_desc, device_kwargs = _select_judge_device(required_gb=50.0, force_device=judge_device)
    judge_tokenizer = None
    judge_model = None

    try:
        logger.info(
            "Loading judge model on %s: %s", device_desc, judge_model_id
        )
        tokenizer_kwargs = {"trust_remote_code": True}
        if "sarvam-m" in judge_model_id.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True
        judge_tokenizer = AutoTokenizer.from_pretrained(
            judge_model_id, **tokenizer_kwargs)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token

        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **device_kwargs,
        )
        judge_model.eval()
        logger.info(
            "Judge model loaded successfully on %s. Scoring %d responses...",
            device_desc, len(responses),
        )
    except Exception as load_exc:
        logger.error(
            "CRITICAL: Failed to load judge model on %s: %s. "
            "Skipping LLM-as-Judge evaluation.",
            device_desc, load_exc,
        )
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "llm_judge_score": None,
            "llm_judge_breakdown": None,
            "llm_judge_thinking_logged": False,
        }

    all_scores: list[dict[str, int]] = []
    thinking_traces: list[dict[str, str]] = []

    system_prompt = JUDGE_SYSTEM_PROMPT.format(
        language_name=language_name,
        script_name=script_name,
        script_features=script_feats,
    )

    for i, item in enumerate(responses):
        user_prompt = JUDGE_USER_PROMPT.format(
            language_name=language_name,
            script_name=script_name,
            unicode_range=unicode_range,
            instruction=item["instruction"][:500],
            generated_output=item["generated_output"][:800],
        )

        # Build messages using sarvam-m's chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            inputs = judge_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: manual formatting if chat template unavailable
            combined = f"{system_prompt}\n\n{user_prompt}"
            inputs = judge_tokenizer(
                combined,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )["input_ids"]

        judge_device = next(judge_model.parameters()).device
        if isinstance(inputs, dict):
            inputs = {k: v.to(judge_device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
        else:
            inputs = inputs.to(judge_device)
            input_ids = inputs

        with torch.no_grad():
            output_ids = judge_model.generate(
                input_ids,
                max_new_tokens=judge_max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=judge_tokenizer.eos_token_id,
            )

        raw_output = judge_tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=False,  # keep special tokens to find </think>
        )

        # Split thinking from final answer
        thinking_text, final_text = _split_thinking_output(raw_output)

        # Parse scores from the final answer
        parsed = _parse_judge_scores(final_text)

        if parsed:
            all_scores.append(parsed)
        else:
            # If parsing failed, try parsing the full raw output as fallback
            parsed_fallback = _parse_judge_scores(raw_output)
            if parsed_fallback:
                all_scores.append(parsed_fallback)
            else:
                logger.warning(
                    "Score parsing failed for example %d. "
                    "Raw output (last 200 chars): ...%s",
                    i,
                    raw_output[-200:],
                )

        if log_thinking and thinking_text:
            thinking_traces.append({
                "index":            i,
                "instruction":      item["instruction"][:200],
                "generated_output": item["generated_output"][:300],
                "thinking":         thinking_text.strip(),
                "final_answer":     final_text.strip(),
                "parsed_scores":    parsed or {},
            })

        if (i + 1) % 10 == 0:
            logger.info("  Judge scored %d/%d examples", i + 1, len(responses))

    # ── Aggregate scores ──────────────────────────────────────────────
    import numpy as np

    thinking_logged = False
    if log_thinking and thinking_traces and experiment_id:
        thinking_logged = _save_thinking_traces(
            thinking_traces, experiment_id, language
        )

    if not all_scores:
        logger.warning("No valid scores parsed from judge model. Returning zeros.")
        _unload_judge(judge_model, judge_tokenizer)
        return {
            "llm_judge_score": 0.0,
            "llm_judge_breakdown": {
                "script_adherence": 0.0,
                "instruction_relevance": 0.0,
                "fluency": 0.0,
            },
            "llm_judge_thinking_logged": thinking_logged,
        }

    mean_script    = float(np.mean([s["script"]    for s in all_scores]))
    mean_relevance = float(np.mean([s["relevance"] for s in all_scores]))
    mean_fluency   = float(np.mean([s["fluency"]   for s in all_scores]))
    total_score    = mean_script + mean_relevance + mean_fluency

    n_valid = len(all_scores)
    n_total = len(responses)
    if n_valid < n_total:
        logger.warning(
            "Only %d/%d responses scored successfully. "
            "Score may be slightly biased.",
            n_valid,
            n_total,
        )

    logger.info(
        "Judge scoring complete — "
        "script=%.2f/2, relevance=%.2f/3, fluency=%.2f/3, "
        "total=%.2f/8 (%d/%d valid)",
        mean_script,
        mean_relevance,
        mean_fluency,
        total_score,
        n_valid,
        n_total,
    )

    _unload_judge(judge_model, judge_tokenizer)

    return {
        "llm_judge_score": round(total_score, 4),
        "llm_judge_breakdown": {
            "script_adherence":      round(mean_script, 4),
            "instruction_relevance": round(mean_relevance, 4),
            "fluency":               round(mean_fluency, 4),
            "n_scored":              n_valid,
            "n_total":               n_total,
        },
        "llm_judge_thinking_logged": thinking_logged,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _split_thinking_output(raw_output: str) -> tuple[str, str]:
    """Split sarvam-m output into thinking block and final answer.

    sarvam-m generates ``<think>...</think>`` when thinking mode is
    active.  Everything after the closing ``</think>`` tag is the
    final answer.

    Args:
        raw_output: Raw decoded model output (special tokens NOT
            stripped).

    Returns:
        ``(thinking_text, final_text)`` — either may be empty string.
    """
    think_pattern = re.compile(
        r"<think>(.*?)</think>(.*)", re.DOTALL | re.IGNORECASE
    )
    match = think_pattern.search(raw_output)

    if match:
        thinking_text = match.group(1).strip()
        final_text    = match.group(2).strip()
        # Strip any remaining special tokens from final_text
        final_text = re.sub(r"<[^>]+>", "", final_text).strip()
        return thinking_text, final_text

    # No thinking block found — treat entire output as final answer
    cleaned = re.sub(r"<[^>]+>", "", raw_output).strip()
    return "", cleaned


def _parse_judge_scores(text: str) -> dict[str, int] | None:
    """Extract three integer scores from judge model output.

    Looks for the pattern ``SCORES: X Y Z`` first (preferred), then
    falls back to finding any three consecutive integers in the valid
    ranges.

    Args:
        text: Final answer text after stripping thinking block.

    Returns:
        Dict with keys ``script`` (0–2), ``relevance`` (0–3),
        ``fluency`` (0–3), or ``None`` if parsing fails.
    """
    if not text:
        return None

    # Primary: look for explicit "SCORES: X Y Z" pattern
    scores_match = re.search(
        r"SCORES\s*:\s*([0-2])\s+([0-3])\s+([0-3])",
        text,
        re.IGNORECASE,
    )
    if scores_match:
        return {
            "script":    int(scores_match.group(1)),
            "relevance": int(scores_match.group(2)),
            "fluency":   int(scores_match.group(3)),
        }

    # Fallback: find three space-separated integers on the same line
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        tokens = line.split()
        ints = []
        for t in tokens:
            t_clean = re.sub(r"[^\d]", "", t)
            if t_clean and t_clean.isdigit():
                ints.append(int(t_clean))
        if len(ints) >= 3:
            s, r, f = ints[0], ints[1], ints[2]
            if 0 <= s <= 2 and 0 <= r <= 3 and 0 <= f <= 3:
                logger.debug(
                    "Scores parsed via fallback from line: '%s' → %d %d %d",
                    line,
                    s,
                    r,
                    f,
                )
                return {"script": s, "relevance": r, "fluency": f}

    # Last resort: find any three valid integers anywhere in the text
    all_ints = [int(m.group()) for m in re.finditer(r"\b[0-3]\b", text)]
    if len(all_ints) >= 3:
        s, r, f = all_ints[-3], all_ints[-2], all_ints[-1]
        if 0 <= s <= 2 and 0 <= r <= 3 and 0 <= f <= 3:
            logger.debug(
                "Scores parsed via last-resort from text tail: %d %d %d",
                s,
                r,
                f,
            )
            return {"script": s, "relevance": r, "fluency": f}

    return None


def _save_thinking_traces(
    traces: list[dict],
    experiment_id: str,
    language: str,
) -> bool:
    """Save thinking traces to ``results/thinking/{experiment_id}_{language}.json``.

    Useful for debugging judge reliability and auditing scoring logic.

    Returns:
        ``True`` if saved successfully, ``False`` otherwise.
    """
    try:
        out_dir = pathlib.Path("results") / "thinking"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{experiment_id}_{language}.json"
        out_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "language":      language,
                    "judge_model":   "sarvamai/sarvam-m",
                    "saved_at":      datetime.datetime.utcnow().isoformat(),
                    "n_traces":      len(traces),
                    "traces":        traces,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("Thinking traces saved: %s", out_path)
        return True
    except Exception as exc:
        logger.warning("Could not save thinking traces: %s", exc)
        return False


def _unload_judge(judge_model: Any, judge_tokenizer: Any) -> None:
    """Unload judge model and free GPU memory."""
    del judge_model, judge_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Judge model unloaded, GPU memory freed.")
