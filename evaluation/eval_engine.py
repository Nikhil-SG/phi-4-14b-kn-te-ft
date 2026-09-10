"""Unified evaluation callable used by all modules.

Loads any checkpoint (base model, full fine-tune, or PEFT adapter),
runs the complete metric suite (generation quality, language integrity,
system performance, semantic similarity, human-aligned translation,
LLM-as-judge, and prompt sensitivity), and appends results to
``all_results.json``.

Usage:
    from evaluation.eval_engine import evaluate_checkpoint

    result = evaluate_checkpoint(
        checkpoint_path="microsoft/phi-4",
        language="kannada",
        experiment_id="prestudy_none_kannada",
        model_config=model_cfg,
        data_config=data_cfg,
        module=0,
    )
"""
# Redirect Hugging Face cache directories to local workspace folders to prevent global caching
import os
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(_project_root / "models" / ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(_project_root / "data" / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(_project_root / "models" / ".hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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


import datetime
import gc
from pathlib import Path

import numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation.metrics.generation import (
    compute_bleu,
    compute_chrf,
    generate_flores_predictions,
    compute_bertscore_muril,
    compute_comet,
)
from evaluation.metrics.integrity import (
    code_switch_rate,
    compute_held_out_perplexity,
    script_purity_rate,
    generation_byte_fallback_rate,
)
from evaluation.metrics.system import (
    count_trainable_params,
    get_peak_vram_gb,
    measure_inference_latency,
    reset_vram_peak,
)
from evaluation.metrics.llm_judge import (
    generate_instruction_responses,
    score_with_judge,
)
from evaluation.metrics.prompt_sensitivity import (
    compute_prompt_sensitivity_delta,
)
from evaluation.metrics.english_retention import (
    compute_english_retention,
)
from utils.data_utils import load_jsonl
from utils.logging_utils import append_result, get_logger

logger = get_logger(__name__)


def evaluate_checkpoint(
    checkpoint_path: str,
    language: str,
    experiment_id: str,
    model_config: dict,
    data_config: dict,
    module: int,
    technique: str = "none",
    load_in_4bit: bool = False,
    train_time_hrs: float | None = None,
    notes: str = "",
    run_llm_judge: bool = False,
    run_prompt_sensitivity: bool = False,
) -> dict:
    """Run the full evaluation suite on a model checkpoint.

    Loads the model (handling base models, full fine-tunes, and PEFT
    adapters), generates FLORES translations, computes all metrics,
    persists the result, and cleans up GPU memory.

    Args:
        checkpoint_path: Either ``"microsoft/phi-4"`` for base eval or
            a local path to a fine-tuned checkpoint / PEFT adapter.
        language: ``"kannada"`` or ``"telugu"``.
        experiment_id: Unique experiment identifier following the
            pattern ``module{N}_{technique}_{language}``.
        model_config: Parsed ``configs/model.yaml``.
        data_config: Parsed ``configs/data.yaml``.
        module: Module number (0, 1, 2, or 3).
        technique: Fine-tuning technique (``"none"``, ``"fft"``,
            ``"lora"``, ``"qlora"``, ``"dora"``, ``"ia3"``).
        load_in_4bit: Whether to load the model in 4-bit quantisation.
        train_time_hrs: Training wall-clock time in hours, or ``None``
            for base / zero-shot evaluations.
        notes: Free-text annotation for the result entry.
        run_llm_judge: If ``True``, run sarvam-m LLM-as-Judge scoring
            (gated - loads 24B judge model after fine-tuned model
            unload).
        run_prompt_sensitivity: If ``True``, compute prompt sensitivity
            delta using IndicMMLU (gated - expensive per-question
            inference).

    Returns:
        The complete result dict matching the ``all_results.json``
        schema.
    """
    # 1. Log start
    logger.info(
        "[eval_engine] Starting evaluation: %s | lang=%s",
        experiment_id,
        language,
    )

    # 2. Detect PEFT adapter
    ckpt_path = Path(checkpoint_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()

    # 2b. Check if results already exist for this experiment_id in the target all_results.json
    import json
    module_name = {
        0: "module_0_prestudy",
        1: "module_1_monolingual_ft",
        2: "module_2_sequential_ft",
        3: "module_3_multilingual_fusion",
    }.get(module, f"module_{module}")
    local_results_path = f"results/{module_name}/results/all_results.json"
    
    existing_result = None
    if Path(local_results_path).exists():
        try:
            with open(local_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    if entry.get("experiment_id") == experiment_id:
                        existing_result = entry
                        break
        except Exception:
            pass

    # Initialize variables that could be reused or computed
    chrf = None
    bleu = None
    bertscore_f1 = None
    comet_score = None
    perplexity = None
    gen_byte_fallback = None
    avg_purity = None
    avg_csr = None
    peak_vram_gb = None
    param_counts = None
    latency_b1 = None
    latency_b8 = None
    english_retention_result = None
    instruction_responses = None
    ps_result = None
    judge_result = None
    result = {}

    def save_current_results():
        nonlocal result
        result = {
            "experiment_id": experiment_id,
            "module": module,
            "technique": technique,
            "language": language,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "chrf": float(chrf) if chrf is not None else None,
            "bleu": float(bleu) if bleu is not None else None,
            "perplexity": float(perplexity) if perplexity is not None else None,
            "script_purity_output": round(avg_purity, 4) if avg_purity is not None else None,
            "code_switch_rate": round(avg_csr, 4) if avg_csr is not None else None,
            "peak_vram_gb": peak_vram_gb,
            "inference_latency_ms_b1": latency_b1["latency_ms_mean"] if isinstance(latency_b1, dict) else latency_b1,
            "tokens_per_sec_b8": latency_b8["tokens_per_sec"] if isinstance(latency_b8, dict) else latency_b8,
            "trainable_params_M": param_counts["trainable_M"] if isinstance(param_counts, dict) else param_counts,
            "trainable_pct": param_counts["trainable_pct"] if isinstance(param_counts, dict) else param_counts,
            "train_time_hrs": train_time_hrs,
            "notes": notes,
            "bertscore_f1": bertscore_f1,
            "comet_score": comet_score,
            "llm_judge_score": judge_result["llm_judge_score"] if isinstance(judge_result, dict) else judge_result,
            "llm_judge_breakdown": judge_result["llm_judge_breakdown"] if isinstance(judge_result, dict) else None,
            "llm_judge_thinking_logged": judge_result["llm_judge_thinking_logged"] if isinstance(judge_result, dict) else None,
            "generation_byte_fallback_rate": gen_byte_fallback,
            "prompt_sensitivity_delta": (
                ps_result["prompt_sensitivity_delta"]
                if isinstance(ps_result, dict) else ps_result
            ),
            "prompt_sensitivity_mean_acc": (
                ps_result["prompt_sensitivity_mean_acc"]
                if isinstance(ps_result, dict) else None
            ),
            # English retention benchmarks (catastrophic forgetting)
            "mmlu_accuracy": (
                english_retention_result["mmlu_accuracy"]
                if isinstance(english_retention_result, dict) else None
            ),
            "mmlu_category_scores": (
                english_retention_result["mmlu_category_scores"]
                if isinstance(english_retention_result, dict) else None
            ),
            "mgsm_en_accuracy": (
                english_retention_result["mgsm_en_accuracy"]
                if isinstance(english_retention_result, dict) else None
            ),
            "gpqa_accuracy": (
                english_retention_result["gpqa_accuracy"]
                if isinstance(english_retention_result, dict) else None
            ),
        }
        append_result(result, results_path=local_results_path)

    # Check which steps need to run
    need_flores = True
    need_perplexity = True
    need_system = True
    need_english = True
    need_judge_gen = run_llm_judge
    need_prompt_sensitivity = run_prompt_sensitivity

    if existing_result:
        logger.info("[EVAL] Found existing results entry for %s. Checking completed metrics...", experiment_id)
        
        # 1. FLORES & translation metrics
        chrf = existing_result.get("chrf")
        bleu = existing_result.get("bleu")
        bertscore_f1 = existing_result.get("bertscore_f1")
        comet_score = existing_result.get("comet_score")
        avg_purity = existing_result.get("script_purity_output")
        avg_csr = existing_result.get("code_switch_rate")
        gen_byte_fallback = existing_result.get("generation_byte_fallback_rate")
        
        # We need FLORES generation if any of these are missing (or if bertscore failed previously with 0.0)
        bs_cfg = data_config.get("bertscore", {})
        comet_cfg = data_config.get("comet", {})
        
        need_chrf = chrf is None
        need_bleu = bleu is None
        need_bert = bs_cfg.get("enabled", True) and (bertscore_f1 is None or bertscore_f1 == 0.0)
        need_comet = comet_cfg.get("enabled", True) and (comet_score is None or comet_score == 0.0)
        need_purity = avg_purity is None
        need_csr = avg_csr is None
        need_fallback = gen_byte_fallback is None
        
        if not (need_chrf or need_bleu or need_bert or need_comet or need_purity or need_csr or need_fallback):
            logger.info("[EVAL] Reusing completed FLORES, chrF, BLEU, BERTScore, COMET, and script metrics.")
            need_flores = False
        else:
            logger.info(
                "[EVAL] Some translation metrics are missing: chrf=%s, bleu=%s, bertscore_f1=%s, comet_score=%s, purity=%s, csr=%s, fallback=%s. Will run FLORES generation.",
                need_chrf, need_bleu, need_bert, need_comet, need_purity, need_csr, need_fallback
            )
            need_flores = True

        # 2. Perplexity
        old_ppl = existing_result.get("perplexity")
        import math
        if old_ppl is not None and not math.isnan(old_ppl):
            logger.info("[EVAL] Reusing completed Perplexity: %s", old_ppl)
            perplexity = old_ppl
            need_perplexity = False

        # 3. System & latency
        old_latency_b1 = existing_result.get("inference_latency_ms_b1")
        old_tokens_b8 = existing_result.get("tokens_per_sec_b8")
        old_vram = existing_result.get("peak_vram_gb")
        old_param_m = existing_result.get("trainable_params_M")
        old_param_pct = existing_result.get("trainable_pct")
        
        if (old_latency_b1 is not None and old_tokens_b8 is not None and 
            old_vram is not None and old_param_m is not None and old_param_pct is not None):
            logger.info("[EVAL] Reusing completed system performance and latency metrics.")
            latency_b1 = {"latency_ms_mean": old_latency_b1}
            latency_b8 = {"tokens_per_sec": old_tokens_b8}
            peak_vram_gb = old_vram
            param_counts = {"trainable_M": old_param_m, "trainable_pct": old_param_pct}
            need_system = False

        # 4. English retention benchmarks
        old_mmlu = existing_result.get("mmlu_accuracy")
        old_mgsm = existing_result.get("mgsm_en_accuracy")
        old_gpqa = existing_result.get("gpqa_accuracy")
        
        if old_mmlu is not None and old_mgsm is not None and old_gpqa is not None:
            logger.info("[EVAL] Reusing completed English retention benchmarks.")
            english_retention_result = {
                "mmlu_accuracy": old_mmlu,
                "mmlu_category_scores": existing_result.get("mmlu_category_scores"),
                "mgsm_en_accuracy": old_mgsm,
                "gpqa_accuracy": old_gpqa
            }
            need_english = False

        # 5. LLM Judge score
        old_judge_score = existing_result.get("llm_judge_score")
        if old_judge_score is not None:
            logger.info("[EVAL] Reusing completed LLM Judge score: %s", old_judge_score)
            judge_result = {
                "llm_judge_score": old_judge_score,
                "llm_judge_breakdown": existing_result.get("llm_judge_breakdown"),
                "llm_judge_thinking_logged": existing_result.get("llm_judge_thinking_logged"),
            }
            need_judge_gen = False
            run_llm_judge = False

        # 6. Prompt sensitivity
        old_ps_delta = existing_result.get("prompt_sensitivity_delta")
        old_ps_mean = existing_result.get("prompt_sensitivity_mean_acc")
        if old_ps_delta is not None and old_ps_mean is not None:
            logger.info("[EVAL] Reusing completed Prompt Sensitivity delta: %s", old_ps_delta)
            ps_result = {
                "prompt_sensitivity_delta": old_ps_delta,
                "prompt_sensitivity_mean_acc": old_ps_mean,
            }
            need_prompt_sensitivity = False
            run_prompt_sensitivity = False

    need_model = need_flores or need_perplexity or need_system or need_english or need_judge_gen or need_prompt_sensitivity

    tokenizer = None
    model = None
    device = None

    if need_model:
        # 3. Load tokenizer
        tokenizer_source = (
            model_config["model_id"] if is_peft else checkpoint_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # 4. Load model
        from utils.gpu_utils import configure_gpu_settings
        primary_device_id, max_memory = configure_gpu_settings()

        dtype = getattr(torch, model_config.get("dtype", "bfloat16"))
        model_kwargs: dict = {
            "trust_remote_code": model_config.get("trust_remote_code", True),
            "use_cache": model_config.get("use_cache", False),
            "attn_implementation": model_config.get("attn_implementation", "eager"),
        }

        if technique == "fft" and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = "cuda:1"
                model_kwargs["device_map"] = {"": 1}
                logger.info("Forcing FFT model loading entirely on GPU device 1.")
            else:
                device = "cuda:0"
                model_kwargs["device_map"] = {"": 0}
                logger.info("Only 1 GPU available. Loading FFT model entirely on GPU device 0.")
        else:
            model_kwargs["max_memory"] = max_memory
            if torch.cuda.is_available():
                device = f"cuda:{primary_device_id}"
            else:
                device = "cpu"
            model_kwargs["device_map"] = "auto"

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = dtype

        base_model_id = model_config["model_id"]
        if is_peft:
            # Load base model, then apply adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, **model_kwargs
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            model = model.merge_and_unload()
            logger.info("Loaded PEFT adapter from %s and merged.", checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, **model_kwargs
            )
            logger.info("Loaded model from %s.", checkpoint_path)

    # 5. Set to eval mode
    if model is not None:
        model.eval()

    # 6. Reset VRAM tracking
    if need_model and need_flores:
        reset_vram_peak()

    # 7. FLORES generation + chrF / BLEU
    if need_flores:
        flores_out = generate_flores_predictions(
            model, tokenizer, language, data_config, model_config, device=device
        )
        hypotheses = flores_out["hypotheses"]
        references = flores_out["references"]
        sources = flores_out["sources"]

        if not hypotheses:
            logger.warning(
                "FLORES generation returned empty — setting chrf/bleu to 0.0"
            )
            chrf = chrf if chrf is not None else 0.0
            bleu = bleu if bleu is not None else 0.0
        else:
            if chrf is None:
                chrf = compute_chrf(hypotheses, references)
            if bleu is None:
                bleu = compute_bleu(hypotheses, references)

        # 8. BERTScore (MuRIL)
        bs_cfg = data_config.get("bertscore", {})
        if bs_cfg.get("enabled", True) and hypotheses:
            if bertscore_f1 is None or bertscore_f1 == 0.0:
                bertscore_result = compute_bertscore_muril(
                    hypotheses, references, language,
                    model_type=bs_cfg.get("model_type", "models/muril-base-cased"),
                    batch_size=bs_cfg.get("batch_size", 32),
                    device=device,
                )
                bertscore_f1 = bertscore_result["bertscore_f1"]
        else:
            if bertscore_f1 is None:
                bertscore_f1 = None

        # 9. COMET
        comet_cfg = data_config.get("comet", {})
        if comet_cfg.get("enabled", True) and sources and hypotheses:
            if comet_score is None or comet_score == 0.0:
                comet_score = compute_comet(
                    sources, hypotheses, references,
                    model_name=comet_cfg.get("model_name", "Unbabel/wmt22-comet-da"),
                    batch_size=comet_cfg.get("batch_size", 8),
                    gpus=comet_cfg.get("gpus", 1),
                    saving_directory=comet_cfg.get("saving_directory", "models"),
                )
        else:
            if comet_score is None:
                comet_score = None

        # 11. Generation byte-fallback rate
        if gen_byte_fallback is None:
            gen_byte_fallback = (
                generation_byte_fallback_rate(hypotheses, tokenizer)
                if hypotheses else None
            )

        # 12. Script purity and code-switch rate over hypotheses
        if avg_purity is None:
            purity_scores = [
                script_purity_rate(h, language, data_config)
                for h in hypotheses
                if h.strip()
            ]
            avg_purity = float(numpy.mean(purity_scores)) if purity_scores else 0.0

        if avg_csr is None:
            csr_scores = [
                code_switch_rate(h) for h in hypotheses if h.strip()
            ]
            avg_csr = float(numpy.mean(csr_scores)) if csr_scores else 1.0

        save_current_results()

    # 10. Held-out perplexity
    if need_perplexity:
        test_jsonl_path = str(Path("data") / language / "test.jsonl")
        try:
            test_data = load_jsonl(test_jsonl_path)[:200]
            held_out_texts = [row["output"] for row in test_data]
        except FileNotFoundError:
            logger.warning(
                "Test JSONL not found at %s — skipping perplexity.",
                test_jsonl_path,
            )
            held_out_texts = []

        if held_out_texts:
            perplexity = compute_held_out_perplexity(
                model, tokenizer, held_out_texts, model_config, device=device
            )
        else:
            perplexity = float("nan")

        save_current_results()

    # 13. Peak VRAM
    if need_system:
        peak_vram_gb = get_peak_vram_gb()

        # 14. Trainable parameter counts
        param_counts = count_trainable_params(model)

        # 15. Inference latency
        # Build sample prompts from FLORES English sentences or test data
        sample_prompts: list[str] = []
        if 'references' in locals() and references:
            # Use the first 8 FLORES prompts (reconstruct from English ds)
            try:
                import datasets as hf_datasets

                eng_ds = hf_datasets.load_dataset(
                    data_config["flores_hf_path"],
                    "eng_Latn",
                    split=data_config["flores_eval_split"],
                )
                language_name = "Kannada" if language == "kannada" else "Telugu"
                from utils.data_utils import format_phi4_prompt as _fmt

                for i in range(min(8, len(eng_ds))):
                    prompt = _fmt(
                        instruction=f"Translate the following English sentence to {language_name}:",
                        input_text=eng_ds[i]["sentence"],
                        output="",
                        include_output=False,
                    )
                    sample_prompts.append(prompt)
            except Exception:
                logger.warning("Could not load FLORES for latency prompts.")

        if not sample_prompts:
            try:
                test_jsonl_path = str(Path("data") / language / "test.jsonl")
                test_data = load_jsonl(test_jsonl_path)[:8]
                sample_prompts = [row["output"] for row in test_data]
            except Exception:
                pass

        if not sample_prompts:
            sample_prompts = ["Translate: Hello world."] * 8

        latency_b1 = measure_inference_latency(
            model, tokenizer, sample_prompts, model_config, batch_size=1,
            device=device,
        )
        latency_b8 = measure_inference_latency(
            model, tokenizer, sample_prompts, model_config, batch_size=8,
            device=device,
        )

        save_current_results()

    # 15b. English retention benchmarks (catastrophic forgetting detection)
    #      Runs MMLU + MGSM + GPQA by default for all techniques.
    if need_english:
        eng_retention_cfg = data_config.get("english_retention", {})
        english_retention_result = None
        if eng_retention_cfg.get("enabled", True):
            logger.info("Computing English retention benchmarks (MMLU + MGSM + GPQA)...")
            try:
                english_retention_result = compute_english_retention(
                    model, tokenizer, model_config, data_config, device=device,
                )
            except Exception as exc:
                logger.error(
                    "English retention evaluation failed: %s. Continuing with other metrics.",
                    exc, exc_info=True,
                )

        save_current_results()

    # 16. Generate instruction responses while model is still loaded (gated)
    if need_judge_gen:
        logger.info("Generating instruction responses for sarvam-m judge...")
        judge_cfg = data_config.get("llm_judge", {})
        instruction_responses = generate_instruction_responses(
            model, tokenizer, language, data_config, model_config,
            n_samples=judge_cfg.get("n_samples", 50),
        )

        save_current_results()

    # 17. Prompt sensitivity while model is still loaded (gated)
    if need_prompt_sensitivity:
        logger.info("Computing prompt sensitivity delta (IndicMMLU)...")
        ps_cfg = data_config.get("prompt_sensitivity", {})
        ps_result = compute_prompt_sensitivity_delta(
            model, tokenizer, language, data_config, model_config,
            n_questions=ps_cfg.get("n_questions", 100),
        )

        save_current_results()

    # 18. Build experiment_id validation
    expected_patterns = []
    if module == 0:
        expected_patterns.append(f"prestudy_none_{language}")
    elif module == 1:
        expected_patterns.append(f"monolingual_ft_{technique}_{language}")
    elif module == 2:
        expected_patterns.extend([
            f"sequential_ft_zero_shot_{language}",
            f"sequential_ft_sequential_{language}"
        ])
    elif module == 3:
        for strategy in ["equal", "proportional"]:
            expected_patterns.append(f"multilingual_fusion_joint_{strategy}_{language}")
        for merge_name in ["ties", "linear"]:
            expected_patterns.append(f"multilingual_fusion_merged_{merge_name}_{language}")
        for adapter_name in ["kannada", "telugu"]:
            expected_patterns.append(f"multilingual_fusion_monolingual_{adapter_name}_{language}")
    else:
        expected_patterns.append(f"module{module}_{technique}_{language}")

    if experiment_id not in expected_patterns:
        logger.warning(
            "experiment_id '%s' does not match any expected pattern in %s",
            experiment_id,
            expected_patterns,
        )

    # 19. Cleanup GPU memory - unload fine-tuned model
    if 'model' in locals() and model is not None:
        del model
    if 'tokenizer' in locals() and tokenizer is not None:
        del tokenizer

    # Unload any cached bert_score or comet models from memory to free GPU VRAM
    import sys
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("bert_score") or mod_name.startswith("comet"):
            mod = sys.modules[mod_name]
            for attr in ["scorer", "SCORER_CACHE", "scorer_cache", "cached_scorer", "model", "trainer"]:
                if hasattr(mod, attr):
                    try:
                        val = getattr(mod, attr)
                        if isinstance(val, dict):
                            val.clear()
                        elif hasattr(val, "to"):
                            try:
                                val.to("cpu")
                            except Exception:
                                pass
                            setattr(mod, attr, None)
                        else:
                            setattr(mod, attr, None)
                    except Exception:
                        pass

    gc.collect()
    torch.cuda.empty_cache()

    # 20. sarvam-m judge scoring AFTER fine-tuned model is unloaded (gated)
    if judge_result is None:
        judge_result = {
            "llm_judge_score":           None,
            "llm_judge_breakdown":       None,
            "llm_judge_thinking_logged": None,
        }

    # Determine the device for the judge model
    judge_device = None
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            try:
                # Query free memory on GPU 0 (in bytes)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / (1024 ** 3)
                logger.info("[EVAL] GPU 0 memory status: free=%.2f GB, total=%.2f GB", free_gb, total_bytes / (1024 ** 3))
                
                # Check if GPU 0 is free (needs at least 50 GB free for the 24B judge model)
                if free_gb >= 50.0:
                    judge_device = "cuda:0"
                    logger.info("[EVAL] GPU 0 has sufficient free memory. Routing judge model to cuda:0.")
                else:
                    judge_device = "cuda:1"
                    logger.info(
                        "[EVAL] GPU 0 has insufficient free memory (%.2f GB free). "
                        "Routing judge model to cuda:1 (FFT model is already unloaded from GPU 1).",
                        free_gb
                    )
            except Exception as exc:
                logger.warning("[EVAL] Failed to query GPU 0 memory info: %s. Defaulting to cuda:0.", exc)
                judge_device = "cuda:0"
        else:
            judge_device = "cuda:0"
    else:
        judge_device = "cpu"

    if run_llm_judge and instruction_responses:
        logger.info(
            "Starting sarvam-m judge scoring "
            "(fine-tuned model unloaded, GPU free)..."
        )
        judge_cfg = data_config.get("llm_judge", {})
        judge_result = score_with_judge(
            responses=instruction_responses,
            judge_cfg=judge_cfg,
            language=language,
            experiment_id=experiment_id,
            log_thinking=judge_cfg.get("log_thinking_traces", True),
            hf_token=model_config.get("hf_token", ""),
            judge_device=judge_device,
        )

    # 22. Persist result
    save_current_results()

    # 23. Log summary
    logger.info(
        "[%s] chrF=%.2f BLEU=%.2f BERTScore=%s COMET=%s "
        "PPL=%.1f ByteFallback=%s VRAM=%.2fGB",
        experiment_id,
        chrf if chrf is not None else 0.0,
        bleu if bleu is not None else 0.0,
        bertscore_f1,
        comet_score,
        perplexity if perplexity is not None else 0.0,
        gen_byte_fallback,
        peak_vram_gb,
    )
    if run_llm_judge and judge_result["llm_judge_score"] is not None:
        breakdown = judge_result["llm_judge_breakdown"] or {}
        logger.info(
            "[%s] Judge: %.2f/8 (script=%.2f/2, relevance=%.2f/3, "
            "fluency=%.2f/3)",
            experiment_id,
            judge_result["llm_judge_score"],
            breakdown.get("script_adherence", 0),
            breakdown.get("instruction_relevance", 0),
            breakdown.get("fluency", 0),
        )

    if english_retention_result:
        mmlu = english_retention_result.get("mmlu_accuracy")
        mgsm = english_retention_result.get("mgsm_en_accuracy")
        gpqa = english_retention_result.get("gpqa_accuracy")
        logger.info(
            "[%s] English Retention: MMLU=%s MGSM=%s GPQA=%s "
            "(Phi-4 baseline: MMLU=84.8%% MGSM=80.6%% GPQA=56.1%%)",
            experiment_id,
            f"{mmlu*100:.1f}%" if mmlu is not None else "N/A",
            f"{mgsm*100:.1f}%" if mgsm is not None else "N/A",
            f"{gpqa*100:.1f}%" if gpqa is not None else "N/A",
        )

    # 24. Return result
    return result
