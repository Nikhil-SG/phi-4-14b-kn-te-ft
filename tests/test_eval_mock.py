"""Unit tests with mocking for evaluation engine.

Ensures eval_engine logic is correct (loading PEFT/base models, resetting VRAM,
calculating BLEU/chrF/perplexity/latency, appending results) without loading 14B models.
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy
import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_engine import evaluate_checkpoint


@pytest.fixture
def mock_configs():
    """Return mock model and data configurations."""
    model_config = {
        "model_id": "mock/model",
        "tokenizer_id": "mock/model",
        "dtype": "bfloat16",
        "max_context_length": 1024,
        "trust_remote_code": True,
        "use_cache": False,
        "attn_implementation": "eager",
    }
    data_config = {
        "flores_hf_path": "facebook/flores",
        "flores_language_codes": {
            "kannada": "kan_Knda",
            "telugu": "tel_Telu",
        },
        "flores_eval_split": "devtest",
        "flores_eval_samples": 10,
        "datasets": {
            "kannada": {
                "script_unicode_start": "0C80",
                "script_unicode_end": "0CFF",
            },
            "telugu": {
                "script_unicode_start": "0C00",
                "script_unicode_end": "0C7F",
            },
        },
        "results_dir": "results",
    }
    return model_config, data_config


@patch("evaluation.eval_engine.compute_english_retention")
@patch("evaluation.eval_engine.AutoTokenizer")
@patch("evaluation.eval_engine.AutoModelForCausalLM")
@patch("evaluation.eval_engine.generate_flores_predictions")
@patch("evaluation.eval_engine.compute_held_out_perplexity")
@patch("evaluation.eval_engine.load_jsonl")
@patch("evaluation.eval_engine.measure_inference_latency")
@patch("evaluation.eval_engine.append_result")
@patch("evaluation.eval_engine.compute_bertscore_muril")
@patch("evaluation.eval_engine.compute_comet")
@patch("evaluation.eval_engine.Path.exists")
def test_evaluate_checkpoint_base(
    mock_exists,
    mock_comet,
    mock_bertscore,
    mock_append_result,
    mock_measure_latency,
    mock_load_jsonl,
    mock_perplexity,
    mock_generate_predictions,
    mock_auto_model,
    mock_auto_tokenizer,
    mock_english_retention,
    mock_configs,
):
    """Test evaluate_checkpoint with a mocked base model."""
    mock_exists.return_value = False
    model_cfg, data_cfg = mock_configs

    # Setup tokenizer mock
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    mock_auto_tokenizer.from_pretrained.return_value = tokenizer

    # Setup model mock
    model = MagicMock()
    mock_auto_model.from_pretrained.return_value = model

    # Setup metrics return values
    mock_generate_predictions.return_value = {
        "hypotheses": ["ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ ಒಂದು ದೊಡ್ಡ ವಾಕ್ಯ ಇಲ್ಲಿ", "ತೆಲುಗು ವಾಕ್ಯ ಚಾಲಾ ಪೆದ್ದದಿ ಇಲ್ಲಾ"],
        "references": ["ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ ಒಂದು ದೊಡ್ಡ ವಾಕ್ಯ ಇಲ್ಲಿ", "ತೆಲುಗು ವಾಕ್ಯ ಚಾಲಾ ಪೆದ್ದದಿ ಇಲ್ಲಾ"],
        "sources": ["Write a long sentence in Kannada here", "Write a long sentence in Telugu here"],
    }
    mock_perplexity.return_value = 15.5
    mock_load_jsonl.return_value = [{"output": "ಟೆಕ್ಸ್ಟ್ 1"}, {"output": "ಟೆಕ್ಸ್ಟ್ 2"}]
    mock_measure_latency.return_value = {
        "latency_ms_mean": 120.0,
        "latency_ms_p95": 135.0,
        "tokens_per_sec": 45.0,
    }
    mock_bertscore.return_value = {"bertscore_f1": 0.85, "bertscore_precision": 0.84, "bertscore_recall": 0.86}
    mock_comet.return_value = 0.78
    mock_english_retention.return_value = {
        "mmlu_accuracy": 0.82,
        "mmlu_category_scores": {"STEM": 0.80, "Humanities": 0.81, "Social Sciences": 0.83, "Other": 0.84},
        "mgsm_en_accuracy": 0.79,
        "gpqa_accuracy": 0.54,
    }

    # Run evaluation
    result = evaluate_checkpoint(
        checkpoint_path="mock/model",
        language="kannada",
        experiment_id="prestudy_none_kannada",
        model_config=model_cfg,
        data_config=data_cfg,
        module=0,
        technique="none",
        load_in_4bit=False,
    )

    # Verify model loading kwargs
    mock_auto_model.from_pretrained.assert_called_once()
    args, kwargs = mock_auto_model.from_pretrained.call_args
    assert args[0] == "mock/model"
    assert kwargs["trust_remote_code"] is True

    # Verify metrics called
    mock_generate_predictions.assert_called_once()
    mock_perplexity.assert_called_once()
    mock_measure_latency.assert_called()
    mock_bertscore.assert_called_once()
    mock_comet.assert_called_once()

    # Verify results dict contents
    assert result["experiment_id"] == "prestudy_none_kannada"
    assert result["chrf"] > 0.0
    assert result["bleu"] > 0.0
    assert result["perplexity"] == 15.5
    assert result["bertscore_f1"] == 0.85
    assert result["comet_score"] == 0.78
    assert result["script_purity_output"] == 1.0  # pure Kannada script hypotheses
    assert result["code_switch_rate"] == 0.0  # no ASCII/Latin characters
    assert result["inference_latency_ms_b1"] == 120.0
    assert result["tokens_per_sec_b8"] == 45.0
    assert result["mmlu_accuracy"] == 0.82
    assert result["mgsm_en_accuracy"] == 0.79
    assert result["gpqa_accuracy"] == 0.54

    # Verify serialization was called
    mock_append_result.assert_called()
    args, kwargs = mock_append_result.call_args
    assert args[0] == result
    assert "results_path" in kwargs
    assert "module_0_prestudy" in kwargs["results_path"]
