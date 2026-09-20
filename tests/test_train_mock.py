"""Unit tests with mocking for Monolingual FT training dispatch logic.

Ensures that configs are loaded correctly, arguments are parsed, and the correct
trainer is invoked for each technique.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monolingual_ft.train import main


@pytest.fixture
def mock_configs():
    """Mock configurations loaded at runtime."""
    return {
        "model_id": "microsoft/phi-4",
        "tokenizer_id": "microsoft/phi-4",
        "dtype": "bfloat16",
    }


@patch("monolingual_ft.train.load_config")
@patch("monolingual_ft.train.checkpoint_exists")
@patch("monolingual_ft.train.evaluate_checkpoint")
@patch("monolingual_ft.train.get_checkpoint_path")
@patch("monolingual_ft.trainers.fft_trainer.run")
@patch("monolingual_ft.trainers.peft_trainer.run")
@patch("monolingual_ft.trainers.qlora_trainer.run")
def test_train_dispatch_fft(
    mock_qlora_run,
    mock_peft_run,
    mock_fft_run,
    mock_get_ckpt_path,
    mock_eval,
    mock_ckpt_exists,
    mock_load_config,
    mock_configs,
):
    """Test that fft technique parses configs and dispatches to fft_trainer."""
    mock_load_config.return_value = mock_configs
    mock_ckpt_exists.return_value = False
    mock_get_ckpt_path.return_value = Path("results/monolingual_ft/fft_kannada")

    test_args = [
        "train.py",
        "--technique", "fft",
        "--language", "kannada",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    # Assert fft trainer was invoked and others were not
    mock_fft_run.assert_called_once()
    mock_peft_run.assert_not_called()
    mock_qlora_run.assert_not_called()

    # Assert eval_checkpoint was called with module 1
    mock_eval.assert_called_once()
    args, kwargs = mock_eval.call_args
    assert kwargs["module"] == 1
    assert kwargs["technique"] == "fft"
    assert kwargs["language"] == "kannada"


@patch("monolingual_ft.train.load_config")
@patch("monolingual_ft.train.checkpoint_exists")
@patch("monolingual_ft.train.evaluate_checkpoint")
@patch("monolingual_ft.train.get_checkpoint_path")
@patch("monolingual_ft.trainers.fft_trainer.run")
@patch("monolingual_ft.trainers.peft_trainer.run")
@patch("monolingual_ft.trainers.qlora_trainer.run")
def test_train_dispatch_lora(
    mock_qlora_run,
    mock_peft_run,
    mock_fft_run,
    mock_get_ckpt_path,
    mock_eval,
    mock_ckpt_exists,
    mock_load_config,
    mock_configs,
):
    """Test that lora technique dispatches to peft_trainer."""
    mock_load_config.return_value = mock_configs
    mock_ckpt_exists.return_value = False
    mock_get_ckpt_path.return_value = Path("results/monolingual_ft/lora_kannada")

    test_args = [
        "train.py",
        "--technique", "lora",
        "--language", "kannada",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    mock_peft_run.assert_called_once()
    mock_fft_run.assert_not_called()
    mock_qlora_run.assert_not_called()


@patch("monolingual_ft.train.load_config")
@patch("monolingual_ft.train.checkpoint_exists")
@patch("monolingual_ft.train.evaluate_checkpoint")
@patch("monolingual_ft.train.get_checkpoint_path")
@patch("monolingual_ft.trainers.fft_trainer.run")
@patch("monolingual_ft.trainers.peft_trainer.run")
@patch("monolingual_ft.trainers.qlora_trainer.run")
def test_train_dispatch_qlora(
    mock_qlora_run,
    mock_peft_run,
    mock_fft_run,
    mock_get_ckpt_path,
    mock_eval,
    mock_ckpt_exists,
    mock_load_config,
    mock_configs,
):
    """Test that qlora technique dispatches to qlora_trainer."""
    mock_load_config.return_value = mock_configs
    mock_ckpt_exists.return_value = False
    mock_get_ckpt_path.return_value = Path("results/monolingual_ft/qlora_kannada")

    test_args = [
        "train.py",
        "--technique", "qlora",
        "--language", "kannada",
    ]

    with patch.object(sys, "argv", test_args):
        main()

    mock_qlora_run.assert_called_once()
    mock_peft_run.assert_not_called()
    mock_fft_run.assert_not_called()
