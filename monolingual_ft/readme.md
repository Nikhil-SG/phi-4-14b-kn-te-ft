# Module 1: Monolingual FT readme.md

---
### Quick Navigation
*   **[Main Dashboard](../README.md)**
*   **[Workflow & Data Flow Guide](../readme/workflow.md)**
*   **Module Readmes**: **[Module 0: Prestudy](../prestudy/readme.md)** | **[Module 1: Monolingual FT](readme.md)** | **[Module 2: Sequential FT](../sequential_ft/readme.md)** | **[Module 3: Multilingual Fusion](../multilingual_fusion/readme.md)**
*   **Results**: **[Prestudy Results](../readme/results/prestudy_results.md)** | **[Monolingual FT Results](../readme/results/monolingual_ft_results.md)** | **[Sequential FT Results](../readme/results/sequential_ft_results.md)** | **[Multilingual Fusion Results](../readme/results/multilingual_fusion_results.md)**
---

This module benchmarks 5 fine-tuning techniques (FFT, LoRA, DoRA, QLoRA, and IA³) individually on Kannada and Telugu datasets to identify the best overall SFT configuration.

## Purpose
The purpose of this module is to conduct a Cartesian benchmarking study (5 techniques $\times$ 2 languages = 10 SFT experiments) to analyze the trade-off between resource consumption (trainable parameter count, peak VRAM usage) and translation performance (using chrF++, BLEU, BERTScore F1, COMET, and generation byte-fallback rate).

## Contribution to Project
*   **Optimal Configuration Selection**: Identifies which tuning technique achieves the highest quality metric per unit of hardware resource.
*   **Winner Exportation**: Automatically outputs the Pareto-optimal config to `monolingual_ft_winner.yaml`, which is consumed dynamically by subsequent pipeline runs.

## How It Works

1.  **Orchestration & Dispatch (`train.py`)**:
    *   Accepts `--technique` and `--language` CLI arguments.
    *   Loads configs and dispatches training execution to the matching technique trainer.

2.  **Technique Sub-Trainers (`trainers/`)**:
    *   **Full Parameter SFT (`fft_trainer.py`)**: Loads the model in 16-bit precision and distributes training across multiple GPUs utilizing DeepSpeed ZeRO-3 optimization.
    *   **PEFT-based SFT (`peft_trainer.py`)**: Dynamically injects LoRA, DoRA, or IA³ parameter adapters.
    *   **QLoRA-based SFT (`qlora_trainer.py`)**: Loads the foundation model in 4-bit NF4 double-quantization, training low-rank adapters on a single GPU.

3.  **Pareto Frontier Selection (`tests/test_monolingual_ft.py`)**:
    *   Runs as a pytest post-gate.
    *   Computes a Pareto score: $\text{Score} = \text{chrF} / \text{VRAM}$.
    *   Identifies the winning configuration and writes the YAML file `results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml`.

## Output Artifacts
*   **Fine-Tuned Checkpoints**: Saved under `results/finetuned_models/module_1_monolingual_ft/{technique}_{language}/` (contains adapter/model weights, configuration files, and `train_metadata.json`).
*   **Pareto Winner YAML**: `results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml`
*   **Evaluation metrics**: Logs added to the main results file.

## Executed Test Cases

After Module 1 is completed, the pytest validation gate [tests/test_monolingual_ft.py](../tests/test_monolingual_ft.py) executes the following test cases to ensure configuration, checkpoint, and metrics integrity:

*   **`test_all_checkpoints_exist`**: Verifies that all 10 fine-tuning checkpoint directories (5 SFT techniques $\times$ 2 languages) exist and contain valid model weights/configurations.
*   **`test_all_results_logged`**: Asserts that evaluation metrics for all 10 completed SFT runs are successfully appended to the central `results/module_1_monolingual_ft/results/all_results.json` file.
*   **`test_no_nan_in_metrics`**: Confirms that all logged metrics (chrF++, BLEU, peak VRAM, and trainable parameters) are valid, finite, and non-NaN values.
*   **`test_train_metadata_exists`**: Verifies that `train_metadata.json` exists for every run, containing training loss, time elapsed, and hyperparameter metrics.
*   **`test_pareto_winner_written`**: Calculates the Pareto frontier based on the quality-to-VRAM ratio (chrF / peak VRAM) across Kannada and Telugu, selects the winning SFT config, writes it to `monolingual_ft_winner.yaml`, and asserts that the file is well-formed.
