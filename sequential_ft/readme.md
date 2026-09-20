# Module 2: Sequential FT readme.md

---
### Quick Navigation
*   **[Main Dashboard](../README.md)**
*   **[Workflow & Data Flow Guide](../readme/workflow.md)**
*   **Module Readmes**: **[Module 0: Prestudy](../prestudy/readme.md)** | **[Module 1: Monolingual FT](../monolingual_ft/readme.md)** | **[Module 2: Sequential FT](readme.md)** | **[Module 3: Multilingual Fusion](../multilingual_fusion/readme.md)**
*   **Results**: **[Prestudy Results](../readme/results/prestudy_results.md)** | **[Monolingual FT Results](../readme/results/monolingual_ft_results.md)** | **[Sequential FT Results](../readme/results/sequential_ft_results.md)** | **[Multilingual Fusion Results](../readme/results/multilingual_fusion_results.md)**
---

This module evaluates continual learning, cross-lingual transfer, and catastrophic forgetting by training the Module 1 winner (Telugu FFT) sequentially on Kannada data using a single GPU.

## Purpose
The purpose of this module is to examine how sequentially fine-tuning the winning model on a second closely-related language impacts the model's representations and performance in the original language. The Telugu FFT model (Module 1 winner, chrF 29.08) is continued on Kannada data to study whether it can acquire Kannada translation capability without losing its Telugu performance.

## Contribution to Project
*   **Bidirectional Cross-Lingual Transfer**: Measures zero-shot transfer in both directions (Kannada→Telugu and Telugu→Kannada) before any sequential training.
*   **Continual Learning Benchmarks**: Measures the degree of catastrophic forgetting (representation drift) that occurs when target distributions shift in a multi-stage training sequence.
*   **Forgetting Quantification**: Calculates precise forgetting deltas and retention rates, establishing a benchmark comparison against joint data training.
*   **English Capability Retention**: Tracks MMLU, MGSM, and GPQA scores to measure general-purpose reasoning degradation.

## How It Works

1.  **Stage 1a: Zero-Shot Telugu Evaluation (Kannada Model)**:
    *   Loads the Monolingual FT Kannada FFT checkpoint.
    *   Evaluates the Kannada model on the Telugu test set.
    *   Saves this baseline as the zero-shot cross-lingual metric (Kannada→Telugu direction).

2.  **Stage 1b: Zero-Shot Kannada Evaluation (Telugu Model)**:
    *   Loads the Monolingual FT Telugu FFT checkpoint (the Module 1 winner).
    *   Evaluates the Telugu model on the Kannada test set.
    *   Saves this baseline as the zero-shot cross-lingual metric (Telugu→Kannada direction).

3.  **Stage 2: Continuation SFT (Single GPU)**:
    *   Loads the Telugu FFT checkpoint (winner) weights and marks them as trainable.
    *   Fine-tunes the model on Kannada training data using a single GPU with DeepSpeed ZeRO-2 + CPU optimizer offloading.
    *   Saves the resulting model checkpoint to `results/finetuned_models/module_2_sequential_ft/sequential_model/`.

4.  **Stage 3: Post-FT Evaluation**:
    *   Evaluates the final sequential model on:
        *   **Kannada** test split (did it learn the new language?)
        *   **Telugu** test split (did it forget the original language?)
        *   **English benchmarks** (MMLU, MGSM, GPQA — general capability retention)
        *   **LLM-as-Judge** scoring (using the existing sarvam-m judge strategy)

5.  **Stage 4: Forgetting Analysis**:
    *   Calculates the forgetting metrics:
        *   **Retention Rate**: $TE_{\text{Sequential chrF}} / TE_{\text{Monolingual chrF}}$ — how much Telugu is retained
        *   **Forgetting Delta**: $TE_{\text{Sequential chrF}} - TE_{\text{Monolingual chrF}}$ (negative = forgetting)
        *   **BWT (Backward Transfer)**: Quantifies the influence of learning Kannada on the old task (Telugu). Negative = forgetting.
        *   **FWT (Forward Transfer)**: Quantifies zero-shot Kannada performance from the Telugu-only model. Positive = helpful Dravidian-family transfer.
        *   **PSR (Plasticity-Stability Ratio)**: Ratio balancing plasticity (learning Kannada) and stability (retaining Telugu).
    *   Saves metrics to `results/module_2_sequential_ft/results/retention_report.json`.

## Hardware Configuration
*   **Single GPU**: All training and evaluation runs use 1 GPU only (`CUDA_VISIBLE_DEVICES=0`).
*   **DeepSpeed ZeRO-2**: For FFT continuation, ZeRO-2 with CPU optimizer offloading is used to fit the 14B parameter model within a single GPU's VRAM budget.
*   **Memory Cap**: Device 0 limited to 75 GB VRAM.

## Output Artifacts
*   **Sequential Model Checkpoint**: `results/finetuned_models/module_2_sequential_ft/sequential_model/`
*   **Retention Report JSON**: `results/module_2_sequential_ft/results/retention_report.json`
*   **All Results JSON**: `results/module_2_sequential_ft/results/all_results.json`

## Executed Test Cases

After Module 2 is completed, the pytest validation gate [tests/test_sequential_ft.py](../tests/test_sequential_ft.py) executes the following test cases to ensure sequential SFT checkpoints, metrics, and retention analysis correctness:

*   **`test_sequential_checkpoint_exists`**: Confirms that the output checkpoint for sequential training (`sequential_model`) exists, contains the model files, and includes the `train_metadata.json` descriptor.
*   **`test_train_metadata_valid`**: Validates the sequential model's metadata schema (verifying `direction: telugu_to_kannada`, `training_language: kannada`, Telugu starting checkpoint, `single_gpu: true`, and finite training loss).
*   **`test_zero_shot_measurement_logged`**: Checks that BOTH bidirectional zero-shot evaluations were successfully performed and logged — Kannada model on Telugu AND Telugu model on Kannada.
*   **`test_sequential_evals_logged`**: Verifies that post-sequential training evaluation scores for both Kannada and Telugu are present, including English retention benchmarks (MMLU, MGSM, GPQA) and LLM-as-Judge scores.
*   **`test_retention_report_valid`**: Asserts that the retention report (`retention_report.json`) exists, has valid structure with correct direction (`telugu_to_kannada`), source/target languages, and that the computed Telugu retention rate is within a sensible mathematical range.
*   **`test_cross_lingual_transfer_observable`**: Performs a sanity check on both zero-shot directions to verify that cross-lingual transfer didn't degrade significantly compared to the raw foundation baseline.
