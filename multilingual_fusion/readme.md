# Module 3: Multilingual Fusion readme.md

---
### Quick Navigation
*   **[Main Dashboard](../README.md)**
*   **[Workflow & Data Flow Guide](../readme/workflow.md)**
*   **Module Readmes**: **[Module 0: Prestudy](../prestudy/readme.md)** | **[Module 1: Monolingual FT](../monolingual_ft/readme.md)** | **[Module 2: Sequential FT](../sequential_ft/readme.md)** | **[Module 3: Multilingual Fusion](readme.md)**
*   **Results**: **[Prestudy Results](../readme/results/prestudy_results.md)** | **[Monolingual FT Results](../readme/results/monolingual_ft_results.md)** | **[Sequential FT Results](../readme/results/sequential_ft_results.md)** | **[Multilingual Fusion Results](../readme/results/multilingual_fusion_results.md)**
---

This module compares data-level joint training (bilingual SFT) against post-hoc parameter-level weight merging (TIES-Merge and Linear average) of single-language adapters.

## Purpose
The purpose of this module is to investigate bilingual consolidation strategies. We contrast:
1.  **Bilingual Joint SFT**: Training a single model on interleaved Kannada and Telugu datasets.
2.  **Adapter Weight Fusion**: Merging separate single-language PEFT adapters directly in weight-space without any additional training steps.

## Contribution to Project
*   **Zero-Training Multilingualism**: Evaluates post-hoc merging (TIES-Merge and Linear blending) as a compute-free alternative to joint SFT.
*   **Interference Analysis**: Quantifies performance deltas between joint data SFT and parameter-space merging to reveal which approach minimizes cross-lingual interference.

## How It Works

1.  **Bilingual Joint SFT (`joint_train.py`)**:
    *   Interleaves Kannada and Telugu training datasets.
    *   Applies two sampling ratios:
        *   **Equal**: 50% Kannada / 50% Telugu samples.
        *   **Proportional**: Probabilities are weighted by the relative size of Kannada and Telugu splits.
    *   Fine-tunes the base model on this combined dataset.
    *   Saves checkpoints to `results/finetuned_models/module_3_multilingual_fusion/joint_{equal|proportional}`.

2.  **Post-Hoc Weight Merging (`adapter_merge.py`)**:
    *   Loads the separate Kannada and Telugu monolingual adapters trained in Module 1.
    *   Combines their weights on the CPU:
        *   **Linear average**: Parameter-wise average: $W_{\text{merged}} = 0.5 \cdot W_{\text{KN}} + 0.5 \cdot W_{\text{TE}}$.
        *   **TIES-Merge**: Trims parameters to 50% density, creates a sign agreement mask, and merges disjoint parameters to resolve representation interference.
    *   Saves merged models to `results/finetuned_models/module_3_multilingual_fusion/merged_{ties|linear}`.

3.  **Bilingual Evaluation**:
    *   Evaluates all 4 resulting models on Kannada and Telugu test splits using the expanded metric suite (chrF++, BLEU, BERTScore F1, COMET, and generation byte-fallback rate).
    *   Saves summary report to `results/module_3_multilingual_fusion/results/merge_summary.json`.

## Output Artifacts
*   **Joint Checkpoints**: `results/finetuned_models/module_3_multilingual_fusion/joint_equal/` and `joint_proportional/`
*   **Merged Adapter Weights**: `results/finetuned_models/module_3_multilingual_fusion/merged_ties/` and `merged_linear/`
*   **Merge Summary JSON**: `results/module_3_multilingual_fusion/results/merge_summary.json`

## Executed Test Cases

After Module 3 is completed, the pytest validation gate [tests/test_multilingual_fusion.py](../tests/test_multilingual_fusion.py) executes the following test cases to ensure joint fine-tuning, weight-space merging, and visualization script integrity:

*   **`test_joint_checkpoints_exist`**: Verifies that bilingual joint training outputs (`joint_equal` and `joint_proportional` checkpoints) are successfully written and contain `train_metadata.json` metadata.
*   **`test_merged_adapters_exist`**: Verifies that post-hoc merged adapter directories (`merged_ties` and `merged_linear`) exist and contain required PEFT weight files (`adapter_model.safetensors` and `adapter_config.json`).
*   **`test_multilingual_fusion_results_logged`**: Confirms that all 8 expected multilingual fusion experiment outputs (4 models $\times$ 2 languages) are successfully evaluated and stored in `all_results.json`.
*   **`test_no_nan_in_multilingual_fusion_metrics`**: Validates that no metrics recorded for bilingual fusion are NaN or corrupted.
*   **`test_interference_matrix_data_complete`**: Asserts that all evaluation benchmarks across all 4 modules are complete, ensuring that the cross-lingual interference matrix can be visualised without missing details.
*   **`test_merge_summary_exists`**: Verifies that `merge_summary.json` is generated with valid references to the source monolingual adapters and the paths of the resulting merged models.
*   **`test_viz_scripts_importable`**: Verifies that all Pareto, interference matrix, loss curves, and vocabulary fertility visualization modules are importable and contain standard entry points (`main()`).
