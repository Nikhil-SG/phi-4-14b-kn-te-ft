# Module 0: Prestudy readme.md

---
### Quick Navigation
*   **[Main Dashboard](../README.md)**
*   **[Workflow & Data Flow Guide](../readme/workflow.md)**
*   **Module Readmes**: **[Module 0: Prestudy](readme.md)** | **[Module 1: Monolingual FT](../monolingual_ft/readme.md)** | **[Module 2: Sequential FT](../sequential_ft/readme.md)** | **[Module 3: Multilingual Fusion](../multilingual_fusion/readme.md)**
*   **Results**: **[Prestudy Results](../readme/results/prestudy_results.md)** | **[Monolingual FT Results](../readme/results/monolingual_ft_results.md)** | **[Sequential FT Results](../readme/results/sequential_ft_results.md)** | **[Multilingual Fusion Results](../readme/results/multilingual_fusion_results.md)**
---

This module handles dataset download, text normalization, script-filtering, deduplication, tokenizer fertility analysis, and baseline evaluation of the unmodified `microsoft/phi-4` model.

## Purpose
The purpose of this module is to establish a high-quality, leak-free data foundation and map the zero-shot capabilities of the base model on Kannada and Telugu translation. This sets up the control group metrics for the entire study.

## Contribution to Project
*   **Data Quality**: Filters raw inputs to isolate pure script boundaries and filters out noisy data.
*   **Leakage Prevention**: Employs SHA-256 content hashing to ensure zero duplicate instruction-output pairs exist between training and testing splits.
*   **Fertility Benchmark**: Quantifies how well the model's vocabulary represents Dravidian subwords, highlighting token-inflation challenges compared to English.

## How It Works

1.  **Dataset Preparation (`dataset_prep.py`)**:
    *   Downloads the raw `ai4bharat/indic-align` dataset.
    *   Filters to high-hygiene subsets only: `Dolly_T` (15,000 rows/lang) and `OpenAssistant_T` (19,900 rows/lang).
    *   Filters by language-specific Unicode script blocks:
        *   Kannada range: `0C80`–`0CFF`
        *   Telugu range: `0C00`–`0C7F`
    *   Deduplicates using SHA-256 hashes of `instruction + output`.
    *   Saves clean splits (`train.jsonl`, `val.jsonl`, `test.jsonl`) under `data/{language}/` along with `metadata.json`.

2.  **Fertility Analysis (`fertility_analysis.py`)**:
    *   Tokenizes dataset splits using `microsoft/phi-4`'s tokenizer.
    *   Calculates the mean subword tokens generated per word.
    *   Outputs results to `results/module_0_prestudy/results/prestudy_fertility.json`.

3.  **Baseline Evaluation (`base_eval.py`)**:
    *   Loads the un-tuned foundation model `microsoft/phi-4` on the CPU/GPU.
    *   Runs the evaluation engine over the FLORES-200 Kannada and Telugu devtest sets.
    *   Saves zero-shot benchmarks to `results/module_0_prestudy/results/baseline_phi4.json` and logs metrics in the main database.

## Output Artifacts
*   **Clean Splits**: `data/kannada/` and `data/telugu/`
*   **Fertility Stats**: `results/module_0_prestudy/results/prestudy_fertility.json`
*   **Baseline Logs**: `results/module_0_prestudy/results/baseline_phi4.json`
*   **Detailed Results Summary**: [prestudy_results.md](../readme/results/prestudy_results.md)
*   **Data Quality Audit Report (Markdown)**: [data_quality_report.md](../results/module_0_prestudy/results/data_quality_report.md)
*   **Data Quality Audit Report (JSON)**: [data_quality_report.json](../results/module_0_prestudy/results/data_quality_report.json)

## Executed Test Cases

After Module 0 is completed, the pytest validation gate [tests/test_prestudy.py](../tests/test_prestudy.py) executes the following test cases to ensure data and baseline evaluation integrity:

*   **`test_data_files_exist`**: Verifies that the dataset training, validation, and test splits (`train.jsonl`, `val.jsonl`, `test.jsonl`) exist for both Kannada and Telugu.
*   **`test_row_counts`**: Validates that dataset row counts meet minimum safety sizes (at least 16,000 train records, 900 validation records, and 900 test records per language).
*   **`test_schema_compliance`**: Checks that the split files conform to the expected schema (requiring keys `id`, `language`, `instruction`, `input`, `output`, `subset`, `script_purity`, `char_count_instruction`, `char_count_output`) and data type correctness.
*   **`test_script_purity_threshold`**: Ensures the average script purity of the generated dataset splits meets or exceeds the required project threshold of 0.80.
*   **`test_no_split_leakage`**: Prevents training-to-testing data leakage by asserting zero overlaps of SHA-256 instruction-output hashes between train and test splits.
*   **`test_fertility_results_exist`**: Confirms that the tokenizer fertility analysis report (`prestudy_fertility.json`) is generated with valid computed mean and std stats.
*   **`test_base_eval_done`**: Assures that zero-shot baseline evaluations for both languages are completed, verified, and correctly logged in `baseline_phi4.json`.
