#!/bin/bash
# =============================================================================
# Sequential FT: Bidirectional Cross-Lingual + Sequential Training
#
# Redesigned Module 2:
#   Stage 1: Bidirectional zero-shot evals (KN→TE and TE→KN)
#   Stage 2: Telugu (winner) → continue training on Kannada (1 GPU)
#   Stage 3: Evaluate on Kannada, Telugu, and English benchmarks
#   Stage 4: Compute retention report
#
# Usage (from project root):
#     bash sequential_ft/run_sequential_ft.sh
#
# Always runs on a SINGLE GPU with DeepSpeed ZeRO-2 + CPU offloading.
# =============================================================================
set -e

echo "=== MODULE 2: Sequential Training (Telugu → Kannada) ==="

# Read winner technique from Monolingual FT
WINNER_TECHNIQUE=$(python -c "
import yaml, pathlib
w = yaml.safe_load(pathlib.Path('results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml').read_text())
print(w['technique'])
")
WINNER_LANGUAGE=$(python -c "
import yaml, pathlib
w = yaml.safe_load(pathlib.Path('results/module_1_monolingual_ft/results/monolingual_ft_winner.yaml').read_text())
print(w.get('language', 'telugu'))
")
echo "Winner technique: $WINNER_TECHNIQUE"
echo "Winner language: $WINNER_LANGUAGE"

# Check if Sequential FT is already complete
if [ -f "results/module_2_sequential_ft/results/retention_report.json" ]; then
    echo "Sequential FT already complete. Delete results/module_2_sequential_ft/ to re-run."
    exit 0
fi

# Always single GPU — restrict to device 1
export CUDA_VISIBLE_DEVICES=1
echo "Running on single GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Launching $WINNER_TECHNIQUE sequential training (single GPU, DeepSpeed ZeRO-2) ..."
torchrun --nproc_per_node=1 sequential_ft/sequential_train.py

echo ""
echo "=========================================="
echo "Sequential FT complete."
echo "Retention report: results/module_2_sequential_ft/results/retention_report.json"
echo "Run: pytest tests/test_sequential_ft.py -v"
echo "=========================================="
