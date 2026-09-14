#!/bin/bash
# =============================================================================
# Multilingual Fusion: Joint & Merged Training
#
# Usage (from project root):
#     bash multilingual_fusion/run_multilingual_fusion.sh
#
# Runs:
#   A. Joint bilingual — equal 50/50 sampling
#   B. Joint bilingual — proportional sampling
#   C. Adapter merge — TIES-Merge + linear average
# =============================================================================
set -e

echo "=== MODULE 3: Joint & Merged Training ==="

# Read winner technique from Monolingual FT
WINNER_TECHNIQUE=$(python -c "
import yaml, pathlib
w = yaml.safe_load(pathlib.Path('results/monolingual_ft_winner.yaml').read_text())
print(w['technique'])
")
echo "Winner technique: $WINNER_TECHNIQUE"

# Helper function: run joint training
run_joint() {
    local strategy=$1
    local output_done="results/multilingual_fusion/joint_${strategy}/train_metadata.json"

    if [ -f "$output_done" ]; then
        echo "SKIP: joint_${strategy} already done"
        return
    fi

    echo ""
    echo "--- Joint ${strategy} ---"

    if [ "$WINNER_TECHNIQUE" = "qlora" ]; then
        python multilingual_fusion/joint_train.py --sampling_strategy "${strategy}"
    else
        accelerate launch \
            --num_processes 2 \
            --mixed_precision bf16 \
            multilingual_fusion/joint_train.py --sampling_strategy "${strategy}"
    fi

    echo "--- Joint ${strategy} done ---"
}

# Run joint experiments
run_joint "equal"
run_joint "proportional"

# Run adapter merge
if [ -f "results/multilingual_fusion/merge_summary.json" ]; then
    echo ""
    echo "SKIP: adapter merge already done"
else
    echo ""
    echo "--- Adapter Merge (TIES + Linear) ---"
    python multilingual_fusion/adapter_merge.py
    echo "--- Adapter merge done ---"
fi

echo ""
echo "=== Multilingual Fusion complete ==="
echo "Run: pytest tests/test_multilingual_fusion.py -v"
echo "Then run: python visualize/pareto_frontier.py"
echo "          python visualize/interference_matrix.py"
echo "          python visualize/loss_curves.py"
echo "          python visualize/fertility_plots.py"
