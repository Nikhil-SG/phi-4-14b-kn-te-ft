#!/bin/bash
# =============================================================================
# Monolingual FT: Run all 10 fine-tuning experiments sequentially.
#
# Usage (from project root, inside tmux/screen):
#     bash monolingual_ft/run_all_monolingual_ft.sh
#
# ─── HOW TO RUN (RECOMMENDED — tmux for disconnect protection) ───────────
#
#   Step 1: SSH into your HPC node
#       ssh nikhilsg@ishpc
#
#   Step 2: Start a tmux session (this survives browser/SSH disconnects)
#       tmux new -s module1
#
#   Step 3: Activate your conda environment
#       conda activate ft
#
#   Step 4: Navigate to project root
#       cd /data/nikhilsg/phi-4-14b-kn-te-ft
#
#   Step 5: Run this script
#       bash monolingual_ft/run_all_monolingual_ft.sh 2>&1 | tee results/module1_run.log
#
#   Step 6: Detach from tmux (training continues in background)
#       Press: Ctrl+B, then D
#
#   Step 7: Close your browser/SSH — the training keeps running!
#
#   Step 8: Re-attach later to check progress
#       tmux attach -t module1
#
#   Step 9: After everything finishes, kill the tmux session
#       tmux kill-session -t module1
#
# ─── ALTERNATIVE: Run a single technique+language ────────────────────────
#
#   # FFT Kannada only:
#   accelerate launch --num_processes 2 --mixed_precision bf16 \
#       monolingual_ft/train.py --technique fft --language kannada
#
#   # QLoRA Telugu only:
#   python monolingual_ft/train.py --technique qlora --language telugu
#
# ─── QUICK TMUX CHEATSHEET ───────────────────────────────────────────────
#
#   tmux new -s NAME        Create a new session named NAME
#   tmux attach -t NAME     Re-attach to session NAME
#   tmux ls                 List all sessions
#   Ctrl+B, D               Detach (training keeps running)
#   Ctrl+B, [               Scroll mode (q to exit)
#   tmux kill-session -t NAME   Kill session NAME
#
# =============================================================================
#
# Techniques: FFT, LoRA, QLoRA, DoRA, IA³
# Languages:  Kannada, Telugu
# Total runs: 5 × 2 = 10
#
# Skip logic: If train_metadata.json exists for a run, it is skipped.
#             A directory without train_metadata.json means a crashed run.
# =============================================================================
set -e

TECHNIQUES=("fft" "lora" "qlora" "dora" "ia3")
LANGUAGES=("kannada" "telugu")

# Techniques that use multi-GPU via accelerate (DDP or DeepSpeed)
DDP_TECHNIQUES=("fft" "lora" "dora" "ia3")

# Techniques that run on a single GPU
QLORA_TECHNIQUES=("qlora")

TOTAL_START=$SECONDS

echo ""
echo "=========================================="
echo "  Module 1: Monolingual Fine-Tuning"
echo "  Started: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"
echo "  Runs: ${#TECHNIQUES[@]} techniques × ${#LANGUAGES[@]} languages = $(( ${#TECHNIQUES[@]} * ${#LANGUAGES[@]} )) total"
echo "=========================================="

for technique in "${TECHNIQUES[@]}"; do
    for language in "${LANGUAGES[@]}"; do
        CKPT_DIR="results/finetuned_models/module_1_monolingual_ft/${technique}_${language}"
        METADATA_FILE="${CKPT_DIR}/train_metadata.json"

        # Skip if already completed
        if [ -f "$METADATA_FILE" ]; then
            echo "SKIP: ${technique}_${language} (already complete)"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "==== START: ${technique} on ${language} ===="
        echo "=========================================="
        echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%S UTC')"

        RUN_START=$SECONDS

        # Dispatch based on technique type
        if [[ " ${QLORA_TECHNIQUES[*]} " =~ " ${technique} " ]]; then
            python monolingual_ft/train.py \
                --technique "$technique" \
                --language "$language"
        else
            accelerate launch \
                --num_processes 2 \
                --mixed_precision bf16 \
                monolingual_ft/train.py \
                --technique "$technique" \
                --language "$language"
        fi

        RUN_ELAPSED=$(( SECONDS - RUN_START ))
        RUN_HOURS=$(echo "scale=2; $RUN_ELAPSED / 3600" | bc)
        echo ""
        echo "==== DONE: ${technique}_${language} (${RUN_ELAPSED}s ≈ ${RUN_HOURS}h) ===="
        echo ""
    done
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_ELAPSED / 3600" | bc)

echo ""
echo "=========================================="
echo "All Monolingual FT runs complete."
echo "Total elapsed: ${TOTAL_ELAPSED}s ≈ ${TOTAL_HOURS}h"
echo "MLflow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "Run tests:  pytest tests/test_monolingual_ft.py -v"
echo "=========================================="
