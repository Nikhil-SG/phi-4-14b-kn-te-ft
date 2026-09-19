#!/bin/bash
# =============================================================================
# SLURM Batch Script: Module 2 — Sequential Fine-Tuning
# =============================================================================
#
# This script submits the sequential fine-tuning job to the SLURM scheduler
# so it runs on a dedicated compute node with GPU access.
#
# ─── USAGE ───────────────────────────────────────────────────────────────
#
# IMPORTANT: Run the sbatch command from the PROJECT ROOT directory:
#     cd /data/nikhilsg/phi-4-14b-kn-te-ft
#
# ── Standard Run ─────────────────────────────────────────────────────────
#   sbatch sequential_ft/slurm_sequential_ft.sh
#
# ── Dry Run (Simulation) ─────────────────────────────────────────────────
#   sbatch sequential_ft/slurm_sequential_ft.sh --dry-run
#
# ─── MONITORING ──────────────────────────────────────────────────────────
#
#   squeue -u $USER                          # Check job status
#   tail -f slurm_logs/module2_<JOBID>.out   # Watch live output
#   scancel <JOBID>                          # Cancel a running job
#   sacct -j <JOBID> --format=JobID,Elapsed,State,MaxRSS   # Job stats
#
# =============================================================================

# ── SLURM directives ────────────────────────────────────────────────────
#SBATCH --job-name=m2_seq_ft
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=slurm_logs/module2_%j.out
#SBATCH --error=slurm_logs/module2_%j.err

# =============================================================================
# Parse command-line arguments: --dry-run
# =============================================================================
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run|--dry_run)
            DRY_RUN=1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
cd /data/nikhilsg/phi-4-14b-kn-te-ft
source env/ft/bin/activate

# Create log directory (SLURM needs it to exist for --output/--error)
mkdir -p slurm_logs

# Disable torch.compile globally — it causes massive RAM overhead with DeepSpeed
export TORCHDYNAMO_DISABLE=1

# Clear system caches for maximum available RAM
if [ "$DRY_RUN" != "1" ]; then
    echo "[SETUP] Clearing system caches..."
    sync
    (echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null || true
fi

TOTAL_START=$SECONDS

echo ""
echo "============================================================"
echo "  Module 2: Sequential Fine-Tuning (SLURM Job: $SLURM_JOB_ID)"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Node    : $(hostname)"
if [ "$DRY_RUN" != "1" ]; then
    echo "  GPUs    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd', ' || echo 'N/A')"
    echo "  RAM     : $(free -h 2>/dev/null | awk '/^Mem:/ {print $2 " total, " $7 " available"}' || echo 'N/A')"
fi
echo "============================================================"

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
CKPT_DIR="results/finetuned_models/module_2_sequential_ft/sequential_model"
METADATA_FILE="${CKPT_DIR}/train_metadata.json"
RETENTION_REPORT="results/module_2_sequential_ft/results/retention_report.json"

if [ -f "$RETENTION_REPORT" ] && [ -f "$METADATA_FILE" ] && [ "$DRY_RUN" != "1" ]; then
    echo "[SKIP] Sequential FT already complete. Delete results/module_2_sequential_ft/ to re-run."
    exit 0
fi

# Decouple train and eval stage executions
if [ "$DRY_RUN" == "1" ]; then
    echo "[DRY-RUN] Launching python dry-run verification..."
    python sequential_ft/sequential_train.py --dry_run
    EXIT_CODE=$?
else
    # 1. Run zero-shot evaluation and continuation training on GPU 1
    export CUDA_VISIBLE_DEVICES=1
    echo "[INFO] Running training on GPU 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
    torchrun --nproc_per_node=1 sequential_ft/sequential_train.py --skip_eval
    EXIT_CODE=$?

    # 2. Run post-FT evaluation in a separate clean process on GPU 1
    if [ $EXIT_CODE -eq 0 ]; then
        # Load training time from metadata
        TRAIN_TIME_HRS=$(python -c "
import json, pathlib
meta_path = pathlib.Path('${METADATA_FILE}')
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print(meta.get('train_time_hrs', 0.0))
else:
    print(0.0)
")
        export CUDA_VISIBLE_DEVICES=1
        echo "[INFO] Running evaluations in separate process on GPU 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
        python sequential_ft/sequential_train.py --eval_only --train_time_hrs "${TRAIN_TIME_HRS}"
        EXIT_CODE=$?
    fi
fi

# =============================================================================
# Summary
# =============================================================================
TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_ELAPSED / 3600" | bc)

echo ""
echo "============================================================"
echo "  Module 2 Complete"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Elapsed  : ${TOTAL_ELAPSED}s (${TOTAL_HOURS}h)"
echo "  Exit Code: $EXIT_CODE"
echo "  Log file : slurm_logs/module2_${SLURM_JOB_ID}.out"
echo "============================================================"

exit $EXIT_CODE
