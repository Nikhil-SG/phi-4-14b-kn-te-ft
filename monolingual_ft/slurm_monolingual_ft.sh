#!/bin/bash
# =============================================================================
# SLURM Batch Script: Module 1 — Monolingual Fine-Tuning
# =============================================================================
#
# This script submits training jobs to the SLURM scheduler so they run on
# a dedicated compute node with guaranteed GPU access and walltime.
#
# ─── USAGE ───────────────────────────────────────────────────────────────
#
# IMPORTANT: Run ALL sbatch commands from the PROJECT ROOT directory:
#     cd /data/nikhilsg/phi-4-14b-kn-te-ft
#
# ── Combination 1: FFT only, both languages (RECOMMENDED FIRST RUN) ─────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques fft
#
# ── Combination 2: FFT only, Kannada only ────────────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques fft --languages kannada
#
# ── Combination 3: FFT only, Telugu only ─────────────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques fft --languages telugu
#
# ── Combination 3b: FFT on a specific GPU (default: GPU 1) ──────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques fft --languages kannada telugu
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques fft --languages kannada telugu --device 0
#
# ── Combination 4: LoRA + QLoRA, both languages ─────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques lora qlora
#
# ── Combination 5: LoRA only, both languages ────────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques lora
#
# ── Combination 6: QLoRA only, both languages ───────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques qlora
#
# ── Combination 7: DoRA only, both languages ────────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques dora
#
# ── Combination 8: IA3 only, both languages ─────────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques ia3
#
# ── Combination 9: PEFT methods only (LoRA + QLoRA + DoRA + IA3) ────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques lora qlora dora ia3
#
# ── Combination 10: ALL techniques, both languages (full run) ───────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh
#
# ── Combination 11: ALL techniques, Kannada only ────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --languages kannada
#
# ── Combination 12: ALL techniques, Telugu only ─────────────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --languages telugu
#
# ── Combination 13: Single technique + single language ──────────────────
#   sbatch monolingual_ft/slurm_monolingual_ft.sh --techniques dora --languages telugu
#
# ─── MONITORING ──────────────────────────────────────────────────────────
#
#   squeue -u $USER                          # Check job status
#   tail -f slurm_logs/module1_<JOBID>.out   # Watch live output
#   scancel <JOBID>                          # Cancel a running job
#   sacct -j <JOBID> --format=JobID,Elapsed,State,MaxRSS   # Job stats
#
# ─── PARTITION NOTE ──────────────────────────────────────────────────────
#
#   Run 'sinfo' to see available partitions on your cluster.
#   If the default partition does not have GPUs, uncomment and set:
#       #SBATCH --partition=gpu
#   to the correct GPU partition name.
#
# =============================================================================

# ── SLURM directives ────────────────────────────────────────────────────
#SBATCH --job-name=m1_mono_ft
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=slurm_logs/module1_%j.out
#SBATCH --error=slurm_logs/module1_%j.err
# Uncomment and edit the line below if your cluster requires a partition name:
# #SBATCH --partition=gpu

# =============================================================================
# Parse command-line arguments: --techniques, --languages, --device, --dry-run
# =============================================================================
TECHNIQUES=()
LANGUAGES=()
DEVICES=()
DRY_RUN=0
CURRENT_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --techniques)
            CURRENT_FLAG="techniques"
            ;;
        --languages)
            CURRENT_FLAG="languages"
            ;;
        --device|--devices)
            CURRENT_FLAG="devices"
            ;;
        --dry-run|--dry_run)
            DRY_RUN=1
            CURRENT_FLAG=""
            ;;
        *)
            if [ "$CURRENT_FLAG" == "techniques" ]; then
                TECHNIQUES+=("$arg")
            elif [ "$CURRENT_FLAG" == "languages" ]; then
                LANGUAGES+=("$arg")
            elif [ "$CURRENT_FLAG" == "devices" ]; then
                DEVICES+=("$arg")
            fi
            ;;
    esac
done

# Defaults: all techniques, both languages
if [ ${#TECHNIQUES[@]} -eq 0 ]; then
    TECHNIQUES=("fft" "lora" "qlora" "dora" "ia3")
fi
if [ ${#LANGUAGES[@]} -eq 0 ]; then
    LANGUAGES=("kannada" "telugu")
fi

# Normalize and split devices
NORMALIZED_DEVICES=()
for dev in "${DEVICES[@]}"; do
    for subdev in $dev; do
        NORMALIZED_DEVICES+=("$subdev")
    done
done

WANT_DEV_0=0
WANT_DEV_1=0

if [ ${#NORMALIZED_DEVICES[@]} -eq 0 ]; then
    # Default: use device 1 only
    WANT_DEV_1=1
else
    for dev in "${NORMALIZED_DEVICES[@]}"; do
        if [ "$dev" == "0" ]; then
            WANT_DEV_0=1
        elif [ "$dev" == "1" ]; then
            WANT_DEV_1=1
        fi
    done
fi

# Print status of desired devices
echo "[SETUP] Desired devices: GPU 0=${WANT_DEV_0}, GPU 1=${WANT_DEV_1}"

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

# MLflow database policy: APPEND-ONLY — never delete mlflow.db.
# Once a metric is written to mlflow.db it must not be removed or altered.
# Experiments accumulate across job runs under "Module1_MonolingualFT".
# If the DB is corrupted, fft_trainer._setup_mlflow() will back it up
# (mlflow.db.bak.<timestamp>) and start a fresh one automatically.
if [ -f "mlflow.db" ]; then
    echo "[SETUP] mlflow.db exists — preserving existing experiment history (append-only policy)."
else
    echo "[SETUP] No mlflow.db found — a new one will be created on first run."
fi

TOTAL_START=$SECONDS
TOTAL_RUNS=$(( ${#TECHNIQUES[@]} * ${#LANGUAGES[@]} ))
COMPLETED=0
FAILED=0

echo ""
echo "============================================================"
echo "  Module 1: Monolingual Fine-Tuning (SLURM Job: $SLURM_JOB_ID)"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Node    : $(hostname)"
if [ "$DRY_RUN" != "1" ]; then
    echo "  GPUs    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd', ' || echo 'N/A')"
    echo "  RAM     : $(free -h 2>/dev/null | awk '/^Mem:/ {print $2 " total, " $7 " available"}' || echo 'N/A')"
fi
echo "  Runs    : ${#TECHNIQUES[@]} techniques × ${#LANGUAGES[@]} languages = $TOTAL_RUNS total"
echo "  Techniques: ${TECHNIQUES[*]}"
echo "  Languages : ${LANGUAGES[*]}"
echo "  Walltime  : $(squeue -j $SLURM_JOB_ID -o '%l' --noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

if [ "$DRY_RUN" != "1" ]; then
    # Print GPU status
    nvidia-smi 2>/dev/null || true
    echo ""
    echo "System RAM:"
    free -h 2>/dev/null || true
    echo ""
fi

# =============================================================================
# Run experiments with automatic retry
# =============================================================================
MAX_RETRIES=3

for technique in "${TECHNIQUES[@]}"; do
    for language in "${LANGUAGES[@]}"; do

        CKPT_DIR="results/finetuned_models/module_1_monolingual_ft/${technique}_${language}"
        METADATA_FILE="${CKPT_DIR}/train_metadata.json"

        # ── Skip if already completed (unless in dry-run mode) ───────
        if [ -f "$METADATA_FILE" ] && [ "$DRY_RUN" != "1" ]; then
            echo ""
            echo "[SKIP] ${technique}_${language} — train_metadata.json exists (already complete)"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi

        RUN_SUCCESS=0
        RUN_START=$SECONDS

        for attempt in $(seq 1 $MAX_RETRIES); do

            echo ""
            echo "============================================================"
            echo "  START: ${technique} on ${language} (attempt ${attempt}/${MAX_RETRIES})"
            echo "  Time : $(date '+%Y-%m-%d %H:%M:%S %Z')"
            echo "============================================================"

            # ── Pre-run diagnostics ──────────────────────────────────
            if [ "$DRY_RUN" != "1" ]; then
                echo ""
                echo "[DIAG] Pre-run memory status:"
                free -h 2>/dev/null | head -2 || echo "  (N/A)"
                echo "[DIAG] Existing checkpoints:"
                ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null || echo "  (none)"
                echo ""
            fi

            ATTEMPT_START=$SECONDS

            # ── Determine run parameters for this technique ──────────
            if [ "$technique" == "qlora" ]; then
                if [ $WANT_DEV_0 -eq 1 ] && [ $WANT_DEV_1 -eq 0 ]; then
                    CUR_VISIBLE="0"
                    EVAL_VISIBLE="1"
                else
                    CUR_VISIBLE="1"
                    EVAL_VISIBLE="0"
                fi
                CUR_PROCS=1
                LAUNCH_CMD="CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} python monolingual_ft/train.py --technique ${technique} --language ${language} --skip_eval"
            elif [ "$technique" == "fft" ]; then
                if [ $WANT_DEV_0 -eq 1 ] && [ $WANT_DEV_1 -eq 0 ]; then
                    CUR_VISIBLE="0"
                    EVAL_VISIBLE="0"
                else
                    CUR_VISIBLE="1"
                    EVAL_VISIBLE="0,1"
                fi
                CUR_PROCS=1
                LAUNCH_CMD="CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} torchrun --nproc_per_node=${CUR_PROCS} monolingual_ft/train.py --technique ${technique} --language ${language} --skip_eval"
            else
                # lora, dora, ia3
                if [ $WANT_DEV_0 -eq 1 ] && [ $WANT_DEV_1 -eq 1 ]; then
                    CUR_VISIBLE="0,1"
                    CUR_PROCS=2
                    EVAL_VISIBLE="0"
                elif [ $WANT_DEV_0 -eq 1 ]; then
                    CUR_VISIBLE="0"
                    CUR_PROCS=1
                    EVAL_VISIBLE="1"
                else
                    CUR_VISIBLE="1"
                    CUR_PROCS=1
                    EVAL_VISIBLE="0"
                fi
                LAUNCH_CMD="CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} accelerate launch --num_processes ${CUR_PROCS} --mixed_precision bf16 monolingual_ft/train.py --technique ${technique} --language ${language} --skip_eval"
            fi

            echo "[INFO] Launching command: $LAUNCH_CMD"

            if [ "$DRY_RUN" == "1" ]; then
                echo "[DRY-RUN] Simulating successful execution."
                EXIT_CODE=0
            else
                # Actual execution
                if [ "$technique" == "qlora" ]; then
                    CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} python monolingual_ft/train.py \
                        --technique "$technique" \
                        --language "$language" \
                        --skip_eval
                elif [ "$technique" == "fft" ]; then
                    CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} torchrun --nproc_per_node=${CUR_PROCS} \
                        monolingual_ft/train.py \
                            --technique "$technique" \
                            --language "$language" \
                            --skip_eval
                else
                    CUDA_VISIBLE_DEVICES=${CUR_VISIBLE} accelerate launch \
                        --num_processes ${CUR_PROCS} \
                        --mixed_precision bf16 \
                        monolingual_ft/train.py \
                        --technique "$technique" \
                        --language "$language" \
                        --skip_eval
                fi
                EXIT_CODE=$?
            fi

            # If training succeeded, run evaluation on the other GPU
            if [ $EXIT_CODE -eq 0 ]; then
                TRAIN_ELAPSED=$(( SECONDS - ATTEMPT_START ))
                TRAIN_TIME_HRS=$(echo "scale=4; $TRAIN_ELAPSED / 3600" | bc)

                EVAL_CMD="CUDA_VISIBLE_DEVICES=${EVAL_VISIBLE} python monolingual_ft/train.py --technique ${technique} --language ${language} --eval_only --train_time_hrs ${TRAIN_TIME_HRS}"
                
                echo "[INFO] Evaluation command: $EVAL_CMD"

                if [ "$DRY_RUN" == "1" ]; then
                    echo "[DRY-RUN] Simulating successful evaluation."
                    EXIT_CODE=0
                else
                    CUDA_VISIBLE_DEVICES=${EVAL_VISIBLE} python monolingual_ft/train.py \
                        --technique "$technique" \
                        --language "$language" \
                        --eval_only \
                        --train_time_hrs "$TRAIN_TIME_HRS"
                    EXIT_CODE=$?
                fi
            fi

            ATTEMPT_ELAPSED=$(( SECONDS - ATTEMPT_START ))

            # ── Post-run diagnostics ─────────────────────────────────
            if [ "$DRY_RUN" != "1" ]; then
                echo ""
                echo "[DIAG] Post-run memory status:"
                free -h 2>/dev/null | head -2 || echo "  (N/A)"
                nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  (N/A)"
                echo ""
            fi

            # ── Diagnose the exit code ───────────────────────────────
            if [ $EXIT_CODE -eq 0 ]; then
                # Verify completion
                if [ "$DRY_RUN" == "1" ] || [ -f "$METADATA_FILE" ]; then
                    RUN_ELAPSED=$(( SECONDS - RUN_START ))
                    RUN_HOURS=$(echo "scale=2; $RUN_ELAPSED / 3600" | bc)
                    echo "[DONE] ${technique}_${language} — completed in ${RUN_ELAPSED}s (${RUN_HOURS}h) ✓"
                    RUN_SUCCESS=1
                    break
                else
                    echo "[WARN] ${technique}_${language} — exit code 0 but train_metadata.json missing!"
                fi
            else
                echo "[FAIL] ${technique}_${language} — exit code $EXIT_CODE after ${ATTEMPT_ELAPSED}s"

                # Decode the signal from exit code
                if [ $EXIT_CODE -gt 128 ]; then
                    SIG_NUM=$((EXIT_CODE - 128))
                    case $SIG_NUM in
                        1)  echo "[DIAG] Signal: SIGHUP (1) — SSH disconnect or session timeout" ;;
                        2)  echo "[DIAG] Signal: SIGINT (2) — Ctrl+C or interactive interrupt" ;;
                        9)  echo "[DIAG] Signal: SIGKILL (9) — OOM killer or forced termination (cannot be caught)" ;;
                        15) echo "[DIAG] Signal: SIGTERM (15) — SLURM walltime or graceful shutdown" ;;
                        *)  echo "[DIAG] Signal: $SIG_NUM — Unknown signal" ;;
                    esac
                fi

                # Check for crash forensics files
                if ls ${CKPT_DIR}/monitor/crash_*.json 2>/dev/null; then
                    echo "[DIAG] Crash report found:"
                    cat $(ls -t ${CKPT_DIR}/monitor/crash_*.json | head -1)
                fi

                # Check if checkpoints exist for resume
                LATEST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
                if [ -n "$LATEST_CKPT" ]; then
                    echo "[DIAG] Latest checkpoint: $LATEST_CKPT — will auto-resume on retry"
                else
                    echo "[DIAG] No checkpoints found — next attempt starts from scratch"
                fi
            fi

            if [ $attempt -lt $MAX_RETRIES ] && [ "$DRY_RUN" != "1" ]; then
                echo ""
                echo "[RETRY] Waiting 60s before retry ${attempt}/$((MAX_RETRIES-1)) to let memory settle..."
                echo "[RETRY] Clearing Python/GPU caches..."

                # Force cleanup between retries
                sync
                echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
                sleep 60

                echo "[RETRY] Memory after cleanup:"
                free -h | head -2
                echo ""
            fi
        done

        if [ $RUN_SUCCESS -eq 1 ]; then
            COMPLETED=$((COMPLETED + 1))
        else
            echo ""
            echo "[GIVE UP] ${technique}_${language} — failed after $MAX_RETRIES attempts"
            FAILED=$((FAILED + 1))
        fi

    done
done

# =============================================================================
# Summary
# =============================================================================
TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_ELAPSED / 3600" | bc)

echo ""
echo "============================================================"
echo "  Module 1 Complete"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Elapsed  : ${TOTAL_ELAPSED}s (${TOTAL_HOURS}h)"
echo "  Results  : $COMPLETED completed, $FAILED failed, out of $TOTAL_RUNS total"
echo "  Log file : slurm_logs/module1_${SLURM_JOB_ID}.out"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "[WARNING] $FAILED run(s) failed. Check the log above for details."
    echo "  Re-submit the same command — completed runs will be auto-skipped."
    exit 1
fi
