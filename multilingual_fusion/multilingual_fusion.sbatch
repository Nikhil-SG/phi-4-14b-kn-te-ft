#!/bin/bash
#SBATCH --job-name=module3_adapter_merge
#SBATCH --output=slurm_logs/module3_%j.out
#SBATCH --error=slurm_logs/module3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu     # Adjust this to match your cluster's GPU partition name

# Exit on error
set -e

echo "Starting Module 3 Job: $(date)"

# Activate virtual environment (SLURM typically runs on Linux)
source env/ft/bin/activate

# Force execution on GPU device 1 only (does not touch device 0)
export CUDA_VISIBLE_DEVICES=1

# Execute the unified pipeline orchestrator
# This automatically runs adapter_merge.py, runs the pytest gate, 
# and updates pipeline.json + test_results.json upon success.
python main.py --module multilingual_fusion --fusion-strategies merge

echo "Module 3 Job Completed: $(date)"
