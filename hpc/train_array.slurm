#!/bin/bash
#SBATCH --job-name=ppo_spurious_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Trains all 6 conditions (2 baselines + A/B/C/D) in parallel as array tasks.
# Each task produces a checkpoint at runs/<cond_tag>__<exp_name>__<seed>__<ts>/agent.pt
#
# Submit:    sbatch hpc/train_array.slurm
# Re-run one:  sbatch --array=3 hpc/train_array.slurm    # just condition B

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
source venv/bin/activate

# Array task -> condition mapping
case "$SLURM_ARRAY_TASK_ID" in
  0) NAME="baseline_fixed";  GOAL=1;       YELLOW=off    ;;
  1) NAME="baseline_random"; GOAL=random;  YELLOW=off    ;;
  2) NAME="cond_A";          GOAL=1;       YELLOW=follow ;;
  3) NAME="cond_B";          GOAL=1;       YELLOW=random ;;
  4) NAME="cond_C";          GOAL=random;  YELLOW=follow ;;
  5) NAME="cond_D";          GOAL=random;  YELLOW=random ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "[start] $(date)  host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  name=$NAME"
nvidia-smi --query-gpu=name --format=csv

python -u ppo.py \
    --exp-name "$NAME" \
    --goal-room "$GOAL" \
    --yellow-room "$YELLOW" \
    --total-timesteps 500000 \
    --seed 1 \
    --track \
    --wandb-project-name ppo-spurious

echo "[done] $(date)  name=$NAME"
