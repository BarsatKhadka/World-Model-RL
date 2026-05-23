#!/bin/bash
#SBATCH --job-name=ppo_spurious_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --exclude=node049
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Trains all 6 conditions (2 baselines + A/B/C/D) in parallel as array tasks.
# Each task produces a checkpoint at runs/<cond_tag>__<exp_name>__<seed>__<ts>/agent.pt
#
# Submit:    sbatch hpc/train_array.slurm
# Re-run one:  sbatch --array=3 hpc/train_array.slurm    # just condition B

set -euo pipefail   # -e: abort job on first error
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
source venv/bin/activate

TIMESTEPS="${TIMESTEPS:-2000000}"   # 2M steps
SEED="${SEED:-1}"                    # override with: SEED=2 sbatch --array=9 hpc/train_array.slurm

# Array maps to (policy, condition):
#   tasks 0-5   -> mlp, conditions 0-5
#   tasks 6-11  -> cnn, conditions 0-5
# To run a single policy only:   sbatch --array=0-5  hpc/train_array.slurm
#                                sbatch --array=6-11 hpc/train_array.slurm
if   [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    POLICY="mlp"
    COND_ID=$SLURM_ARRAY_TASK_ID
else
    POLICY="cnn"
    COND_ID=$((SLURM_ARRAY_TASK_ID - 6))
fi

case "$COND_ID" in
  0) NAME="baseline_fixed";  GOAL=1;       YELLOW=off    ;;
  1) NAME="baseline_random"; GOAL=random;  YELLOW=off    ;;
  2) NAME="cond_A";          GOAL=1;       YELLOW=follow ;;
  3) NAME="cond_B";          GOAL=1;       YELLOW=random ;;
  4) NAME="cond_C";          GOAL=random;  YELLOW=follow ;;
  5) NAME="cond_D";          GOAL=random;  YELLOW=random ;;
  *) echo "Unknown COND_ID=$COND_ID"; exit 1 ;;
esac

echo "[start] $(date)  host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  policy=$POLICY  name=$NAME"
nvidia-smi --query-gpu=name --format=csv

# Sanity check: wandb credentials must be present in ~/.netrc.
# Run `wandb login` on the login node once before submitting.
if [ ! -f "$HOME/.netrc" ] || ! grep -q "api.wandb.ai" "$HOME/.netrc"; then
    echo "ERROR: no wandb credentials in $HOME/.netrc. Run 'wandb login' on the login node."
    exit 1
fi

python -u ppo.py \
    --exp-name "$NAME" \
    --goal-room "$GOAL" \
    --yellow-room "$YELLOW" \
    --policy "$POLICY" \
    --total-timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --track \
    --wandb-project-name ppo-spurious

echo "[done] $(date)  policy=$POLICY  name=$NAME"
