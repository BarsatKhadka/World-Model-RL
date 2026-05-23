#!/bin/bash
#SBATCH --job-name=ppo_spurious_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --exclude=node049
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# For each condition, evaluates the trained agent twice:
#   - S_train: in its own training env config
#   - S_test:  in the held-out env (goal=random, yellow=off)
#
# Picks the most recent checkpoint matching the condition's run name prefix.
# Submit AFTER train_array.slurm has finished:
#   sbatch --dependency=afterok:<TRAIN_JOBID> hpc/eval_array.slurm

set -uo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
source venv/bin/activate

# Same array layout as train_array.slurm: 0-5 mlp, 6-11 cnn.
if   [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    POLICY="mlp"
    COND_ID=$SLURM_ARRAY_TASK_ID
else
    POLICY="cnn"
    COND_ID=$((SLURM_ARRAY_TASK_ID - 6))
fi
mkdir -p "results/${POLICY}"

case "$COND_ID" in
  0) NAME="baseline_fixed";  GOAL=1;       YELLOW=off    ;;
  1) NAME="baseline_random"; GOAL=random;  YELLOW=off    ;;
  2) NAME="cond_A";          GOAL=1;       YELLOW=follow ;;
  3) NAME="cond_B";          GOAL=1;       YELLOW=random ;;
  4) NAME="cond_C";          GOAL=random;  YELLOW=follow ;;
  5) NAME="cond_D";          GOAL=random;  YELLOW=random ;;
  *) echo "Unknown COND_ID=$COND_ID"; exit 1 ;;
esac

# Find the most recent run directory for this condition + policy.
CKPT="$(ls -td runs/${POLICY}__g${GOAL}_y${YELLOW}__${NAME}__*/agent.pt 2>/dev/null | head -1)"
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found for $NAME (pattern: runs/${POLICY}__g${GOAL}_y${YELLOW}__${NAME}__*/agent.pt)"
  exit 1
fi
echo "[start] $(date)  task=$SLURM_ARRAY_TASK_ID  policy=$POLICY  name=$NAME  ckpt=$CKPT"

# Evaluate in own training env (S_train)
echo "----- S_train: own env (goal=$GOAL, yellow=$YELLOW) -----"
python -u eval.py \
    --agent-path "$CKPT" \
    --goal-room "$GOAL" \
    --yellow-room "$YELLOW" \
    --policy "$POLICY" \
    --num-episodes 100 \
    --seed 1000 \
    | tee "results/${POLICY}/${NAME}_strain.txt"

# Evaluate in held-out env (S_test)
echo "----- S_test: held-out (goal=random, yellow=off) -----"
python -u eval.py \
    --agent-path "$CKPT" \
    --goal-room random \
    --yellow-room off \
    --policy "$POLICY" \
    --num-episodes 100 \
    --seed 2000 \
    | tee "results/${POLICY}/${NAME}_stest.txt"

echo "[done] $(date)  policy=$POLICY  name=$NAME"
