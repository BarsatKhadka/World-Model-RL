#!/bin/bash
#SBATCH --job-name=ppo_spurious_interv
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --exclude=node049
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# For each (policy, condition), runs the intervention test in BOTH modes
# (visible and invisible teleport), writing results to:
#   results_intervention/<policy>/<cond>_visible.txt
#   results_intervention/<policy>/<cond>_invisible.txt
#
# Same 12-task layout as train_array.slurm / eval_array.slurm:
#   tasks 0-5   -> mlp, conditions baseline_fixed..cond_D
#   tasks 6-11  -> cnn, conditions baseline_fixed..cond_D
#
# Submit:           sbatch hpc/intervention_array.slurm
# Re-run one:       sbatch --array=8 hpc/intervention_array.slurm   # CNN cond_A
# Single policy:    sbatch --array=6-11 hpc/intervention_array.slurm  # CNN only

set -euo pipefail
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_intervention
source venv/bin/activate

NUM_EPISODES="${NUM_EPISODES:-200}"
INTERVENTION_STEP="${INTERVENTION_STEP:-30}"

# Array task -> (policy, condition)
if   [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    POLICY="mlp"
    COND_ID=$SLURM_ARRAY_TASK_ID
else
    POLICY="cnn"
    COND_ID=$((SLURM_ARRAY_TASK_ID - 6))
fi
mkdir -p "results_intervention/${POLICY}"

case "$COND_ID" in
  0) NAME="baseline_fixed";  GOAL=1;       YELLOW=off    ;;
  1) NAME="baseline_random"; GOAL=random;  YELLOW=off    ;;
  2) NAME="cond_A";          GOAL=1;       YELLOW=follow ;;
  3) NAME="cond_B";          GOAL=1;       YELLOW=random ;;
  4) NAME="cond_C";          GOAL=random;  YELLOW=follow ;;
  5) NAME="cond_D";          GOAL=random;  YELLOW=random ;;
  *) echo "Unknown COND_ID=$COND_ID"; exit 1 ;;
esac

CKPT="$(ls -td runs/${POLICY}__g${GOAL}_y${YELLOW}__${NAME}__*/agent.pt 2>/dev/null | head -1)"
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found for ${POLICY} ${NAME} (pattern: runs/${POLICY}__g${GOAL}_y${YELLOW}__${NAME}__*/agent.pt)"
    exit 1
fi
echo "[start] $(date)  task=$SLURM_ARRAY_TASK_ID  policy=$POLICY  name=$NAME  ckpt=$CKPT"

# All intervention tests use the HELD-OUT env (goal_room=random, yellow_room=off)
# so the agent sees no spurious signal to fall back on -- pure goal-tracking test.

echo "----- visible teleport (goal moves into agent's 7x7 view) -----"
python -u intervention.py \
    --agent-path "$CKPT" \
    --policy "$POLICY" \
    --teleport-mode visible \
    --goal-room random \
    --yellow-room off \
    --intervention-step "$INTERVENTION_STEP" \
    --num-episodes "$NUM_EPISODES" \
    --seed 3000 \
    | tee "results_intervention/${POLICY}/${NAME}_visible.txt"

echo "----- invisible teleport (goal moves far outside agent's view) -----"
python -u intervention.py \
    --agent-path "$CKPT" \
    --policy "$POLICY" \
    --teleport-mode invisible \
    --goal-room random \
    --yellow-room off \
    --intervention-step "$INTERVENTION_STEP" \
    --num-episodes "$NUM_EPISODES" \
    --seed 4000 \
    | tee "results_intervention/${POLICY}/${NAME}_invisible.txt"

echo "[done] $(date)  policy=$POLICY  name=$NAME"
