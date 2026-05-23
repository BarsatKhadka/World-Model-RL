# HPC job scripts — spurious-signal experiment (Method 1)

Three files run the full Method 1 pipeline on the cluster.

## One-time HPC setup

```bash
# On the login node, from project root
module load python/2025.12-2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wandb login    # paste API key
```

## Run the experiment

```bash
# 1. Train all 6 conditions in parallel (each ~10 min wall on GPU)
TRAIN_JOBID=$(sbatch --parsable hpc/train_array.slurm)

# 2. Eval each agent in own env + held-out env (auto-runs after training)
sbatch --dependency=afterok:$TRAIN_JOBID hpc/eval_array.slurm

# 3. After eval finishes, print the Method 1 table
python hpc/summarize.py
```

## What each script does

- **train_array.slurm** — SLURM array of 6 tasks. Task i trains condition i:
  - 0 baseline_fixed   (goal=1,      yellow=off)
  - 1 baseline_random  (goal=random, yellow=off)
  - 2 cond_A           (goal=1,      yellow=follow)
  - 3 cond_B           (goal=1,      yellow=random)
  - 4 cond_C           (goal=random, yellow=follow)
  - 5 cond_D           (goal=random, yellow=random)
  Saves checkpoint to `runs/g{GOAL}_y{YELLOW}__{name}__{seed}__{ts}/agent.pt`
  and logs to W&B project `ppo-spurious`.

- **eval_array.slurm** — same array. For each condition, evals the trained agent
  in its own env (S_train) and the held-out env (S_test), 100 episodes each.
  Writes per-eval text files to `results/<cond>_strain.txt` and `_stest.txt`.

- **summarize.py** — parses those text files and prints the final table:

  ```
  Condition          S_train    S_test   Reliance   R_train    R_test
  ----------------------------------------------------------------------
  baseline_fixed       0.950     0.150     +0.800    0.620     0.075
  baseline_random      0.700     0.700     +0.000    0.350     0.350
  cond_A               0.980     0.120     +0.860    0.730     0.060
  ...
  ```

  Reliance = S_train − S_test. Big = relied on shortcuts.

## Re-running a single condition

```bash
sbatch --array=3 hpc/train_array.slurm   # just cond_B
sbatch --array=3 hpc/eval_array.slurm
```

## Watching progress

```bash
squeue -u $USER                          # which jobs are queued/running
tail -f logs/ppo_spurious_train_*_3.out  # live output for cond_B
```
