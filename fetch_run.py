"""
Fetch a W&B run's metrics as a DataFrame and print a text summary.

Usage:
    python fetch_run.py                      # latest run in default project
    python fetch_run.py <run_id>             # specific run by id
    python fetch_run.py --project foo        # different project
"""
import argparse
import sys

import pandas as pd
import wandb

DEFAULT_ENTITY = None  # None = your default entity (your username)
DEFAULT_PROJECT = "ppo-minigrid"


def fetch(entity, project, run_id=None):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    if run_id:
        run = api.run(f"{path}/{run_id}")
    else:
        runs = api.runs(path, order="-created_at")
        run = next(iter(runs), None)
        if run is None:
            sys.exit(f"No runs found in {path}")

    print(f"Run:     {run.name}")
    print(f"ID:      {run.id}")
    print(f"State:   {run.state}")
    print(f"URL:     {run.url}")
    print(f"Config:  env_id={run.config.get('env_id')}, "
          f"total_timesteps={run.config.get('total_timesteps')}, "
          f"seed={run.config.get('seed')}")
    print()

    history = run.history(samples=10000, pandas=True)
    if history.empty:
        sys.exit("No logged metrics yet.")

    print(f"Logged {len(history)} rows, columns: {list(history.columns)}\n")

    summarize(history, "charts/episodic_return", "Episodic return (main learning signal)")
    summarize(history, "charts/episodic_length", "Episodic length")
    summarize(history, "losses/policy_loss", "Policy loss")
    summarize(history, "losses/value_loss", "Value loss")
    summarize(history, "losses/entropy", "Entropy (exploration)")
    summarize(history, "losses/approx_kl", "Approx KL (policy change per update)")
    summarize(history, "charts/SPS", "Steps per second")

    return history


def summarize(df, col, label):
    if col not in df.columns:
        return
    s = df[col].dropna()
    if s.empty:
        return
    first = s.iloc[0]
    last = s.iloc[-1]
    early = s.head(max(1, len(s) // 10)).mean()
    late = s.tail(max(1, len(s) // 10)).mean()
    print(f"{label}  [{col}]")
    print(f"  points={len(s)}  first={first:.4f}  last={last:.4f}  "
          f"min={s.min():.4f}  max={s.max():.4f}")
    print(f"  early-10%-mean={early:.4f}  ->  late-10%-mean={late:.4f}  "
          f"(delta={late - early:+.4f})")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--csv", default=None, help="optional path to dump full history CSV")
    args = p.parse_args()

    df = fetch(args.entity, args.project, args.run_id)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
