"""Parse results_intervention/{mlp,cnn}/<cond>_{visible,invisible}.txt
and print a comparison table.

Usage:
    python hpc/summarize_intervention.py
"""
import os
import re

CONDITIONS = [
    "baseline_fixed",
    "baseline_random",
    "cond_A",
    "cond_B",
    "cond_C",
    "cond_D",
]
POLICIES = ["mlp", "cnn"]
MODES = ["visible", "invisible"]


def parse(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    reach = re.search(r"Reach-new-goal rate:\s+([0-9.]+)", text)
    drop = re.search(r"Distance drop in first 5 steps:\s+([+\-0-9.]+)", text)
    min_d = re.search(r"Closest approach \(min dist after\):\s+mean=([0-9.]+)", text)
    init_d = re.search(r"Initial dist to new goal:\s+mean=([0-9.]+)", text)
    if reach is None or drop is None:
        return None
    return {
        "reach": float(reach.group(1)),
        "drop": float(drop.group(1)),
        "min_dist": float(min_d.group(1)) if min_d else None,
        "init_dist": float(init_d.group(1)) if init_d else None,
    }


def print_table(policy, mode):
    rows = []
    any_found = False
    for cond in CONDITIONS:
        r = parse(f"results_intervention/{policy}/{cond}_{mode}.txt")
        if r is None:
            rows.append((cond, None))
        else:
            any_found = True
            rows.append((cond, r))

    if not any_found:
        print(f"== {policy.upper()} / {mode}-teleport ==  (no results found)\n")
        return

    print(f"== {policy.upper()} / {mode}-teleport ==")
    print(f"{'Condition':<18} {'Reach':>7} {'Drop_5':>8} {'Init_d':>7} {'Min_d':>7}")
    print("-" * 55)
    for cond, r in rows:
        if r is None:
            print(f"{cond:<18}  (missing)")
            continue
        init_str = f"{r['init_dist']:7.2f}" if r['init_dist'] is not None else "      -"
        min_str = f"{r['min_dist']:7.2f}" if r['min_dist'] is not None else "      -"
        print(f"{cond:<18} {r['reach']:>7.3f} {r['drop']:>+8.2f} {init_str} {min_str}")
    print()


if __name__ == "__main__":
    for policy in POLICIES:
        for mode in MODES:
            print_table(policy, mode)

    print("Reach   = fraction of intervention episodes where agent reached the new goal")
    print("Drop_5  = change in distance to new goal over the first 5 post-teleport steps")
    print("          (negative = moved toward; ~0 = no response; positive = moved away)")
    print("Init_d  = mean distance from agent to new goal immediately after teleport")
    print("Min_d   = closest the agent got to the new goal post-teleport")
    print()
    print("Causal-learning signature: high Reach + large negative Drop_5 in visible mode.")
    print("Spurious-learning signature: low Reach + near-zero Drop_5 (agent ignores teleport).")
