"""Parse results/*.txt produced by eval_array.slurm and print the Method 1 table.

Usage:
    python hpc/summarize.py
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


def parse(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"Success rate:\s+([0-9.]+)", text)
    r = re.search(r"Mean return:\s+([0-9.\-]+)", text)
    return {
        "success": float(m.group(1)) if m else None,
        "mean_return": float(r.group(1)) if r else None,
    }


if __name__ == "__main__":
    print(f"{'Condition':<18} {'S_train':>9} {'S_test':>9} {'Reliance':>10} {'R_train':>9} {'R_test':>9}")
    print("-" * 70)
    for cond in CONDITIONS:
        s_train = parse(f"results/{cond}_strain.txt")
        s_test = parse(f"results/{cond}_stest.txt")
        if s_train is None or s_test is None:
            print(f"{cond:<18}  (missing results)")
            continue
        reliance = s_train["success"] - s_test["success"]
        print(f"{cond:<18} "
              f"{s_train['success']:>9.3f} "
              f"{s_test['success']:>9.3f} "
              f"{reliance:>+10.3f} "
              f"{s_train['mean_return']:>9.4f} "
              f"{s_test['mean_return']:>9.4f}")
    print()
    print("Reliance = S_train - S_test.  Larger = agent relied more on shortcuts.")
    print("Expected ordering: A > B,C > D ~ baselines.")
