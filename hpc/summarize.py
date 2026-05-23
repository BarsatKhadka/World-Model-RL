"""Parse results/{mlp,cnn}/*.txt produced by eval_array.slurm and print the Method 1 table.

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
POLICIES = ["mlp", "cnn"]


def parse(path):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"Success rate:\s+([0-9.]+)", text)
    r = re.search(r"Mean return:\s+([0-9.\-]+)", text)
    if m is None or r is None:
        return None  # file exists but is empty / malformed (e.g., job crashed)
    return {"success": float(m.group(1)), "mean_return": float(r.group(1))}


def find(policy, cond, kind):
    # Prefer subdir layout (results/<policy>/<cond>_<kind>.txt).
    p1 = f"results/{policy}/{cond}_{kind}.txt"
    if os.path.exists(p1):
        return parse(p1)
    # Fallback: flat layout (results/<cond>_<kind>.txt), counted as the "default" policy.
    if policy == POLICIES[0]:
        p2 = f"results/{cond}_{kind}.txt"
        if os.path.exists(p2):
            return parse(p2)
    return None


def print_policy_table(policy):
    any_found = False
    rows = []
    for cond in CONDITIONS:
        s_train = find(policy, cond, "strain")
        s_test = find(policy, cond, "stest")
        if s_train is None or s_test is None:
            rows.append((cond, None))
            continue
        any_found = True
        reliance = s_train["success"] - s_test["success"]
        rows.append((cond, {
            "s_train": s_train["success"],
            "s_test": s_test["success"],
            "reliance": reliance,
            "r_train": s_train["mean_return"],
            "r_test": s_test["mean_return"],
        }))

    if not any_found:
        print(f"== {policy.upper()} ==  (no results found)\n")
        return

    print(f"== {policy.upper()} ==")
    print(f"{'Condition':<18} {'S_train':>9} {'S_test':>9} {'Reliance':>10} {'R_train':>9} {'R_test':>9}")
    print("-" * 70)
    for cond, row in rows:
        if row is None:
            print(f"{cond:<18}  (missing)")
            continue
        print(f"{cond:<18} "
              f"{row['s_train']:>9.3f} "
              f"{row['s_test']:>9.3f} "
              f"{row['reliance']:>+10.3f} "
              f"{row['r_train']:>9.4f} "
              f"{row['r_test']:>9.4f}")
    print()


if __name__ == "__main__":
    for policy in POLICIES:
        print_policy_table(policy)

    print("Reliance = S_train - S_test.  Larger = agent relied more on shortcuts.")
    print("Expected ordering: A > B,C > D ~ baselines.")
    print("Compare MLP vs CNN: did the architecture change which shortcuts get learned?")
