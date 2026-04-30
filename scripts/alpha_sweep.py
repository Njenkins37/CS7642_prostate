"""
Focal alpha grid sweep. This is continuation of random search that yielded marginal gains only.
Later, can be combined into one as a single grid search.

Only alpha modiefied.

Out to data/alpha_sweep_results.csv.

Run:
    python scripts/alpha_sweep.py
"""

import csv
import math
import subprocess
import sys
import time
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "models" / "weights"

# True passes --no_attention to train.py and outputs
NO_ATTENTION = True

# Separate outputs to prevent overwrite
RESULTS_CSV = REPO_ROOT /"data" /(
    "alpha_sweep_results_noatt.csv" if NO_ATTENTION else "alpha_sweep_results.csv")

# Fixed HPs (selected from random search)
LR = 5.19e-04
WD = 1.80e-03
DW = 0.46
FW = 0.54
BATCH_SIZE = 64
EPOCHS_PER_TRIAL = 25

# Grid over Focal alpha: 6
ALPHAS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]

# Unique output naming
def run_tag(alpha):
    return (f"lr{LR:.2e}_wd{WD:.2e}_dw{DW:.2f}_fw{FW:.2f}"
            f"_a{alpha:.2f}_bs{BATCH_SIZE}"+ ("_noatt" if NO_ATTENTION else ""))

def best_val_from_csv(csv_path):
    best, best_epoch = float("inf"), -1
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["Val_Loss"])
            except (TypeError, ValueError):
                continue
            if not math.isnan(v) and v < best:
                best, best_epoch = v, int(row["Epoch"])
    return best, best_epoch



def run_trial(idx, alpha):
    tag = run_tag(alpha)
    print(f"\n=== trial {idx}/{len(ALPHAS)}: alpha={alpha:.2f}  ({tag}) ===", flush=True)
    t0 = time.time()
    cmd = [sys.executable, "trainer/train.py", 
           "--epochs", str(EPOCHS_PER_TRIAL),
           "--batch_size", str(BATCH_SIZE),
           "--lr", str(LR),
           "--weight_decay", str(WD),
           "--dice_weight", str(DW),
           "--focal_weight", str(FW),
           "--alpha", str(alpha)]
    if NO_ATTENTION:
        cmd.append("--no_attention")
    proc =subprocess.run(cmd, cwd=str(REPO_ROOT))
    
    elapsed = time.time() - t0
    csv_path = WEIGHTS_DIR / f"metrics_{tag}.csv"
    base = {"trial": idx, "alpha": alpha, "tag": tag, "elapsed_s": round(elapsed, 1)}
    if proc.returncode != 0 or not csv_path.exists():
        return {**base, "best_val": float("nan"), "best_epoch": -1, "status": "failed"}
    bv, be = best_val_from_csv(csv_path)
    return {**base, "best_val": bv, "best_epoch": be, "status": "ok"}



def main():
    results = [run_trial(i, a) for i, a in enumerate(ALPHAS, 1)]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    results.sort(key=lambda r: (math.isnan(r["best_val"]), r["best_val"]))
    print("\n=== ALPHA SWEEP RANKS ===", flush=True)
    print(f"{'rank':<5}{'alpha':<7}{'best_val':<11}{'epoch':<7}{'min':<6}{'status'}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5}{r['alpha']:<7.2f}{r['best_val']:<11.4f}"
              f"{r['best_epoch']:<7}{r['elapsed_s']/60:<6.1f}{r['status']}")
    print(f"\nFull results in: {RESULTS_CSV}")



if __name__ == "__main__":
    main()
