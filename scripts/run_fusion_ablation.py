#!/usr/bin/env python3
"""Minimal bottleneck fusion ablation: cross-attention on vs 1x1 concat (off).

Fixed: lr=5e-4, weight_decay=1e-4, feature_base=64, focal_alpha=0.25.

From repo root:
    python scripts/run_fusion_ablation.py

Then generate curves + prediction PNGs:
    python scripts/analyze_runs.py --split test
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN = [
    sys.executable,
    str(REPO_ROOT / "trainer" / "train.py"),
    "--lr",
    "5e-4",
    "--weight_decay",
    "1e-4",
    "--feature_base",
    "64",
    "--focal_alpha",
    "0.25",
]


def main():
    for use_attention in (True, False):
        cmd = list(TRAIN)
        if not use_attention:
            cmd.append("--no_attention")
        name = "cross_attention" if use_attention else "linear_fusion"
        print(f"\n=== {name} ===")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            sys.exit(proc.returncode)

    print("\nOK. Next: python scripts/analyze_runs.py --split test")


if __name__ == "__main__":
    main()
