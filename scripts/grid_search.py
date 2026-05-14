#!/usr/bin/env python3
"""
Hyperparameter grid search: trains each combo via trainer/train.py, then scores
checkpoints by **validation global Dice** (same definition as analyze_runs).

Also logs test Dice for convenience; **rank and select the winner by val_dice** to
avoid tuning on the test set.

Usage (from repository root):
    python scripts/grid_search.py

Optional:
    python scripts/grid_search.py --quick          # tiny grid for smoke tests
    python scripts/grid_search.py --batch_size 32
    python scripts/grid_search.py --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.unet2_5d import UNet2_5D
from trainer.dataset import PICAI25DDataset
from trainer.train import build_run_tag, feature_list_from_base


def global_dice_and_sensitivity(model, loader, device):
    """Micro-averaged Dice and recall over all pixels (matches analyze_runs)."""
    model.eval()
    total_tp = total_fp = total_fn = 0.0
    with torch.no_grad():
        for t2, adc, mask in tqdm(loader, desc="Dice eval", leave=False):
            t2, adc, mask = t2.to(device), adc.to(device), mask.to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits, _ = model(t2, adc)
            else:
                logits, _ = model(t2, adc)
            preds = (torch.sigmoid(logits) > 0.5).float()
            preds_np = preds.cpu().numpy().astype(bool)
            mask_np = mask.cpu().numpy().astype(bool)
            total_tp += np.logical_and(preds_np, mask_np).sum()
            total_fp += np.logical_and(preds_np, np.logical_not(mask_np)).sum()
            total_fn += np.logical_and(np.logical_not(preds_np), mask_np).sum()
    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + 1e-6)
    sens = total_tp / (total_tp + total_fn + 1e-6)
    return float(dice), float(sens)


def run_one_train(
    *,
    lr: float,
    weight_decay: float,
    feature_base: int,
    focal_alpha: float,
    use_attention: bool,
    batch_size: int,
    epochs: int,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "trainer" / "train.py"),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--feature_base",
        str(feature_base),
        "--focal_alpha",
        str(focal_alpha),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
    ]
    if not use_attention:
        cmd.append("--no_attention")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def load_and_score_checkpoint(
    pth_path: Path,
    *,
    use_attention: bool,
    feature_base: int,
    manifest_path: Path,
    data_dir: Path,
    device: torch.device,
):
    features = feature_list_from_base(feature_base)
    model = UNet2_5D(
        in_channels=3,
        n_classes=1,
        features=features,
        use_attention=use_attention,
    ).to(device)
    state = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    val_ds = PICAI25DDataset(manifest_path, data_dir, split="val", pad_edges=False)
    test_ds = PICAI25DDataset(manifest_path, data_dir, split="test", pad_edges=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    val_dice, val_sens = global_dice_and_sensitivity(model, val_loader, device)
    test_dice, test_sens = global_dice_and_sensitivity(model, test_loader, device)
    return val_dice, val_sens, test_dice, test_sens


def main():
    parser = argparse.ArgumentParser(description="Grid search for 2.5D U-Net (ranked by val Dice)")
    parser.add_argument("--quick", action="store_true", help="Run a 4-trial smoke grid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Fixed focal alpha for all trials")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(REPO_ROOT / "config" / "dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    crop = config["dataset"]["crop_size"]
    strategy = config["dataset"]["strategy"]
    data_dir = REPO_ROOT / f"data/processed_tensors_{strategy}_{crop}"
    manifest_path = REPO_ROOT / f"data/ml_manifest_{strategy}_{crop}.json"

    if not manifest_path.exists():
        logging.error(f"Missing manifest {manifest_path}. Run generate_splits.py first.")
        sys.exit(1)

    if args.quick:
        grid_lr = [5e-4]
        grid_wd = [1e-4]
        grid_fcb = [64]
        grid_attn = [True, False]
    else:
        grid_lr = [3e-4, 5e-4, 7e-4, 1e-3]
        grid_wd = [1e-4, 5e-4]
        grid_fcb = [32, 48, 64]
        grid_attn = [True, False]

    trials = list(product(grid_lr, grid_wd, grid_fcb, grid_attn))
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_csv = out_dir / f"grid_search_summary_{stamp}.csv"
    summary_json = out_dir / f"grid_search_summary_{stamp}.json"

    fieldnames = [
        "trial",
        "lr",
        "weight_decay",
        "feature_base",
        "use_attention",
        "focal_alpha",
        "train_exit_code",
        "checkpoint",
        "val_dice",
        "val_sensitivity",
        "test_dice",
        "test_sensitivity",
        "notes",
    ]

    rows = []
    best_row = None

    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for i, (lr, wd, fcb, attn) in enumerate(trials, start=1):
            logging.info(f"\n======== Trial {i}/{len(trials)} | lr={lr} wd={wd} fcb={fcb} attn={attn} ========")
            rc = run_one_train(
                lr=lr,
                weight_decay=wd,
                feature_base=fcb,
                focal_alpha=args.focal_alpha,
                use_attention=attn,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )

            run_tag = build_run_tag(lr, args.focal_alpha, attn, wd, fcb)
            ckpt = REPO_ROOT / "models" / "weights" / f"unet25d_{run_tag}_best.pth"
            notes = ""
            val_dice = val_sens = test_dice = test_sens = float("nan")

            if rc != 0:
                notes = f"train_failed_exit_{rc}"
                logging.warning(notes)
            elif not ckpt.is_file():
                notes = "missing_checkpoint"
                logging.warning(notes)
            else:
                try:
                    val_dice, val_sens, test_dice, test_sens = load_and_score_checkpoint(
                        ckpt,
                        use_attention=attn,
                        feature_base=fcb,
                        manifest_path=manifest_path,
                        data_dir=data_dir,
                        device=device,
                    )
                except Exception as e:
                    notes = f"eval_error: {e}"
                    logging.exception("Evaluation failed")

            row = {
                "trial": i,
                "lr": lr,
                "weight_decay": wd,
                "feature_base": fcb,
                "use_attention": attn,
                "focal_alpha": args.focal_alpha,
                "train_exit_code": rc,
                "checkpoint": str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_file() else "",
                "val_dice": f"{val_dice:.6f}" if val_dice == val_dice else "",
                "val_sensitivity": f"{val_sens:.6f}" if val_sens == val_sens else "",
                "test_dice": f"{test_dice:.6f}" if test_dice == test_dice else "",
                "test_sensitivity": f"{test_sens:.6f}" if test_sens == test_sens else "",
                "notes": notes,
            }
            writer.writerow(row)
            fcsv.flush()
            rows.append({**row, "val_dice_f": val_dice})

            if val_dice == val_dice and (best_row is None or val_dice > best_row["val_dice_f"]):
                best_row = {**row, "val_dice_f": val_dice}

    with open(summary_json, "w") as jf:
        json.dump(
            {
                "created_utc": stamp,
                "manifest": str(manifest_path),
                "ranked_by": "val_dice",
                "trials": len(trials),
                "rows": [{k: r[k] for k in fieldnames} for r in rows],
                "best": {k: v for k, v in (best_row or {}).items() if k != "val_dice_f"},
            },
            jf,
            indent=2,
        )

    logging.info(f"\nSaved table: {summary_csv}")
    logging.info(f"Saved JSON:  {summary_json}")
    if best_row:
        logging.info(
            "\n*** Best trial (by validation Dice) ***\n"
            f"  val_dice:  {best_row['val_dice']}\n"
            f"  test_dice: {best_row['test_dice']}\n"
            f"  lr: {best_row['lr']} | weight_decay: {best_row['weight_decay']} | "
            f"feature_base: {best_row['feature_base']} | use_attention: {best_row['use_attention']}\n"
            f"  checkpoint: {best_row['checkpoint']}"
        )
    else:
        logging.error("No successful trial produced a validation Dice.")


if __name__ == "__main__":
    main()
