import argparse
import re
import csv
import logging
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.unet2_5d import UNet2_5D
from trainer.dataset import PICAI25DDataset
from trainer.train import feature_list_from_base
import yaml


def parse_checkpoint_path(pth_path: Path):
    """
    Returns (run_tag, use_attention, legacy_lr, feature_base) or Nones if unrecognized.
    legacy_lr: for old lr-only checkpoints, folder run_lr_{legacy_lr}; else None.
    """
    name = pth_path.name
    m_full = re.match(
        r"unet25d_lr([\d.eE+-]+)_focal([\d.eE+-]+)_wd([\d.eE+-]+)_fcb(\d+)_(attn|noattn)_best\.pth$",
        name,
    )
    if m_full:
        lr, fa, wd, fcb, attn = m_full.groups()
        run_tag = f"lr{lr}_focal{fa}_wd{wd}_fcb{fcb}_{attn}"
        use_attention = attn == "attn"
        return run_tag, use_attention, None, int(fcb)

    m_mid = re.match(
        r"unet25d_lr([\d.eE+-]+)_focal([\d.eE+-]+)_(attn|noattn)_best\.pth$",
        name,
    )
    if m_mid:
        lr, fa, attn = m_mid.groups()
        run_tag = f"lr{lr}_focal{fa}_{attn}"
        use_attention = attn == "attn"
        return run_tag, use_attention, None, 64

    m_old = re.match(r"unet25d_(lr[\d.]+)_best\.pth$", name)
    if m_old:
        lr_token = m_old.group(1)
        legacy_lr = lr_token[2:] if lr_token.startswith("lr") else lr_token
        return lr_token, True, legacy_lr, 64

    return None, None, None, None


def plot_learning_curve(csv_path, output_path):
    epochs, train_loss, val_loss = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_loss.append(float(row['Train_Loss']))
            val_loss.append(float(row['Val_Loss']))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Combined Focal-Dice Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def evaluate_model(model, loader, device, output_dir):
    model.eval()
    total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
    best_dice = -1.0
    best_visuals = None

    with torch.no_grad():
        for t2, adc, mask in tqdm(loader, desc="Evaluating", leave=False):
            t2, adc, mask = t2.to(device), adc.to(device), mask.to(device)
            
            with torch.amp.autocast('cuda'):
                logits, _ = model(t2, adc)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            preds_np = preds.cpu().numpy().astype(bool)
            mask_np = mask.cpu().numpy().astype(bool)
            
            tp = np.logical_and(preds_np, mask_np).sum()
            fp = np.logical_and(preds_np, np.logical_not(mask_np)).sum()
            fn = np.logical_and(np.logical_not(preds_np), mask_np).sum()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if mask_np.sum() > 0:
                current_dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
                if current_dice > best_dice:
                    best_dice = current_dice
                    best_visuals = {
                        "t2": t2[0, 1].cpu().numpy(),
                        "mask": mask_np[0, 0],
                        "pred": preds_np[0, 0],
                        "dice": current_dice
                    }

    global_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + 1e-6)
    sensitivity = total_tp / (total_tp + total_fn + 1e-6)

    # Plot sample shot.
    if best_visuals is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Best Prediction (Slice Dice: {best_visuals['dice']:.4f})", fontsize=16)
        
        axes[0].imshow(best_visuals['t2'], cmap='gray')
        axes[0].set_title("Raw T2 MRI")
        axes[0].axis('off')
        
        axes[1].imshow(best_visuals['t2'], cmap='gray')
        axes[1].imshow(np.ma.masked_where(best_visuals['mask'] == 0, best_visuals['mask']), cmap='Greens', alpha=0.6)
        axes[1].set_title("Ground Truth (Green)")
        axes[1].axis('off')
        
        axes[2].imshow(best_visuals['t2'], cmap='gray')
        axes[2].imshow(np.ma.masked_where(best_visuals['pred'] == 0, best_visuals['pred']), cmap='Reds', alpha=0.6)
        axes[2].set_title("Model Prediction (Red)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "best_prediction.png", dpi=300, bbox_inches='tight')
        plt.close()

    return global_dice, sensitivity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Load configuration.
    with open("config/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    CROP_SIZE = config['dataset']['crop_size']
    STRATEGY = config['dataset']['strategy']
    DATA_DIR = Path(f"data/processed_tensors_{STRATEGY}_{CROP_SIZE}")
    MANIFEST_PATH = Path(f"data/ml_manifest_{STRATEGY}_{CROP_SIZE}.json")

    dataset = PICAI25DDataset(MANIFEST_PATH, DATA_DIR, split=args.split, pad_edges=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    weights_dir = Path("models/weights")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Find all trained models (new tagged names and legacy lr-only names).
    pth_files = sorted(set(weights_dir.glob("unet25d_*_best.pth")))
    
    if not pth_files:
        logging.error("No weight files found in models/weights/")
        return

    for pth_path in pth_files:
        run_tag, use_attention, legacy_lr, feature_base = parse_checkpoint_path(pth_path)
        if run_tag is None:
            logging.warning(f"Skipping unrecognized checkpoint: {pth_path.name}")
            continue

        if legacy_lr is not None:
            run_dir = results_dir / f"run_lr_{legacy_lr}"
            csv_glob = f"metrics_{run_tag}_bs*.csv"
        else:
            run_dir = results_dir / f"run_{run_tag}"
            csv_glob = f"metrics_{run_tag}_bs*.csv"

        features = feature_list_from_base(feature_base)

        logging.info(f"\n======================================")
        logging.info(
            f"Processing Run: {run_tag} (attention={use_attention}, feature_base={feature_base})"
        )
        logging.info(f"======================================")

        run_dir.mkdir(exist_ok=True)

        csv_files = list(weights_dir.glob(csv_glob))
        if csv_files:
            plot_learning_curve(csv_files[0], run_dir / "learning_curve.png")
            logging.info(f"  [+] Learning curve saved.")
        else:
            logging.warning(f"  [!] No matching CSV for pattern {csv_glob}")

        logging.info(f"  [*] Evaluating on {args.split.upper()} set...")
        model = UNet2_5D(
            in_channels=3,
            n_classes=1,
            features=features,
            use_attention=use_attention,
        ).to(device)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        global_dice, sensitivity = evaluate_model(model, loader, device, run_dir)

        summary_path = run_dir / "metrics_summary.log"
        with open(summary_path, "w") as f:
            f.write(f"Run tag: {run_tag}\n")
            f.write(f"feature_base: {feature_base} (widths {features})\n")
            f.write(f"use_attention: {use_attention}\n")
            f.write(f"Evaluation Split: {args.split.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Global Dice Score: {global_dice:.4f}\n")
            f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")

        logging.info(f"  [+] Evaluation complete. Global Dice: {global_dice:.4f}")
        logging.info(f"  [+] All assets saved to {run_dir}/")

if __name__ == "__main__":
    main()