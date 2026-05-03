import argparse
import re
import csv
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.unet2_5d import UNet2_5D
from trainer.dataset import PICAI25DDataset
import yaml

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

    # Find all trained models.
    pth_files = list(weights_dir.glob("unet25d_lr*_best.pth"))
    
    if not pth_files:
        logging.error("No weight files found in models/weights/")
        return

    for pth_path in pth_files:
        # Derive RUN_TAG from the .pth filename: unet25d_<RUN_TAG>_best.pth. Should work for legacy runs too
        stem = pth_path.stem  # drops .pth
        if not stem.startswith("unet25d_") or not stem.endswith("_best"):
            continue
        run_tag = stem[len("unet25d_"):-len("_best")]

        # Extract LR for logs (scientific notation fix for readability).
        lr_match = re.search(r"lr([0-9.eE+\-]+?)(?:_|$)", run_tag)
        lr_val = lr_match.group(1) if lr_match else run_tag

        logging.info(f"\n======================================")
        logging.info(f"Processing Run: {run_tag}")
        logging.info(f"======================================")

        # Create run-specific sub directory to avoid overwrites
        run_dir = results_dir / f"run_{run_tag}"
        run_dir.mkdir(exist_ok=True)

        # Plot learning curve, but try naming match first, revert to legacy write
        exact = weights_dir / f"metrics_{run_tag}.csv"
        if exact.exists():
            csv_files = [exact]
        else:
            csv_files = list(weights_dir.glob(f"metrics_lr{lr_val}_bs*.csv"))
        if csv_files:
            plot_learning_curve(csv_files[0], run_dir / "learning_curve.png")
            logging.info(f"  [+] Learning curve saved.")
        else:
            logging.warning(f"  [!] No matching CSV found for LR {lr_val}.")

        # Model evaluation. Match to saved dict; no attention runs  use the fallback
        use_attention = not run_tag.endswith("_noatt")
        model = UNet2_5D(in_channels=3, n_classes=1, use_attention=use_attention).to(device)
        logging.info(f"  [*] Evaluating on {args.split.upper()} set... (attention={'on' if use_attention else 'off'})")
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        global_dice, sensitivity = evaluate_model(model, loader, device, run_dir)

        # Save Text Summary.
        summary_path = run_dir / "metrics_summary.log"
        with open(summary_path, "w") as f:
            f.write(f"Run Configuration: LR = {lr_val}\n")
            f.write(f"Evaluation Split: {args.split.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Global Dice Score: {global_dice:.4f}\n")
            f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
            
        logging.info(f"  [+] Evaluation complete. Global Dice: {global_dice:.4f}")
        logging.info(f"  [+] All assets saved to {run_dir}/")

if __name__ == "__main__":
    main()