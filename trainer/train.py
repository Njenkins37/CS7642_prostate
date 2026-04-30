import argparse
import math
import yaml
import logging
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm

from models.unet2_5d import UNet2_5D
from trainer.dataset import PICAI25DDataset
from trainer.losses import CombinedFocalDiceLoss

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if math.isnan(val_loss):
            return
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def setup_argparser():
    parser = argparse.ArgumentParser(description="PI-CAI U-Net Training Engine")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    # Defaulting to 64, but this is for my GPU with 16GB VRAM GPU (128x128x3 with Mixed Precision).
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per forward pass")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW L2 regularization")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice term weight in CombinedFocalDiceLoss")
    parser.add_argument("--focal_weight", type=float, default=0.5, help="Focal term weight in CombinedFocalDiceLoss")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal Loss alpha (positive-class weight)")
    parser.add_argument("--no_attention", action="store_true", help="Disable cross-attention bottleneck; use the concat+1x1 fallback fusion.")
    return parser.parse_args()

def main():
    args = setup_argparser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modern PyTorch AMP Scaler to prevent gradient underflow in float16.
    scaler = torch.amp.GradScaler('cuda')
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"--- Booting Training Engine on {device.type.upper()} (LR: {args.lr}) ---")

    # Load Pipeline Configuration.
    with open("config/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    CROP_SIZE = config['dataset']['crop_size']
    STRATEGY = config['dataset']['strategy']
    
    DATA_DIR = Path(f"data/processed_tensors_{STRATEGY}_{CROP_SIZE}")
    MANIFEST_PATH = Path(f"data/ml_manifest_{STRATEGY}_{CROP_SIZE}.json")
    WEIGHTS_DIR = Path("models/weights")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup CSV Logger for the presentation.
    RUN_TAG = (f"lr{args.lr:.2e}_wd{args.weight_decay:.2e}"
               f"_dw{args.dice_weight:.2f}_fw{args.focal_weight:.2f}"
               f"_a{args.alpha:.2f}_bs{args.batch_size}"+ ("_noatt" if args.no_attention else ""))
    csv_path = WEIGHTS_DIR / f"metrics_{RUN_TAG}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Learning_Rate"])

    # Initialize DataLoaders.
    train_dataset = PICAI25DDataset(MANIFEST_PATH, DATA_DIR, split="train", pad_edges=False)
    val_dataset = PICAI25DDataset(MANIFEST_PATH, DATA_DIR, split="val", pad_edges=False)
    
    # Change num_workers accordingly to your CPU.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)
    
    logging.info(f"Loaded {len(train_dataset)} Train slices and {len(val_dataset)} Validation slices.")

    # Initialize Architecture.
    model = UNet2_5D(in_channels=3, n_classes=1,  use_attention=not args.no_attention).to(device) #attention switch
    criterion = CombinedFocalDiceLoss(
        alpha=args.alpha,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Drops learning rate by half if validation loss stalls for 3 epochs.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Initialize Early Stopping.
    early_stopping = EarlyStopping(patience=7)

    # Master Training Loop.
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Training Pass.
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for t2, adc, mask in train_pbar:
            t2, adc, mask = t2.to(device), adc.to(device), mask.to(device)
            
            # Clear gradients after every batch but set to none instead of zero to save memory.
            optimizer.zero_grad(set_to_none=True)
            
            # Cast forward pass to float16.
            with torch.amp.autocast('cuda'):
                logits, _ = model(t2, adc)
                loss = criterion(logits, mask)

            if torch.isnan(loss):
                logging.error(f"\n[FATAL] NaN Loss detected at Epoch {epoch}. Model weights are corrupted. Aborting this LR run.")
                return
                
            # Scale gradients back up to float32 and backpropagate.
            scaler.scale(loss).backward()

            # Gradient Clipping.
            # We must unscale the gradients first so we are clipping their true mathematical values.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation Pass.
        model.eval()
        val_loss = 0.0
        val_batches = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  ")
        with torch.no_grad():
            for t2, adc, mask in val_pbar:
                t2, adc, mask = t2.to(device), adc.to(device), mask.to(device)

                with torch.amp.autocast('cuda'):
                    logits, _ = model(t2, adc)
                    loss = criterion(logits, mask)

                if torch.isnan(loss):
                    logging.warning("[WARNING] NaN val loss on this batch — skipping.")
                    continue

                val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
        current_lr = optimizer.param_groups[0]['lr']

        # Guard the scheduler: ReduceLROnPlateau corrupts its state if fed NaN.
        if not math.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)
        
        logging.info(f"End of Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")
        
        # Log to CSV.
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_val_loss, current_lr])
        
        # Model Checkpointing.
        if not math.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = WEIGHTS_DIR / f"unet25d_{RUN_TAG}_best.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"  [*] New best model saved to {save_path.name}")

        # Early Stopping Check.
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info(f"\n[!] Early stopping triggered at epoch {epoch}. Validation loss plateaued.")
            break

if __name__ == "__main__":
    main()