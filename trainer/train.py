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


def _slug_float(x: float) -> str:
    """Compact, filename-safe float (e.g. 5e-4 -> 0.0005)."""
    return f"{x:g}"


def feature_list_from_base(feature_base: int) -> list:
    """UNet width pyramid: [b, 2b, 4b, 8b] (default b=64 matches original)."""
    b = int(feature_base)
    if b < 8:
        raise ValueError("feature_base must be >= 8 (CrossAttention needs enough channels).")
    return [b, b * 2, b * 4, b * 8]


def build_run_tag(
    lr: float,
    focal_alpha: float,
    use_attention: bool,
    weight_decay: float,
    feature_base: int,
) -> str:
    attn = "attn" if use_attention else "noattn"
    return (
        f"lr{_slug_float(lr)}_focal{_slug_float(focal_alpha)}"
        f"_wd{_slug_float(weight_decay)}_fcb{int(feature_base)}_{attn}"
    )


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
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha (CombinedFocalDiceLoss)",
    )
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Disable cross-attention; use 1x1 concat fusion at the bottleneck",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--feature_base",
        type=int,
        default=64,
        help="First U-Net level width; pyramid is [b, 2b, 4b, 8b] (default 64 => [64,128,256,512])",
    )
    return parser.parse_args()

def main():
    args = setup_argparser()
    use_attention = not args.no_attention
    features = feature_list_from_base(args.feature_base)
    run_tag = build_run_tag(
        args.lr,
        args.focal_alpha,
        use_attention,
        args.weight_decay,
        args.feature_base,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modern PyTorch AMP Scaler to prevent gradient underflow in float16.
    scaler = torch.amp.GradScaler('cuda')
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(
        f"--- Booting Training on {device.type.upper()} | LR={args.lr} focal_alpha={args.focal_alpha} "
        f"wd={args.weight_decay} features={features} attention={use_attention} ---"
    )

    # Load Pipeline Configuration.
    with open("config/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    CROP_SIZE = config['dataset']['crop_size']
    STRATEGY = config['dataset']['strategy']
    
    DATA_DIR = Path(f"data/processed_tensors_{STRATEGY}_{CROP_SIZE}")
    MANIFEST_PATH = Path(f"data/ml_manifest_{STRATEGY}_{CROP_SIZE}.json")
    WEIGHTS_DIR = Path("models/weights")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup CSV Logger (tag encodes lr, focal alpha, attention variant).
    csv_path = WEIGHTS_DIR / f"metrics_{run_tag}_bs{args.batch_size}.csv"
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
    model = UNet2_5D(
        in_channels=3,
        n_classes=1,
        features=features,
        use_attention=use_attention,
    ).to(device)
    criterion = CombinedFocalDiceLoss(alpha=args.focal_alpha)
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
            save_path = WEIGHTS_DIR / f"unet25d_{run_tag}_best.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"  [*] New best model saved to {save_path.name}")

        # Early Stopping Check.
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info(f"\n[!] Early stopping triggered at epoch {epoch}. Validation loss plateaued.")
            break

if __name__ == "__main__":
    main()