import yaml
import json
import random
import logging
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    with open("config/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    CROP_SIZE = config['dataset']['crop_size']
    STRATEGY = config['dataset']['strategy']
    MODE = config['balancing']['mode']
    CLINICAL_NEGS = config['balancing']['clinical_negative_cases']
    TRAIN_R = config['splits']['train_ratio']
    VAL_R = config['splits']['val_ratio']
    SEED = config['splits']['random_seed']

    INVENTORY_FILE = Path("data") / f"processed_inventory_{STRATEGY}_{CROP_SIZE}.json"
    MANIFEST_FILE = Path("data") / f"ml_manifest_{STRATEGY}_{CROP_SIZE}.json"

    if not INVENTORY_FILE.exists():
        logging.error(f"Inventory missing at {INVENTORY_FILE}. Run preprocess_dataset.py first.")
        exit(1)

    with open(INVENTORY_FILE, "r") as f:
        inventory = json.load(f)

    pos_cases = inventory.get("clean_positives", [])
    all_neg_cases = inventory.get("clean_negatives", [])

    logging.info(f"--- Generating ML Splits | Strategy: {STRATEGY.upper()} | Crop: {CROP_SIZE} ---")

    # Balance Classes based on YAML Config.
    random.seed(SEED)
    if MODE == "auto_balance":
        logging.info(f"Mode [Auto-Balance]: Selecting {len(pos_cases)} negatives from {len(all_neg_cases)} available.")
        neg_cases = random.sample(all_neg_cases, len(pos_cases))
        
    elif MODE == "clinical_override":
        logging.info(f"Mode [Clinical Override]: Using {len(CLINICAL_NEGS)} YAML-provided negative cases.")
        # Intersect with clean cases to ensure we don't include a patient that crashed during preprocessing
        neg_cases = [c for c in CLINICAL_NEGS if c in all_neg_cases]
        if len(neg_cases) != len(CLINICAL_NEGS):
            logging.warning("  [!] Warning: Some clinical negatives were rejected during preprocessing.")
            
    else:
        logging.error(f"Unknown balancing mode: {MODE}")
        exit(1)

    # We shuffle them independently so the distributions are even.
    # Here, the shuffle is based on patients, not tensors, to prevent data leakage.
    random.shuffle(pos_cases)
    random.shuffle(neg_cases)

    def get_splits(lst, train_r, val_r):
        n = len(lst)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    pos_train, pos_val, pos_test = get_splits(pos_cases, TRAIN_R, VAL_R)
    neg_train, neg_val, neg_test = get_splits(neg_cases, TRAIN_R, VAL_R)

    # Build and Save the PyTorch Manifest.
    manifest = {
        "metadata": config,
        "splits": {
            "train": pos_train + neg_train,
            "val": pos_val + neg_val,
            "test": pos_test + neg_test
        },
        "class_balance": {
            "train": {"pos": len(pos_train), "neg": len(neg_train)},
            "val": {"pos": len(pos_val), "neg": len(neg_val)},
            "test": {"pos": len(pos_test), "neg": len(neg_test)}
        }
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

    logging.info(f"\n--- ML Manifest Generated ---")
    logging.info(f"Train: {len(manifest['splits']['train'])} | Val: {len(manifest['splits']['val'])} | Test: {len(manifest['splits']['test'])}")
    logging.info(f"Saved to {MANIFEST_FILE}")