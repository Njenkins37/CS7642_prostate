import argparse
import numpy as np
import json
import logging
import random
from pathlib import Path
from scipy.ndimage import center_of_mass

from utils.loader import load_picai_case, pos_list, IMAGES_ROOT, LABELS_ROOT
from preprocess_dataset import setup_argparser, get_crop_indices

def extract_negative_cookie_cutter(case_id, args, output_dir, qc_results):
    try:
        data = load_picai_case(case_id, IMAGES_ROOT, LABELS_ROOT)
        t2, adc, gland, lesion = data["t2"], data["adc"], data["gland_t2"], data["lesion_t2"]

        if t2.shape[0] < args.crop_size or t2.shape[1] < args.crop_size:
            qc_results["rejected_too_small_t2"].append(case_id)
            return
        if np.sum(gland) == 0:
            qc_results["rejected_no_prostate"].append(case_id)
            return

        # Always strict for negatives: center exactly on the gland.
        x_c, y_c, _ = center_of_mass(gland)
        x_start, x_end = get_crop_indices(x_c, args.crop_size, t2.shape[0])
        y_start, y_end = get_crop_indices(y_c, args.crop_size, t2.shape[1])

        np.savez_compressed(
            output_dir / f"{case_id}_clean.npz",
            t2=t2[x_start:x_end, y_start:y_end, :],
            adc=adc[x_start:x_end, y_start:y_end, :],
            lesion=lesion[x_start:x_end, y_start:y_end, :] # This is now perfectly 0s thanks to the loader patch
        )
        
        logging.info(f"  [SUCCESS] Saved NEGATIVE tensor for {case_id}")
        qc_results["clean_cases"].append(case_id)

    except Exception as e:
        logging.error(f"  [!] CRASHED {case_id}: {e}")
        qc_results["crashed_unknown_error"].append(case_id)

def generate_splits(pos_json_path, neg_clean_list, output_json_path):
    """Enforces a strict, randomized 70/15/15 split across both classes."""
    with open(pos_json_path, 'r') as f:
        pos_data = json.load(f)
    
    pos_cases = pos_data.get("clean_cases", [])
    neg_cases = neg_clean_list
    
    random.seed(42) # Deterministic splitting
    random.shuffle(pos_cases)
    random.shuffle(neg_cases)
    
    def split_list(lst):
        n = len(lst)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
        
    pos_train, pos_val, pos_test = split_list(pos_cases)
    neg_train, neg_val, neg_test = split_list(neg_cases)
    
    final_manifest = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
        "class_balance": {
            "train": {"pos": len(pos_train), "neg": len(neg_train)},
            "val": {"pos": len(pos_val), "neg": len(neg_val)},
            "test": {"pos": len(pos_test), "neg": len(neg_test)}
        }
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(final_manifest, f, indent=4)
    logging.info(f"\nFinal ML Manifest Saved: {len(final_manifest['train'])} Train | {len(final_manifest['val'])} Val | {len(final_manifest['test'])} Test")

if __name__ == "__main__":
    args = setup_argparser()
    OUTPUT_DIR = Path(f"data/processed_tensors_{args.strategy}_{args.crop_size}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_FILE = Path("data") / f"preprocessing_negatives_{args.strategy}_{args.crop_size}.log"
    POS_JSON_FILE = Path("data") / f"dataset_splits_{args.strategy}_{args.crop_size}.json"
    FINAL_MANIFEST = Path("data") / f"ml_manifest_{args.strategy}_{args.crop_size}.json"

    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode='w')]
    )

    # Dynamically discover the negative cases AFTER logging is configured.
    all_cases = [int(p.name) for p in IMAGES_ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    neg_list = sorted(list(set(all_cases) - set(pos_list)))
    
    logging.info(f"Discovered {len(all_cases)} total patients. {len(pos_list)} positive, {len(neg_list)} negative.")

    qc_results = {"clean_cases": [], "rejected_too_small_t2": [], "rejected_no_prostate": [], "crashed_unknown_error": []}

    logging.info("--- Extracting Negative Cases ---")
    for cid in neg_list:
        extract_negative_cookie_cutter(cid, args, OUTPUT_DIR, qc_results)
        
    logging.info("\n--- Generating Final Train/Val/Test Splits ---")
    if POS_JSON_FILE.exists():
        generate_splits(POS_JSON_FILE, qc_results["clean_cases"], FINAL_MANIFEST)
    else:
        logging.error(f"Could not find positive cases JSON at {POS_JSON_FILE}. Run preprocess_dataset.py first.")