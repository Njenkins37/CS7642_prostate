import yaml
import numpy as np
import json
import logging
from pathlib import Path
from scipy.ndimage import center_of_mass

from utils.loader import load_picai_case, pos_list, IMAGES_ROOT, LABELS_ROOT

def get_crop_indices(center, size, max_dim):
    """Calculates start/end indices. Shifts if out of bounds to prevent crashes."""
    start = int(center - (size // 2))
    end = start + size
    
    if start < 0:
        start = 0
        end = size
    if end > max_dim:
        end = max_dim
        start = end - size
    return start, end

def extract_cookie_cutter(case_id, is_positive, strategy, crop_size, output_dir, inventory):
    try:
        data = load_picai_case(case_id, IMAGES_ROOT, LABELS_ROOT)
        t2, adc = data["t2"], data["adc"]
        gland, lesion = data["gland_t2"], data["lesion_t2"]

        # Defensive Checks.
        if t2.shape[0] < crop_size or t2.shape[1] < crop_size:
            logging.info(f"  [!] REJECTED {case_id}: T2 image too small.")
            inventory["rejected_too_small_t2"].append(case_id)
            return
        if adc.shape[0] < crop_size or adc.shape[1] < crop_size:
            logging.info(f"  [!] REJECTED {case_id}: ADC image too small.")
            inventory["rejected_too_small_adc"].append(case_id)
            return
        if np.sum(gland) == 0:
            logging.info(f"  [!] REJECTED {case_id}: No prostate gland mask.")
            inventory["rejected_no_prostate"].append(case_id)
            return

        # Initial Crop (Centered on Prostate Gland, Bosma22b).
        x_c, y_c, _ = center_of_mass(gland)
        x_start, x_end = get_crop_indices(x_c, crop_size, t2.shape[0])
        y_start, y_end = get_crop_indices(y_c, crop_size, t2.shape[1])

        # Apply cropping strategy ONLY if the patient actually has a tumor (positive case)
        if is_positive:
            lesion_cropped = lesion[x_start:x_end, y_start:y_end, :]

            # When the tumor is clipped by the bounding box, "shift" strategy will recalculate the center on the tumor, not the gland.
            if np.sum(lesion_cropped) != np.sum(lesion):
                if strategy == "strict":
                    logging.info(f"  [!] REJECTED {case_id}: Tumor clipped (Strict Mode).")
                    inventory["rejected_clipped_tumor_strict"].append(case_id)
                    return
                
                elif strategy == "shift":
                    logging.info(f"  [*] SHIFTING {case_id}: Tumor clipped. Recalculating center.")
                    inventory["shifted_cases"].append(case_id)
                    
                    # Recalculate center based on the tumor, not the gland.
                    lx_c, ly_c, _ = center_of_mass(lesion)
                    x_start, x_end = get_crop_indices(lx_c, crop_size, t2.shape[0])
                    y_start, y_end = get_crop_indices(ly_c, crop_size, t2.shape[1])
                    
                    # Verify the shift actually saved the tumor.
                    if np.sum(lesion[x_start:x_end, y_start:y_end, :]) != np.sum(lesion):
                        logging.info(f"  [!] REJECTED {case_id}: Tumor too massive to fit in {crop_size}x{crop_size}.")
                        inventory["rejected_unrecoverable_clipping"].append(case_id)
                        return

        # Finalize and save.
        # Note that all the slides (Z-axis) are saved including the ones without cancer regions.
        np.savez_compressed(
            output_dir / f"{case_id}_clean.npz",
            t2=t2[x_start:x_end, y_start:y_end, :],
            adc=adc[x_start:x_end, y_start:y_end, :],
            lesion=lesion[x_start:x_end, y_start:y_end, :]
        )
        
        logging.info(f"  [SUCCESS] Saved {'POS' if is_positive else 'NEG'} tensors for {case_id}")
        if is_positive:
            inventory["clean_positives"].append(case_id)
        else:
            inventory["clean_negatives"].append(case_id)

    except FileNotFoundError:
        logging.info(f"  [-] SKIPPED {case_id}: Missing human label.")
        inventory["skipped_missing_labels"].append(case_id)
    except Exception as e:
        logging.error(f"  [!] CRASHED {case_id}: {e}")
        inventory["crashed_unknown_error"].append(case_id)


if __name__ == "__main__":
    with open("config/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    CROP_SIZE = config['dataset']['crop_size']
    STRATEGY = config['dataset']['strategy']
    
    # Dynamically setup directories and files based on the chosen strategy.
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path(f"data/processed_tensors_{STRATEGY}_{CROP_SIZE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_FILE = DATA_DIR / f"preprocessing_{STRATEGY}_{CROP_SIZE}.log"
    INVENTORY_FILE = DATA_DIR / f"processed_inventory_{STRATEGY}_{CROP_SIZE}.json"

    # Reset logging handlers so they point to the correct strategy file.
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Dynamically discover negative patients in the images folder.
    all_cases = [int(p.name) for p in IMAGES_ROOT.iterdir() if p.is_dir() and p.name.isdigit()]
    neg_list = sorted(list(set(all_cases) - set(pos_list)))

    # Tracking Dictionary.
    inventory = {
        "metadata": {"strategy": STRATEGY, "crop_size": CROP_SIZE},
        "clean_positives": [],
        "clean_negatives": [],
        "shifted_cases": [],  # Only populates during 'shift' strategy.
        "rejected_too_small_t2": [],
        "rejected_too_small_adc": [],
        "rejected_no_prostate": [],
        "rejected_clipped_tumor_strict": [],
        "rejected_unrecoverable_clipping": [],
        "skipped_missing_labels": [],
        "crashed_unknown_error": []
    }

    logging.info(f"--- Starting Unified Pipeline | Strategy: {STRATEGY.upper()} | Crop: {CROP_SIZE}x{CROP_SIZE} ---")
    logging.info(f"Found {len(pos_list)} POSITIVE and {len(neg_list)} NEGATIVE raw patients.\n")
    
    # Process every patient through the unified cookie-cutter.
    for cid in all_cases:
        is_pos = cid in pos_list
        extract_cookie_cutter(cid, is_pos, STRATEGY, CROP_SIZE, OUTPUT_DIR, inventory)
            
    # Save the raw inventory to JSON.
    with open(INVENTORY_FILE, "w") as f:
        json.dump(inventory, f, indent=4)
        
    logging.info(f"\nPipeline Complete ({STRATEGY.upper()}).")
    logging.info(f"Yielded {len(inventory['clean_positives'])} Clean Positives and {len(inventory['clean_negatives'])} Clean Negatives.")
    logging.info(f"Logs saved to {LOG_FILE}")
    logging.info(f"Inventory saved to {INVENTORY_FILE}")