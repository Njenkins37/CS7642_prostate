import argparse
import numpy as np
import json
import logging
from pathlib import Path
from scipy.ndimage import center_of_mass
from loader import load_picai_case, pos_list, IMAGES_ROOT, LABELS_ROOT

def setup_argparser():
    parser = argparse.ArgumentParser(description="PI-CAI Preprocessing Pipeline")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["strict", "shift"], 
        default="strict",
        help="How to handle tumors that are clipped by the bounding box. 'strict' drops them, 'shift' adjusts the box to save them."
    )
    parser.add_argument(
        "--crop_size", 
        type=int, 
        default=128,
        help="The height and width of the extracted tensor."
    )
    return parser.parse_args()

def get_crop_indices(center, size, max_dim, allow_shift=False):
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

def extract_cookie_cutter(case_id, args, output_dir, qc_results):
    try:
        data = load_picai_case(case_id, IMAGES_ROOT, LABELS_ROOT)
        t2, adc = data["t2"], data["adc"]
        gland, lesion = data["gland_t2"], data["lesion_t2"]

        # Defensive Checks.
        if t2.shape[0] < args.crop_size or t2.shape[1] < args.crop_size:
            logging.info(f"  [!] REJECTED {case_id}: T2 image too small.")
            qc_results["rejected_too_small_t2"].append(case_id)
            return
        if adc.shape[0] < args.crop_size or adc.shape[1] < args.crop_size:
            logging.info(f"  [!] REJECTED {case_id}: ADC image too small.")
            qc_results["rejected_too_small_adc"].append(case_id)
            return
        if np.sum(gland) == 0:
            logging.info(f"  [!] REJECTED {case_id}: No prostate gland mask.")
            qc_results["rejected_no_prostate"].append(case_id)
            return

        # Initial Crop (Centered on Prostate Gland, Bosma22b).
        x_c, y_c, _ = center_of_mass(gland)
        x_start, x_end = get_crop_indices(x_c, args.crop_size, t2.shape[0])
        y_start, y_end = get_crop_indices(y_c, args.crop_size, t2.shape[1])

        lesion_cropped = lesion[x_start:x_end, y_start:y_end, :]

        # When the tumor is clipped by the bounding box, "shift" strategy will recalculate the center on the tumor, not the gland.
        if np.sum(lesion_cropped) != np.sum(lesion):
            if args.strategy == "strict":
                logging.info(f"  [!] REJECTED {case_id}: Tumor clipped (Strict Mode).")
                qc_results["rejected_clipped_tumor_strict"].append(case_id)
                return
            
            elif args.strategy == "shift":
                logging.info(f"  [*] SHIFTING {case_id}: Tumor clipped. Recalculating center.")
                qc_results["shifted_cases"].append(case_id)
                
                # Recalculate center based on the tumor, not the gland.
                lx_c, ly_c, _ = center_of_mass(lesion)
                x_start, x_end = get_crop_indices(lx_c, args.crop_size, t2.shape[0])
                y_start, y_end = get_crop_indices(ly_c, args.crop_size, t2.shape[1])
                
                # Verify the shift actually saved the tumor.
                if np.sum(lesion[x_start:x_end, y_start:y_end, :]) != np.sum(lesion):
                    logging.info(f"  [!] REJECTED {case_id}: Tumor too massive to fit in {args.crop_size}x{args.crop_size}.")
                    qc_results["rejected_unrecoverable_clipping"].append(case_id)
                    return

        # Finalize and save.
        # Note that all the slides (Z-axis) are saved including the ones without cancer regions.
        np.savez_compressed(
            output_dir / f"{case_id}_clean.npz",
            t2=t2[x_start:x_end, y_start:y_end, :],
            adc=adc[x_start:x_end, y_start:y_end, :],
            lesion=lesion[x_start:x_end, y_start:y_end, :]
        )
        
        logging.info(f"  [SUCCESS] Saved tensors for {case_id}")
        qc_results["clean_cases"].append(case_id)

    except FileNotFoundError:
        logging.info(f"  [-] SKIPPED {case_id}: Missing human label.")
        qc_results["skipped_missing_labels"].append(case_id)
    except Exception as e:
        logging.error(f"  [!] CRASHED {case_id}: {e}")
        qc_results["crashed_unknown_error"].append(case_id)


if __name__ == "__main__":
    args = setup_argparser()
    
    # Dynamically setup directories and files based on the chosen strategy.
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path(f"data/processed_tensors_{args.strategy}_{args.crop_size}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOG_FILE = DATA_DIR / f"preprocessing_{args.strategy}_{args.crop_size}.log"
    JSON_FILE = DATA_DIR / f"dataset_splits_{args.strategy}_{args.crop_size}.json"

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

    # Tracking Dictionary.
    qc_results = {
        "metadata": {"strategy": args.strategy, "crop_size": args.crop_size},
        "clean_cases": [],
        "shifted_cases": [],  # Only populates during 'shift' strategy.
        "rejected_too_small_t2": [],
        "rejected_too_small_adc": [],
        "rejected_no_prostate": [],
        "rejected_clipped_tumor_strict": [],
        "rejected_unrecoverable_clipping": [],
        "skipped_missing_labels": [],
        "crashed_unknown_error": []
    }

    logging.info(f"--- Starting Pipeline | Strategy: {args.strategy.upper()} | Crop: {args.crop_size}x{args.crop_size} ---")
    
    for cid in pos_list:
        extract_cookie_cutter(cid, args, OUTPUT_DIR, qc_results)
            
    # Save the tracking lists to the strategy-specific JSON file.
    with open(JSON_FILE, "w") as f:
        json.dump(qc_results, f, indent=4)
        
    logging.info(f"\nPipeline Complete ({args.strategy.upper()}).")
    logging.info(f"Yielded {len(qc_results['clean_cases'])}/{len(pos_list)} clean cases.")
    logging.info(f"Logs saved to {LOG_FILE}")
    logging.info(f"Patient lists saved to {JSON_FILE}")