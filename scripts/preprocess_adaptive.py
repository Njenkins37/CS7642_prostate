"""
Adaptive preprocessing for prostate MRI

  1. Load T2+ ADC volumes plus gland/lesion masks via `loader.load_picai_case`.
     Normalization/resampling done in loader
  2. Build 2D bounding box around prostate plus any lesion voxels (z projection)
     Pad with existing pixels, make square capped at max_crop, clamp to image boundaries.
  3. Resize the cropped T2 and ADC to a output_size^2 x Z. Output NPZ
  4. Logs under output_dir/logs/:
       - preprocessing.log   : general log
       - conversion_log.csv  : list of conversions and crops
       - dataset_splits.json : list potentially to be used in models later
Usage
---------------------------
  python preprocess_adaptive.py --output_size 128 --padding 16
  python preprocess_adaptive.py --positives_only --max_cases 50

  Optional flags:
    --output_size   final H=W in pixels (default 128)
    --padding       pixels of context around gland+lesion bbox (default 16)
    --max_crop      hard cap on pre-resize square edge (default 384)
    --positives_only  restrict to IDs in loader.pos_list
    --max_cases     cap iterations for smoke tests
    --images_root / --labels_root / --output_dir

Python / notebook:
    from preprocess_adaptive import run_preprocess
    summary = run_preprocess(max_cases=20, output_size=128, padding=16)
    summary["clean_cases"]              # IDs that produced a <case>_clean.npz
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from skimage.transform import resize

from utils.loader_adaptive import load_picai_case, pos_list


# Path structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_ROOT = PROJECT_ROOT / "images"
DEFAULT_LABELS_ROOT = PROJECT_ROOT / "labels"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocess_v2_adaptive"

# Log output
CSV_FIELDS = [
    "timestamp",
    "case_id",
    "label",
    "status",
    "t2_shape_original",
    "adc_shape_original",
    "crop_size",
    "output_size",
    "lesion_capture_pct",
    "reason"]


# Argument handling for CMD
def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive preprocessing")
    parser.add_argument("--output_size", type=int, default=128, help="Final H and W after resize.")
    parser.add_argument("--padding", type=int, default=16, help="Pad pixels gland/lesion bbox before squaring.")
    parser.add_argument("--max_crop", type=int, default=384, help="Max square crop size before resize. Prevents too large img")
    parser.add_argument("--positives_only", action="store_true", help="Process only cases listed in loader.pos_list (positives).")
    parser.add_argument("--max_cases", type=int, default=None, help="Cap on number of cases processed. Useful for smoke tests.")
    parser.add_argument("--images_root", type=str, default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--labels_root", type=str, default=str(DEFAULT_LABELS_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser


# Case discovery to include negative cases. Only pos_list captured so far
def _case_sort_key(case_id):
    case_id = str(case_id)
    return (0, int(case_id)) if case_id.isdigit() else (1, case_id)

def discover_case_ids(image_root: Path) -> list:
    """
    Return subdirs under image_root that contain .mha s
    """
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    case_ids = [
        path.name for path in image_root.iterdir()
        if path.is_dir() and any(path.glob("*.mha")) ]
    return sorted(case_ids, key=_case_sort_key)


# Bounding box + adaptive square crop
def get_mask_bbox(mask_3d: np.ndarray):
    """
    Compute the 2D bounding box across all slices for a binary mask.
    Project 3D mask to xy plane (max over Z slices)
    
    Return: (x_min, x_max, y_min, y_max) or None if mask is empty.
    """
    projection = np.max(mask_3d, axis=2)
    if projection.sum() == 0:
        return None
    rows = np.any(projection, axis=1)
    cols = np.any(projection, axis=0)
    
    # Get the outermost coordinates of any labels
    x_min, x_max = np.where(rows)[0][[0, -1]]
    y_min, y_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(x_max), int(y_min), int(y_max)


def compute_adaptive_crop(gland_bbox, lesion_bbox, padding, max_crop, image_shape):
    """
    Compute a square crop region that contains both gland and lesion (if present),
    with padding. Sticks to image bounds and caps at max_crop.
    
    Steps:
    1. COmbine of gland and lesion bounding boxes
    2. Add padding
    3. Make it square (based on the larger dimension)
    4. Cap at max_crop
    5. Center the square on the combined center
    6. Stick to image boundaries
    
    Return: (x_start, x_end, y_start, y_end, crop_size)
    """
    x_min, x_max, y_min, y_max = gland_bbox
    
    # Expand to include lesion if present: take highest dim based on lesion
    if lesion_bbox is not None:
        lx_min, lx_max, ly_min, ly_max = lesion_bbox
        x_min = min(x_min, lx_min)
        x_max = max(x_max, lx_max)
        y_min = min(y_min, ly_min)
        y_max = max(y_max, ly_max)

    # Add padding around the box, but keep within the image
    # This padding just adds extra existing pixels from the image
    x_min = max(0, x_min - padding)
    x_max = min(image_shape[0] - 1, x_max + padding) # fix for 0-based indices
    y_min = max(0, y_min - padding)
    y_max = min(image_shape[1] - 1, y_max + padding)

    # Make square using the larger dim (prostate+lesion). +1 for accurate count of width/height
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    # Cap at max_crop as limit
    crop_size = min(max(w, h), max_crop)

    # Center the square crop on the combined center
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    
    x_start = cx - crop_size // 2
    x_end = x_start + crop_size
    y_start = cy - crop_size // 2
    y_end = y_start + crop_size

    # Clamp to actual image boundaries: shift crop when out of boundary
    ## Prevent starts from being negative
    if x_start < 0:
        x_end -= x_start
        x_start = 0
    ## Prevent end from outside of boundary
    if x_end > image_shape[0]:
        x_start -= x_end - image_shape[0]
        x_end = image_shape[0]
        x_start = max(0, x_start)
        
    if y_start < 0:
        y_end -= y_start
        y_start = 0
        
    if y_end > image_shape[1]:
        y_start -= y_end - image_shape[1]
        y_end = image_shape[1]
        y_start = max(0, y_start)

    return x_start, x_end, y_start, y_end, crop_size

# Make every scan the same dims in 2D, preserve scan count
def resize_volume(volume: np.ndarray, target_hw: int, order: int) -> np.ndarray:
    """
    Resize a 3D volume (X, Y, Z) in the XY plane to target_hw x target_hw.
    Preserve Z (slice) dimension.

    order = controls interpolation (scipy/skimage spline orders):
      order=1 bilinear for continuous images (T2, ADC)
      order=3 cubic B-spline for continuous images (T2, ADC) -- matches nnU-Net.
      order=0 nearest neighbor for categorical masks (gland, lesion, zone)
              so integer label values survive intact.
    """
    x, y, z = volume.shape
    # No change if dims already correct
    if x == target_hw and y == target_hw:
        return volume

    output = np.zeros((target_hw, target_hw, z), dtype=volume.dtype)
    # Loop through each slice and resize separately
    for slice_idx in range(z):
        output[:, :, slice_idx] = resize(
            volume[:, :, slice_idx],
            (target_hw, target_hw),
            order=order,
            preserve_range=True,
            anti_aliasing=(order > 0),  # disable binary masks
        )
    return output


def crop_and_resize(volume, gland_mask, lesion_mask, zone_mask, output_size, padding, max_crop):
    """
    Return crop+resize based on gland and lesion mask. Resize the final crop to output_size
    """
    # Return none if there is no gland mask
    gland_bbox = get_mask_bbox(gland_mask)
    if gland_bbox is None:
        return None

    lesion_bbox = get_mask_bbox(lesion_mask)
    # Crop taking into consideration the lesion too. Pad within image bounds
    crop = compute_adaptive_crop(gland_bbox, lesion_bbox, padding, max_crop, volume.shape)
    x_start, x_end, y_start, y_end, _ = crop    # crop size not kept for this

    # Create correspondence between images. Same window
    ## Volume = (x,y,z) image grayscale
    volume_crop = volume[x_start:x_end, y_start:y_end, :]
    gland_crop  = gland_mask[x_start:x_end, y_start:y_end, :]
    lesion_crop = lesion_mask[x_start:x_end, y_start:y_end, :]
    zone_crop   = zone_mask[x_start:x_end, y_start:y_end, :]

    return (
        # Resize the crop to output size, bilinear, nearest neighbor
        # These can be improved by higher order. Probably better to use 3rd and 1st orders (labels) across all
        resize_volume(volume_crop, output_size, order=3), # consider a 3rd order spline from papers
        resize_volume(gland_crop,  output_size, order=0), # consider a 1st order linear
        resize_volume(lesion_crop, output_size, order=0),
        resize_volume(zone_crop,   output_size, order=0),
        crop    )





# The code below is for processing cases based on above fxns, logging, CSV streaming, and QA.
# ------------------------------------------------------------------------------------------------------------------------

# Per-case processing + CSV streaming
@dataclass
# Changeable params
class RunState:
    args: argparse.Namespace
    output_dir: Path
    results: dict
    csv_writer: object
    csv_handle: object


# Add to csv
def _log_row(state: RunState, **fields) -> None:
    row = {k: fields.get(k, "") for k in CSV_FIELDS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    state.csv_writer.writerow(row)
    state.csv_handle.flush()


def process_case(case_id: str, state: RunState) -> None:
    args = state.args
    base_row = {
        "case_id": case_id,
        "output_size": args.output_size,
        "label": "unknown",    }

    try:
        data = load_picai_case(case_id, args.images_root, args.labels_root)
    except FileNotFoundError as exc:
        logging.info(f"  [-] SKIPPED {case_id}: Missing file ({exc}).")
        _log_row(state, **base_row, status="skipped_missing", reason=str(exc))
        return
    except Exception as exc:
        logging.error(f"  CRASHED {case_id}: {exc}")
        _log_row(state, **base_row, status="crashed", reason=str(exc))
        return

    # Label cases based on scans
    has_lesion = bool(data["lesion_t2"].sum() > 0)
    label = "positive" if has_lesion else "negative"

    t2_shape = data["t2"].shape
    adc_shape = data["adc"].shape
    base_row.update({
        "label": label,
        "t2_shape_original": "x".join(map(str, t2_shape)),
        "adc_shape_original": "x".join(map(str, adc_shape)),
    })

    # Reject if T2 volume too small
    min_dim = min(t2_shape[0], t2_shape[1])
    if min_dim < args.output_size // 2:
        logging.info(f"  REJECTED {case_id}: T2 too small ({t2_shape})")
        _log_row(state, **base_row, status="rejected_too_small",
                 reason=f"min_dim={min_dim} < {args.output_size // 2}")
        return

    t2_branch = crop_and_resize(
        data["t2"], data["gland_t2"], data["lesion_t2"], data["zone_t2"],
        output_size=args.output_size,
        padding=args.padding,
        max_crop=args.max_crop,
    )
    if t2_branch is None:
        logging.info(f"  REJECTED {case_id}: No prostate gland mask on T2.")
        _log_row(state, **base_row, status="rejected_no_prostate",
                 reason="gland_t2 bbox empty")
        return

    t2_resized, gland_t2_resized, lesion_t2_resized, zone_t2_resized, t2_crop = t2_branch
    x_start, x_end, y_start, y_end, crop_size = t2_crop

    # Log any trimmed lesions
    status = "complete"
    lesion_capture_pct = ""
    if has_lesion:
        total = int(data["lesion_t2"].sum())
        captured = int(data["lesion_t2"][x_start:x_end, y_start:y_end, :].sum())
        lesion_capture_pct = f"{captured / total:.3f}"
        if captured < total * 0.99:
            logging.warning(
                f"  PARTIAL {case_id}: Captured {captured / total:.1%} "
                f"of lesion voxels (crop_size={crop_size})"
            )
            status = "partial_lesion"

    # ADC crop resize
    adc_branch = crop_and_resize(
        data["adc"], data["gland_adc"], data["lesion_adc"], data["zone_adc"],
        output_size=args.output_size,
        padding=args.padding,
        # padding=max(1, args.padding // 3),
        max_crop=args.max_crop,
    )
    if adc_branch is None:
        adc_resized        = resize_volume(data["adc"],        args.output_size, order=3)
        gland_adc_resized  = resize_volume(data["gland_adc"],  args.output_size, order=0)
        lesion_adc_resized = resize_volume(data["lesion_adc"], args.output_size, order=0)
        zone_adc_resized   = resize_volume(data["zone_adc"],   args.output_size, order=0)
    else:
        adc_resized, gland_adc_resized, lesion_adc_resized, zone_adc_resized, _ = adc_branch

    # Save
    np.savez_compressed(
        state.output_dir / f"{case_id}_clean.npz",
        t2=t2_resized.astype(np.float32),
        lesion_t2=lesion_t2_resized.astype(np.uint8),
        gland_t2=gland_t2_resized.astype(np.uint8),
        zone_t2=zone_t2_resized.astype(np.uint8),
        adc=adc_resized.astype(np.float32),
        lesion_adc=lesion_adc_resized.astype(np.uint8),
        gland_adc=gland_adc_resized.astype(np.uint8),
        zone_adc=zone_adc_resized.astype(np.uint8),
        t2_crop=np.array([x_start, x_end, y_start, y_end, int(crop_size)], dtype=np.int32),
        t2_original_shape=np.array(t2_shape, dtype=np.int32),
    )

    logging.info(
        f"  [OK] {case_id}: {label} crop={crop_size}->{args.output_size}"
        + (f", lesion={lesion_capture_pct}" if has_lesion else "")
    )
    state.results["clean_cases"].append(case_id)
    (state.results["clean_positive_cases"] if has_lesion
     else state.results["clean_negative_cases"]).append(case_id)

    _log_row(state, **base_row, status=status, crop_size=int(crop_size),
             lesion_capture_pct=lesion_capture_pct)


# Helper fxns from notebook runs
def _build_namespace(**kwargs) -> argparse.Namespace:
    parser = setup_argparser()
    args = parser.parse_args([])  # all defaults
    for key, value in kwargs.items():
        if value is not None:
            setattr(args, key, value)
    # Resolve root paths to absolute Path objects.
    args.images_root = Path(args.images_root).resolve()
    args.labels_root = Path(args.labels_root).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    return args

# Summary dict
def run_preprocess(
    output_size: int = 128,
    padding: int = 16,
    max_crop: int = 384,
    positives_only: bool = False,
    max_cases: Optional[int] = None,
    images_root: Optional[Path] = None,
    labels_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
                    ) -> dict:

    args = _build_namespace(
        output_size=output_size, padding=padding, max_crop=max_crop,
        positives_only=positives_only, max_cases=max_cases,
        images_root=str(images_root) if images_root else None,
        labels_root=str(labels_root) if labels_root else None,
        output_dir=str(output_dir) if output_dir else None,
    )

    # Iutput NPZ and logs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "preprocessing.log"
    csv_path = logs_dir / "conversion_log.csv"
    json_path = logs_dir / "dataset_splits.json"

    # Handler reset to prevent dups when called from an NB
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )

    # Use case in models/TBD
    results = {
        "clean_cases": [],
        "clean_positive_cases": [],
        "clean_negative_cases": [],    }

    # Case select
    case_ids = discover_case_ids(args.images_root)
    if args.positives_only:
        pos_set = {str(cid) for cid in pos_list}
        case_ids = [cid for cid in case_ids if cid in pos_set]
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]

    logging.info(
        f"--- Adaptive Pipeline v2 | Output: {args.output_size}x{args.output_size} "
        f"| Pad: {args.padding} | Cases: {len(case_ids)} "
        f"| {'positives-only' if args.positives_only else 'all cases'} ---"
    )

    # CSV handling to be kelpt open
    with open(csv_path, "w", newline="") as csv_handle:
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_handle.flush()

        state = RunState(
            args=args,
            output_dir=args.output_dir,
            results=results,
            csv_writer=csv_writer,
            csv_handle=csv_handle,
        )

        for case_id in case_ids:
            process_case(case_id, state)

    # Maybe get rid of the json if not needed in the models. Kept for now
    with open(json_path, "w") as handle:
        json.dump({"clean_cases": results["clean_cases"]}, handle, indent=2)

    n_clean = len(results["clean_cases"])
    n_total = len(case_ids)
    ratio = (n_clean / n_total) if n_total else 0.0
    logging.info(f"\nDone. {n_clean}/{n_total} cases processed ({ratio:.1%}).")
    logging.info(f"  positives: {len(results['clean_positive_cases'])}")
    logging.info(f"  negatives: {len(results['clean_negative_cases'])}")
    logging.info(f"Logs: {logs_dir}")

    # Release the log file after write. Other can't mod in windows
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    return results


def main() -> None:
    args = setup_argparser().parse_args()
    run_preprocess(
        output_size=args.output_size,
        padding=args.padding,
        max_crop=args.max_crop,
        positives_only=args.positives_only,
        max_cases=args.max_cases,
        images_root=Path(args.images_root),
        labels_root=Path(args.labels_root),
        output_dir=Path(args.output_dir),
    )

if __name__ == "__main__":
    main()
