import numpy as np
import SimpleITK as sitk
from pathlib import Path

IMAGES_ROOT = Path("images")
LABELS_ROOT = Path("labels")

pos_list = [
    10005, 10008, 10012, 10013, 10019, 10021, 10029, 10032, 10036, 10040, 
    10043, 10044, 10048, 10050, 10053, 10059, 10074, 10078, 10079, 10085, 
    10094, 10097, 10100, 10103, 10104, 10106, 10107, 10110, 10112, 10117, 
    10118, 10121, 10132, 10134, 10135, 10140, 10148, 10157, 10164, 10168, 
    10170, 10171, 10182, 10202, 10209, 10211, 10212, 10214, 10216, 10220, 
    10225, 10229, 10232, 10233, 10234, 10235, 10236, 10241, 10252, 10257, 
    10262, 10268, 10271, 10273, 10274, 10280, 10282, 10283, 10289, 10293, 
    10294, 10304, 10312, 10318, 10322, 10324, 10329, 10334, 10338, 10339, 
    10340, 10342, 10350, 10354, 10355, 10358, 10365, 10368, 10369, 10371, 
    10372, 10375, 10376, 10377, 10380, 10383, 10390, 10392, 10393, 10397, 
    10398, 10399, 10400, 10402, 10405, 10408, 10418, 10424, 10426, 10431, 
    10433, 10434, 10442, 10451, 10456, 10458, 10459, 10463, 10464, 10465, 
    10471, 10473, 10481, 10482, 10484, 10486, 10490, 10494, 10496, 10497, 
    10498, 10499, 10501, 10503, 10508, 10510, 10515, 10517, 10519, 10520, 
    10522, 10523, 10526, 10529, 10537, 10539, 10540, 10545, 10547, 10548, 
    10549, 10550, 10554, 10555, 10558, 10560, 10562, 10565, 10567, 10568, 
    10570, 10577, 10584, 10589, 10604, 10605, 10606, 10607, 10611, 10620, 
    10622, 10626, 10630, 10631, 10634, 10635, 10636, 10637, 10638, 10641, 
    10644, 10651, 10652, 10658, 10660, 10661, 10665, 10668, 10670, 10679, 
    10682, 10684, 10686, 10687, 10688, 10690, 10693, 10699, 10700, 10701, 
    10707, 10710, 10717, 10718, 10721, 10726, 10728, 10730, 10739, 10742, 
    10743, 10753, 10754, 10758, 10760, 10763, 10768, 10772, 10773, 10775, 
    10777, 10781, 10797, 10798, 10799, 10804, 10806, 10807, 10808, 10810, 
    10811, 10825, 10827, 10833, 10834, 10836, 10837, 10839, 10842, 10845, 
    10852, 10856, 10857, 10862, 10865, 10867, 10868, 10872, 10873, 10875, 
    10880, 10882, 10883, 10885, 10888, 10889, 10890, 10894, 10895, 10900, 
    10901, 10903, 10904, 10909, 10910, 10911, 10915, 10920, 10921, 10925, 
    10932, 10938, 10942, 10943, 10948, 10953, 10956, 10957, 10961, 10964, 
    10965, 10968, 10970, 10975, 10976, 10985, 10991, 10993, 10995, 11002, 
    11009, 11032, 11037, 11039, 11041, 11043, 11045, 11049, 11050, 11051, 
    11054, 11055, 11063, 11067, 11068, 11072, 11074, 11076, 11080, 11081, 
    11083, 11086, 11093, 11095, 11102, 11111, 11114, 11115, 11117, 11122, 
    11123, 11130, 11133, 11137, 11143, 11146, 11149, 11152, 11154, 11155, 
    11157, 11162, 11163, 11165, 11168, 11169, 11173, 11174, 11177, 11179, 
    11181, 11185, 11186, 11188, 11190, 11194, 11198, 11207, 11208, 11225, 
    11229, 11231, 11236, 11239, 11240, 11243, 11245, 11247, 11248, 11253, 
    11256, 11257, 11258, 11260, 11266, 11269, 11274, 11278, 11280, 11283, 
    11284, 11285, 11296, 11299, 11300, 11302, 11305, 11306, 11313, 11322, 
    11325, 11330, 11332, 11333, 11334, 11336, 11338, 11341, 11345, 11346, 
    11350, 11352, 11357, 11361, 11367, 11369, 11373, 11375, 11377, 11382, 
    11384, 11385, 11397, 11400, 11423, 11425, 11426, 11428, 11437, 11441, 
    11442, 11444, 11446, 11447, 11448, 11450, 11451, 11456, 11462, 11465, 
    11468, 11471, 11472, 11475
]

# ======================================================
# IO helpers
# ======================================================

def load_mha(path):
    return sitk.ReadImage(str(path))

def load_nifti(path):
    return sitk.ReadImage(str(path))

def sitk_to_numpy(img):
    """
    SITK: (Z,Y,X) → NumPy: (X,Y,Z)
    """
    arr = sitk.GetArrayFromImage(img)
    return np.transpose(arr, (2, 1, 0))

# ======================================================
# Resampling
# ======================================================

def resample_to_reference(moving, reference, interpolator):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)

# ======================================================
# Intensity windowing 
# ======================================================

def robust_normalize(img, lower=1, upper=99):
    """
    Percentile-based windowing → [0,1]
    """
    lo, hi = np.percentile(img, [lower, upper])
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)

# ======================================================
# Case loader
# ======================================================

def load_picai_case(case_id, image_root, label_root):
    image_root = Path(image_root)
    label_root = Path(label_root)
    str_id = str(case_id)

    case_dir = image_root / str_id
    
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    mha_files = list(case_dir.glob("*.mha"))
    
    if not mha_files:
        raise FileNotFoundError(f"No MHA files found in {case_dir}")

    # Assumes filename format: caseID_studyID_modality.mha
    study_id = mha_files[0].stem.split("_")[1]

    t2_path  = case_dir / f"{str_id}_{study_id}_t2w.mha"
    adc_path = case_dir / f"{str_id}_{study_id}_adc.mha"

    lesion_path = label_root / "resampled" / f"{str_id}_{study_id}.nii.gz"
    gland_path  = label_root / "Bosma22b"  / f"{str_id}_{study_id}.nii.gz"
    zone_path   = label_root / "Yuan23"    / f"{str_id}_{study_id}.nii.gz"

    for p in [t2_path, adc_path, lesion_path, gland_path, zone_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load Raw Objects
    t2_sitk     = load_mha(t2_path)
    adc_sitk    = load_mha(adc_path)
    lesion_sitk = load_nifti(lesion_path)
    gland_sitk  = load_nifti(gland_path)
    zone_sitk   = load_nifti(zone_path)

    # -----------------------------------------------------------
    # Explicitly resample everything to T2 geometry
    # -----------------------------------------------------------
    # Even if 'lesion_sitk' is supposedly resampled,  run this to guarantee 
    # it matches the T2 voxel grid exactly (Origin/Direction/Spacing).
    
    # 1. Align Label to T2
    lesion_t2 = resample_to_reference(lesion_sitk, t2_sitk, sitk.sitkNearestNeighbor)
    gland_t2  = resample_to_reference(gland_sitk,  t2_sitk, sitk.sitkNearestNeighbor)
    zone_t2   = resample_to_reference(zone_sitk,   t2_sitk, sitk.sitkNearestNeighbor)

    # 2. Align Label to ADC
    # Resample the *original* lesion_sitk to ADC, not the T2 version, to avoid double-interpolation artifacts.
    lesion_adc = resample_to_reference(lesion_sitk, adc_sitk, sitk.sitkNearestNeighbor)
    gland_adc  = resample_to_reference(gland_sitk,  adc_sitk, sitk.sitkNearestNeighbor)
    zone_adc   = resample_to_reference(zone_sitk,   adc_sitk, sitk.sitkNearestNeighbor)

    # sitkLinear for continuous image data (ADC).
    adc_aligned_to_t2 = resample_to_reference(adc_sitk, t2_sitk, sitk.sitkLinear)

    return {
        "t2": robust_normalize(sitk_to_numpy(t2_sitk)),
        #"adc": robust_normalize(sitk_to_numpy(adc_sitk)),
        "adc": robust_normalize(sitk_to_numpy(adc_aligned_to_t2)),
        
        # Now these are guaranteed to be the same shape as "t2"
        "lesion_t2": sitk_to_numpy(lesion_t2),
        "gland_t2":  sitk_to_numpy(gland_t2),
        "zone_t2":   sitk_to_numpy(zone_t2),

        "lesion_adc": sitk_to_numpy(lesion_adc),
        "gland_adc":  sitk_to_numpy(gland_adc),
        "zone_adc":   sitk_to_numpy(zone_adc),
    }