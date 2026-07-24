import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))



def validate_full_brats_dataset(data_dir: str):
    patient_dirs = list(Path(data_dir).glob("BraTS20_*")) + list(Path(data_dir).glob("MICCAI_BraTS20_*"))
    if not patient_dirs:
        patient_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir() and not d.name.startswith(".")]

    total_patients = len(patient_dirs)
    patient_ids = []
    duplicate_ids = []
    missing_modalities = {}
    corrupted_files = []
    image_shapes = set()
    label_classes = set()
    total_files = 0
    valid_masks = 0

    required_mods = ["flair", "t1", "t1ce", "t2"]

    for p_dir in patient_dirs:
        pid = p_dir.name
        if pid in patient_ids:
            duplicate_ids.append(pid)
        else:
            patient_ids.append(pid)

        # Check modalities
        missing_mods_p = []
        for mod in required_mods:
            mod_files = list(p_dir.glob(f"*{mod}.nii*"))
            if not mod_files:
                missing_mods_p.append(mod)
            else:
                total_files += 1
                try:
                    nii = nib.load(str(mod_files[0]))
                    image_shapes.add(nii.shape)
                except Exception as e:
                    corrupted_files.append((str(mod_files[0]), str(e)))

        if missing_mods_p:
            missing_modalities[pid] = missing_mods_p

        # Check mask
        seg_files = list(p_dir.glob("*seg.nii*"))
        if seg_files:
            total_files += 1
            try:
                nii_seg = nib.load(str(seg_files[0]))
                data_seg = nii_seg.get_fdata()
                unique_vals = np.unique(data_seg).astype(int).tolist()
                for v in unique_vals:
                    label_classes.add(v)
                valid_masks += 1
            except Exception as e:
                corrupted_files.append((str(seg_files[0]), str(e)))

    train_count = int(0.8 * total_patients)
    val_count = total_patients - train_count

    report = {
        "dataset_name": "BraTS 2020",
        "dataset_path": str(Path(data_dir).resolve()),
        "total_patients": total_patients,
        "train_split": train_count,
        "val_split": val_count,
        "modalities": required_mods,
        "duplicate_patient_ids": duplicate_ids,
        "missing_modalities_count": len(missing_modalities),
        "corrupted_files_count": len(corrupted_files),
        "image_dimensions": [list(s) for s in image_shapes],
        "mask_labels_found": sorted(list(label_classes)),
        "total_files_verified": total_files,
        "valid_masks_verified": valid_masks,
        "status": "PASS" if len(corrupted_files) == 0 and len(missing_modalities) == 0 else "FAIL"
    }

    with open("dataset_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Dataset Validation Report JSON created:")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    validate_full_brats_dataset("datasets/brats2020_dev")
