import os

import nibabel as nib
import numpy as np


def create_tiny_nifti(filepath):
    data = np.random.rand(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filepath)

def main():
    base_dir = "dummy_brats"
    os.makedirs(base_dir, exist_ok=True)

    # We create two mock studies to allow train/val splitting
    for i in range(2):
        study_id = f"BraTS20_Training_{i:03d}"
        study_dir = os.path.join(base_dir, study_id)
        os.makedirs(study_dir, exist_ok=True)

        create_tiny_nifti(os.path.join(study_dir, f"{study_id}_t1.nii.gz"))
        create_tiny_nifti(os.path.join(study_dir, f"{study_id}_t1ce.nii.gz"))
        create_tiny_nifti(os.path.join(study_dir, f"{study_id}_t2.nii.gz"))
        create_tiny_nifti(os.path.join(study_dir, f"{study_id}_flair.nii.gz"))

        # Create segmentation mask (integers)
        mask_data = np.random.randint(0, 4, size=(32, 32, 32)).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        nib.save(mask_img, os.path.join(study_dir, f"{study_id}_seg.nii.gz"))

    print(f"Minimal BraTS dataset created at {base_dir}")

if __name__ == "__main__":
    main()
