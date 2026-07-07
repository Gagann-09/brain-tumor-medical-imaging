from pathlib import Path

import nibabel as nib
import numpy as np

from medical.domain import MRIImage
from medical.exceptions import CorruptedDataError


def load_nifti(filepath: str | Path, modality: str = "UNKNOWN") -> MRIImage:
    """
    Load a NIfTI file into an MRIImage object.

    Args:
        filepath: Path to the .nii or .nii.gz file.
        modality: The modality of the image (e.g., 'T1', 'T2', 'FLAIR').

    Returns:
        MRIImage object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    try:
        nii = nib.load(str(path))
        # get_fdata returns floating point data, typically needed for analysis
        volume = np.array(nii.get_fdata(dtype=np.float32))
        affine = nii.affine
        header = nii.header

        # Nifti header might contain voxel sizes (zooms)
        zooms = header.get_zooms()
        voxel_sizes = tuple(float(z) for z in zooms[:3]) if zooms else None

        volumes = {modality: volume}

        return MRIImage(
            volumes=volumes,
            affine=affine,
            voxel_sizes=voxel_sizes,
            custom_metadata={"nifti_header": header},
        )
    except Exception as e:
        raise CorruptedDataError(f"Failed to read NIfTI file {path}: {e!s}") from e


def save_nifti(image: MRIImage, filepath: str | Path, modality: str) -> None:
    """
    Save a specific modality from an MRIImage to a NIfTI file.
    """
    path = Path(filepath)
    volume = image.get_volume(modality)

    nii = nib.Nifti1Image(volume, image.affine)
    nib.save(nii, str(path))
