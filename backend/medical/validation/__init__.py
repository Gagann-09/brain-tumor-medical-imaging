from typing import List

from medical.domain import MRIImage
from medical.exceptions import MedicalImagingError


class ImageValidationError(MedicalImagingError):
    """Exception raised for validation errors on MRIImage objects."""
    pass

def validate_mri_image(image: MRIImage) -> bool:
    """
    Validate that an MRIImage has valid structure.
    """
    if not image.volumes:
        raise ImageValidationError("MRIImage contains no volume data.")

    shapes = [vol.shape for vol in image.volumes.values()]
    if not all(s == shapes[0] for s in shapes):
        raise ImageValidationError(f"Inconsistent volume shapes in MRIImage: {shapes}")

    # Check that it's 3D (or 2D/3D acceptable?)
    if len(shapes[0]) not in (2, 3):
        raise ImageValidationError(f"Volumes must be 2D or 3D, got shape {shapes[0]}")

    return True

def validate_multi_modal_consistency(images: list[MRIImage]) -> bool:
    """
    Validate that multiple MRIImage objects (e.g. from different series but same study)
    can be combined or compared. They should typically have the same shape/affine 
    if they are pre-registered, but for raw data this might not be true.
    """
    if not images:
        return True

    # Example logic: ensure patient IDs match if present
    patient_ids = []
    for img in images:
        if img.patient_metadata and img.patient_metadata.patient_id:
            patient_ids.append(img.patient_metadata.patient_id)

    if len(set(patient_ids)) > 1:
        raise ImageValidationError(f"Inconsistent Patient IDs across images: {set(patient_ids)}")

    return True
