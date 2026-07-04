"""
Converters package for converting between different formats and structures.
"""
from typing import List

import pydicom

from medical.dicom.builder import build_mri_image_from_dicom_series
from medical.domain import MRIImage
from medical.exceptions import MedicalImagingError


def dicom_series_to_mri_image(datasets: list[pydicom.dataset.FileDataset]) -> MRIImage:
    """
    Convert a list of DICOM datasets representing a single series into an MRIImage.
    """
    try:
        return build_mri_image_from_dicom_series(datasets)
    except Exception as e:
        raise MedicalImagingError(f"Failed to convert DICOM series to MRIImage: {e!s}")
