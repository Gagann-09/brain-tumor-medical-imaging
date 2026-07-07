import pydicom

from medical.exceptions import DICOMValidationError


def validate_dicom_file(dataset: pydicom.dataset.FileDataset) -> bool:
    """
    Validate that a loaded DICOM dataset contains required attributes.
    """
    required_tags = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality"]

    missing = [tag for tag in required_tags if tag not in dataset]
    if missing:
        raise DICOMValidationError(f"DICOM file is missing required tags: {missing}")

    return True


def validate_dicom_series(datasets: list[pydicom.dataset.FileDataset]) -> bool:
    """
    Validate that a list of DICOM datasets belong to the same series and are consistent.
    """
    if not datasets:
        raise DICOMValidationError("Empty list of DICOM datasets provided for validation.")

    series_uid = datasets[0].SeriesInstanceUID
    for ds in datasets:
        if ds.SeriesInstanceUID != series_uid:
            raise DICOMValidationError(
                "Multiple SeriesInstanceUIDs found in the provided series datasets."
            )
        validate_dicom_file(ds)

    return True
