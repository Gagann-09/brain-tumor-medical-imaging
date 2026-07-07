import pydicom

from medical.metadata.models import ImageMetadata, PatientMetadata, StudyMetadata


def parse_patient_metadata(dataset: pydicom.dataset.FileDataset) -> PatientMetadata:
    """Extract patient metadata from DICOM."""
    return PatientMetadata(
        patient_id=getattr(dataset, "PatientID", "UNKNOWN"),
        patient_name=str(dataset.PatientName) if "PatientName" in dataset else None,
        patient_birth_date=None,  # Needs parsing from DICOM date format (YYYYMMDD) if available
        patient_sex=getattr(dataset, "PatientSex", None),
        patient_age=getattr(dataset, "PatientAge", None),
        patient_weight=getattr(dataset, "PatientWeight", None),
    )


def parse_study_metadata(dataset: pydicom.dataset.FileDataset) -> StudyMetadata:
    """Extract study metadata from DICOM."""
    return StudyMetadata(
        study_instance_uid=getattr(dataset, "StudyInstanceUID", "UNKNOWN"),
        study_id=getattr(dataset, "StudyID", None),
        study_date=None,  # Needs parsing if available
        study_time=None,  # Needs parsing if available
        accession_number=getattr(dataset, "AccessionNumber", None),
        study_description=getattr(dataset, "StudyDescription", None),
        referring_physician=str(dataset.ReferringPhysicianName)
        if "ReferringPhysicianName" in dataset
        else None,
    )


def parse_image_metadata(dataset: pydicom.dataset.FileDataset) -> ImageMetadata:
    """Extract image metadata from DICOM."""
    pixel_spacing = None
    if "PixelSpacing" in dataset:
        pixel_spacing = [float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1])]

    return ImageMetadata(
        series_instance_uid=getattr(dataset, "SeriesInstanceUID", "UNKNOWN"),
        series_number=getattr(dataset, "SeriesNumber", None),
        modality=getattr(dataset, "Modality", "UNKNOWN"),
        series_description=getattr(dataset, "SeriesDescription", None),
        manufacturer=getattr(dataset, "Manufacturer", None),
        manufacturer_model_name=getattr(dataset, "ManufacturerModelName", None),
        magnetic_field_strength=getattr(dataset, "MagneticFieldStrength", None),
        spacing_between_slices=getattr(dataset, "SpacingBetweenSlices", None),
        slice_thickness=getattr(dataset, "SliceThickness", None),
        pixel_spacing=pixel_spacing,
    )
