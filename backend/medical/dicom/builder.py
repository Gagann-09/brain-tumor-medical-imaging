import pydicom

from medical.dicom.conversion import extract_volume_and_affine
from medical.dicom.parsing import parse_image_metadata, parse_patient_metadata, parse_study_metadata
from medical.dicom.validation import validate_dicom_series
from medical.domain import MRIImage


def build_mri_image_from_dicom_series(datasets: list[pydicom.dataset.FileDataset]) -> MRIImage:
    """
    Construct an MRIImage from a single DICOM series.
    """
    validate_dicom_series(datasets)

    volume, affine = extract_volume_and_affine(datasets)

    first_ds = datasets[0]
    patient_meta = parse_patient_metadata(first_ds)
    study_meta = parse_study_metadata(first_ds)
    image_meta = parse_image_metadata(first_ds)

    modality = image_meta.modality

    # Map 'MR' to something more specific if possible, but keep it simple for now
    # Users can rename modalities later or we can add logic to infer T1/T2 from SeriesDescription
    modality_key = modality
    if hasattr(first_ds, 'SeriesDescription'):
        desc = first_ds.SeriesDescription.lower()
        if 't1' in desc and 'ce' in desc:
            modality_key = 'T1ce'
        elif 't1' in desc:
            modality_key = 'T1'
        elif 't2' in desc:
            modality_key = 'T2'
        elif 'flair' in desc:
            modality_key = 'FLAIR'

    volumes = {modality_key: volume}

    return MRIImage(
        volumes=volumes,
        affine=affine,
        patient_metadata=patient_meta,
        study_metadata=study_meta,
        image_metadata=image_meta
    )
