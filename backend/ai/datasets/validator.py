from pydantic import BaseModel, Field

from medical.domain import MRIStudy
from medical.exceptions import MedicalImagingError


class DatasetValidationError(MedicalImagingError):
    """Raised when a dataset fails validation."""
    pass

class ValidationConfig(BaseModel):
    """Configures the strictness and rules of dataset validation."""

    require_all_modalities: bool = Field(True, description="If True, all required modalities must be present")
    required_modalities: list[str] = Field(default_factory=list, description="List of required modalities (e.g., ['T1', 'T1ce', 'T2', 'FLAIR'])")
    enforce_affine_consistency: bool = Field(True, description="Ensure affine matrices match within a certain tolerance")
    enforce_spacing_consistency: bool = Field(True, description="Ensure voxel spacing matches")
    enforce_orientation: bool = Field(False, description="Ensure images match a specific orientation (e.g. RAS)")
    require_patient_id: bool = Field(True, description="Ensure every study has a valid patient ID")
    require_labels: bool = Field(False, description="Ensure the study has at least one annotation")

class DatasetValidator:
    """Validates datasets against configurable rules."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_study(self, study: MRIStudy) -> bool:
        """
        Validate a single MRIStudy object.
        Raises DatasetValidationError if invalid.
        """
        if self.config.require_patient_id and not study.patient_id:
            raise DatasetValidationError(f"Study {study.study_id} is missing a patient ID.")

        if self.config.require_labels and not study.annotations:
            raise DatasetValidationError(f"Study {study.study_id} is missing required annotations.")

        image = study.primary_image

        # Modality Completeness
        if self.config.require_all_modalities and self.config.required_modalities:
            missing = [mod for mod in self.config.required_modalities if mod not in image.modalities]
            if missing:
                raise DatasetValidationError(f"Study {study.study_id} missing required modalities: {missing}")

        # Shape consistency is inherently validated by the MRIImage constructor,
        # but we can do extra checks if we had multiple images or labels.
        # Ensure segmentation masks match image shape
        for seg in study.get_segmentations():
            if seg.mask.shape != image.shape:
                raise DatasetValidationError(
                    f"Segmentation mask shape {seg.mask.shape} does not match image shape {image.shape} in study {study.study_id}."
                )

        # In a more advanced implementation, we would check:
        # - affine consistency against a reference or across images if multiple exist
        # - orientation checks (using nibabel orientation parsing)
        # - spacing consistency

        return True

    def validate_collection(self, studies: list[MRIStudy]) -> bool:
        """
        Validate a collection of studies and check cross-study consistency 
        (e.g. unique study IDs, metadata formatting).
        """
        study_ids = set()
        for study in studies:
            self.validate_study(study)
            if study.study_id in study_ids:
                raise DatasetValidationError(f"Duplicate study ID found: {study.study_id}")
            study_ids.add(study.study_id)

        return True
