import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from medical.domain import MRIStudy
from medical.exceptions import MedicalImagingError
from ai.config.profiles import DatasetProfile

if TYPE_CHECKING:
    from .base import DatasetAdapter

logger = logging.getLogger(__name__)

class DatasetValidationError(MedicalImagingError):
    """Raised when a dataset fails validation."""
    pass

class ValidationConfig(BaseModel):
    """Configures the strictness and rules of dataset validation."""
    require_all_modalities: bool = Field(True, description="If True, all required modalities must be present")
    required_modalities: list[str] = Field(default_factory=list, description="List of required modalities")
    enforce_affine_consistency: bool = Field(True, description="Ensure affine matrices match within a certain tolerance")
    enforce_spacing_consistency: bool = Field(True, description="Ensure voxel spacing matches")
    enforce_orientation: bool = Field(False, description="Ensure images match a specific orientation (e.g. RAS)")
    require_patient_id: bool = Field(True, description="Ensure every study has a valid patient ID")
    require_labels: bool = Field(False, description="Ensure the study has at least one annotation")

class DatasetValidator:
    """Validates datasets against configurable rules and dataset profiles."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_study(self, study: MRIStudy) -> bool:
        """Validate a single MRIStudy object."""
        if self.config.require_patient_id and not study.patient_id:
            raise DatasetValidationError(f"Study {study.study_id} is missing a patient ID.")

        if self.config.require_labels and not study.annotations:
            raise DatasetValidationError(f"Study {study.study_id} is missing required annotations.")

        image = study.primary_image

        if self.config.require_all_modalities and self.config.required_modalities:
            missing = [mod for mod in self.config.required_modalities if mod not in image.modalities]
            if missing:
                raise DatasetValidationError(f"Study {study.study_id} missing required modalities: {missing}")

        for seg in study.get_segmentations():
            if seg.mask.shape != image.shape:
                raise DatasetValidationError(
                    f"Segmentation mask shape {seg.mask.shape} does not match image shape {image.shape} in study {study.study_id}."
                )

        return True

    def validate_collection(self, studies: list[MRIStudy]) -> bool:
        """Validate a collection of studies."""
        study_ids = set()
        for study in studies:
            self.validate_study(study)
            if study.study_id in study_ids:
                raise DatasetValidationError(f"Duplicate study ID found: {study.study_id}")
            study_ids.add(study.study_id)
        return True

    def validate_dataset(self, dataset: "DatasetAdapter", profile: DatasetProfile) -> dict[str, Any]:
        """
        Validates the dataset against the profile's expected criteria.
        Checks case counts, formats, readable files, folder structures.
        Returns a dictionary containing the validation outcome and statistics.
        """
        stats = {
            "total_items": len(dataset),
            "duplicates": 0,
            "missing_modalities": 0,
            "corrupted_files": 0,
            "warnings": [],
            "errors": [],
            "class_distribution": {}
        }

        # 1. Validate Expected Case Count
        expected_count = profile.expected_case_count
        if expected_count is not None and stats["total_items"] != expected_count:
            msg = f"Dataset case count mismatch: expected {expected_count}, found {stats['total_items']}."
            if profile.allow_case_count_warning and not profile.strict_validation:
                logger.warning(msg)
                stats["warnings"].append(msg)
            else:
                stats["errors"].append(msg)
                raise DatasetValidationError(msg)

        # 2. Validate Items Structure and Files
        study_ids = set()
        image_hashes = set() # For optional duplicate image detection

        for i in range(stats["total_items"]):
            try:
                item = dataset.items[i] if hasattr(dataset, 'items') else {}
                
                # Check for duplicate patient IDs
                patient_id = item.get("patient_id")
                if patient_id:
                    if patient_id in study_ids:
                        msg = f"Duplicate patient identifier found: {patient_id}"
                        stats["duplicates"] += 1
                        stats["warnings"].append(msg)
                        if profile.strict_validation:
                            raise DatasetValidationError(msg)
                    study_ids.add(patient_id)

                # BraTS specific logic: verify all required modalities belong to the same patient
                files = item.get("files", {})
                if files and "BraTS" in profile.dataset_name:
                    for expected_modality in profile.expected_modalities:
                        if expected_modality not in files:
                             msg = f"Patient {patient_id} missing expected modality: {expected_modality}"
                             stats["missing_modalities"] += 1
                             stats["warnings"].append(msg)
                             if profile.strict_validation:
                                 raise DatasetValidationError(msg)
                        else:
                            # Basic check to see if the filename contains the patient ID
                            file_name = Path(files[expected_modality]).name
                            if patient_id and patient_id not in file_name:
                                msg = f"Modality file {file_name} does not seem to belong to patient {patient_id}"
                                stats["warnings"].append(msg)
                                if profile.strict_validation:
                                    raise DatasetValidationError(msg)

                # Kaggle specific logic: Class distribution, empty classes, unreadable images
                class_label = item.get("class_label")
                if class_label is not None:
                    stats["class_distribution"][class_label] = stats["class_distribution"].get(class_label, 0) + 1
                
                paths = item.get("paths", [])
                for path_str in paths:
                    path = Path(path_str)
                    
                    # Check format
                    if not any(path.name.endswith(ext) for ext in profile.supported_formats):
                        msg = f"Unsupported file format for {path.name}. Supported: {profile.supported_formats}"
                        stats["warnings"].append(msg)
                        if profile.strict_validation:
                            raise DatasetValidationError(msg)
                            
                    # Unreadable / corrupted check (mocked by checking existence and size > 0)
                    if path.exists() and path.stat().st_size == 0:
                        msg = f"File {path.name} is empty/corrupted."
                        stats["corrupted_files"] += 1
                        stats["warnings"].append(msg)
                        if profile.strict_validation:
                             raise DatasetValidationError(msg)

            except Exception as e:
                msg = f"Error validating item {i}: {str(e)}"
                stats["errors"].append(msg)
                if profile.strict_validation:
                    raise DatasetValidationError(msg)
                logger.warning(msg)

        # Check for empty classes in Kaggle (if expected classes are known, would check here)
        # Assuming supported_tasks contains info or we just check if any class has 0
        if "Kaggle" in profile.dataset_name and len(stats["class_distribution"]) == 0:
             msg = "No classes found for Kaggle dataset. Class distribution is empty."
             stats["warnings"].append(msg)
             if profile.strict_validation:
                 raise DatasetValidationError(msg)

        stats["outcome"] = "Fail" if stats["errors"] else ("Warning" if stats["warnings"] else "Pass")
        return stats
