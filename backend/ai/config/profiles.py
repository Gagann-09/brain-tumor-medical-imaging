from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.config import settings

class DatasetProfileName(str, Enum):
    DEVELOPMENT = "development"
    RESEARCH = "research"
    EXTERNAL_VALIDATION = "external_validation"

class DatasetProfile(BaseModel):
    """Schema for a dataset profile."""
    name: DatasetProfileName = Field(..., description="Internal name of the profile")
    dataset_name: str = Field(..., description="Human-readable name of the dataset")
    dataset_version: str = Field(..., description="Version of the dataset")
    expected_case_count: Optional[int] = Field(None, description="Expected number of cases in the dataset")
    expected_modalities: list[str] = Field(default_factory=list, description="Expected modalities")
    supported_formats: list[str] = Field(default_factory=lambda: [".nii", ".nii.gz"], description="Supported file formats")
    supported_tasks: list[str] = Field(default_factory=list, description="Supported tasks (e.g., segmentation, classification)")
    preprocessing_profile: str = Field(..., description="Identifier for the preprocessing pipeline to use")
    label_mapping_version: str = Field(..., description="Version of label mappings")
    allow_case_count_warning: bool = Field(False, description="If true, emit warning on case count mismatch instead of error")
    is_production_dataset: bool = Field(False, description="Whether this dataset is meant for final model training")
    data_dir: Optional[str] = Field(None, description="Path to the dataset directory")
    strict_validation: bool = Field(False, description="If true, enforce strict validation rules")

    @classmethod
    def load_from_manifest(cls, manifest: dict[str, Any], profile_name: DatasetProfileName, data_dir: str) -> "DatasetProfile":
        return cls(
            name=profile_name,
            dataset_name=manifest.get("dataset_name", "Unknown"),
            dataset_version=manifest.get("dataset_version", "1.0"),
            expected_case_count=manifest.get("expected_case_count"),
            expected_modalities=manifest.get("expected_modalities", []),
            supported_formats=manifest.get("supported_formats", [".nii", ".nii.gz"]),
            supported_tasks=manifest.get("supported_tasks", []),
            preprocessing_profile=manifest.get("preprocessing_profile", "default"),
            label_mapping_version=manifest.get("label_mapping_version", "v1"),
            allow_case_count_warning=manifest.get("allow_case_count_warning", False),
            is_production_dataset=manifest.get("is_production_dataset", False),
            strict_validation=manifest.get("strict_validation", False),
            data_dir=data_dir
        )

# Programmatic fallback profiles in case manifests are not provided or used directly
DEVELOPMENT_PROFILE = DatasetProfile(
    name=DatasetProfileName.DEVELOPMENT,
    dataset_name="BraTS 2020 (Dev Subset)",
    dataset_version="2020",
    expected_case_count=65,
    expected_modalities=["FLAIR", "T1", "T1ce", "T2", "SEG"],
    supported_formats=[".nii", ".nii.gz"],
    supported_tasks=["segmentation"],
    preprocessing_profile="brats_standard",
    label_mapping_version="v1",
    allow_case_count_warning=True,
    is_production_dataset=False,
    strict_validation=False,
    data_dir=settings.BRA_TS_DEV_PATH
)

RESEARCH_PROFILE = DatasetProfile(
    name=DatasetProfileName.RESEARCH,
    dataset_name="BraTS 2020 (Full)",
    dataset_version="2020",
    expected_case_count=None,  # Configurable later
    expected_modalities=["FLAIR", "T1", "T1ce", "T2", "SEG"],
    supported_formats=[".nii", ".nii.gz"],
    supported_tasks=["segmentation"],
    preprocessing_profile="brats_standard",
    label_mapping_version="v1",
    allow_case_count_warning=False,
    is_production_dataset=True,
    strict_validation=False, # configurable
    data_dir=settings.BRA_TS_FULL_PATH
)

KAGGLE_PROFILE = DatasetProfile(
    name=DatasetProfileName.EXTERNAL_VALIDATION,
    dataset_name="Kaggle Brain MRI",
    dataset_version="latest",
    expected_case_count=None,
    expected_modalities=["T1", "T1ce", "T2", "FLAIR"],
    supported_formats=[".jpg", ".png", ".nii", ".nii.gz"],
    supported_tasks=["classification", "external_validation"],
    preprocessing_profile="kaggle_standard",
    label_mapping_version="v1",
    allow_case_count_warning=True,
    is_production_dataset=False,
    strict_validation=False, # configurable
    data_dir=settings.KAGGLE_DATASET_PATH
)

PROFILE_REGISTRY = {
    DatasetProfileName.DEVELOPMENT: DEVELOPMENT_PROFILE,
    DatasetProfileName.RESEARCH: RESEARCH_PROFILE,
    DatasetProfileName.EXTERNAL_VALIDATION: KAGGLE_PROFILE
}

def get_profile(name: DatasetProfileName | str) -> DatasetProfile:
    if isinstance(name, str):
        name = DatasetProfileName(name)
    if name not in PROFILE_REGISTRY:
        raise ValueError(f"Unknown dataset profile: {name}")
    return PROFILE_REGISTRY[name]
