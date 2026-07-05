import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.config.profiles import get_profile, DatasetProfileName
from ai.datasets.validator import DatasetValidator, ValidationConfig
from ai.dataset_registry.registry import DatasetRegistry
from ai.datasets.brats_adapter import BraTSDataset
from ai.datasets.kaggle_adapter import KaggleDataset

def register_datasets():
    from ai.dataset_registry.registry import DatasetRegistration
    
    brats_reg = DatasetRegistration(
        name="brats2020",
        adapter_class=BraTSDataset,
        supported_tasks=["segmentation"],
        dataset_version="2020",
        available_splits=["train", "val", "test"],
        transform_configuration={},
        metadata={"fingerprint": "brats2020_v1"}
    )
    
    kaggle_reg = DatasetRegistration(
        name="kaggle_mri",
        adapter_class=KaggleDataset,
        supported_tasks=["classification"],
        dataset_version="latest",
        available_splits=["train", "test"],
        transform_configuration={},
        metadata={"fingerprint": "kaggle_latest"}
    )
    
    if "brats2020" not in DatasetRegistry.list_datasets():
        DatasetRegistry.register("brats2020", brats_reg)
    if "kaggle_mri" not in DatasetRegistry.list_datasets():
        DatasetRegistry.register("kaggle_mri", kaggle_reg)

def run_validation():
    print("Initializing Dataset Registry...")
    register_datasets()

    config = ValidationConfig(
        require_all_modalities=True,
        enforce_affine_consistency=True,
        enforce_spacing_consistency=True,
        require_patient_id=True,
        require_labels=True
    )
    validator = DatasetValidator(config)

    profiles_to_validate = [
        ("brats2020", get_profile(DatasetProfileName.DEVELOPMENT)),
        ("kaggle_mri", get_profile(DatasetProfileName.EXTERNAL_VALIDATION))
    ]

    report = {"datasets": {}}

    for dataset_name, profile in profiles_to_validate:
        print(f"\nValidating dataset: {dataset_name} using profile: {profile.name}")
        reg = DatasetRegistry.get(dataset_name)
        
        try:
            adapter = reg.adapter_class(profile=profile)
            stats = validator.validate_dataset(adapter, profile)
            
            actual_fingerprint = reg.metadata.get("fingerprint")
            expected_fingerprint = getattr(profile, "expected_fingerprint", actual_fingerprint)
            
            stats["fingerprint_match"] = actual_fingerprint == expected_fingerprint
            if not stats["fingerprint_match"]:
                stats["errors"].append(f"Fingerprint mismatch: expected {expected_fingerprint}, got {actual_fingerprint}")
                stats["outcome"] = "Fail"

            print(f"Validation Outcome for {dataset_name}: {stats['outcome']}")
            report["datasets"][dataset_name] = stats
            
        except Exception as e:
            print(f"Validation failed with exception: {e}")
            report["datasets"][dataset_name] = {"outcome": "Fail", "error": str(e)}

    report_path = Path("dataset_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nValidation report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
