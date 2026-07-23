import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.config.profiles import DatasetProfileName, get_profile
from ai.dataset_registry.registry import DatasetRegistry
from ai.datasets.brats_adapter import BraTSDataset
from ai.datasets.kaggle_adapter import KaggleDataset
from ai.datasets.validator import DatasetValidator, ValidationConfig
from core.config import get_settings


def register_datasets():
    from ai.dataset_registry.registry import DatasetRegistration

    brats_reg = DatasetRegistration(
        name="brats2020",
        adapter_class=BraTSDataset,
        supported_tasks=["segmentation"],
        dataset_version="2020",
        available_splits=["train", "val", "test"],
        transform_configuration={},
        metadata={"fingerprint": "brats2020_v1"},
    )

    kaggle_reg = DatasetRegistration(
        name="kaggle_mri",
        adapter_class=KaggleDataset,
        supported_tasks=["classification"],
        dataset_version="latest",
        available_splits=["train", "test"],
        transform_configuration={},
        metadata={"fingerprint": "kaggle_latest"},
    )

    if "brats2020" not in DatasetRegistry.list_datasets():
        DatasetRegistry.register("brats2020", brats_reg)
    if "kaggle_mri" not in DatasetRegistry.list_datasets():
        DatasetRegistry.register("kaggle_mri", kaggle_reg)


def run_validation():
    settings = get_settings()
    dataset_root = Path(settings.DATASET_ROOT)

    print("Initializing Dataset Registry...")
    register_datasets()

    print(f"\nDataset Root: {dataset_root}")
    print(f"Project Root: {settings.PROJECT_ROOT}\n")

    config = ValidationConfig(
        require_all_modalities=True,
        enforce_affine_consistency=True,
        enforce_spacing_consistency=True,
        require_patient_id=True,
        require_labels=True,
    )
    validator = DatasetValidator(config)

    profiles_to_validate = [
        ("brats2020", get_profile(DatasetProfileName.DEVELOPMENT), "BraTS Dataset"),
        ("kaggle_mri", get_profile(DatasetProfileName.EXTERNAL_VALIDATION), "Kaggle Dataset"),
    ]

    report = {"datasets": {}}

    # Print table header
    header = f"{'Dataset':<20}{'Resolved Path':<60}{'Exists':<10}{'Case Count':<14}{'Image Count':<14}{'Class Count':<14}{'Validation Status'}"
    print(header)
    print("-" * len(header))

    for dataset_name, profile, display_name in profiles_to_validate:
        reg = DatasetRegistry.get(dataset_name)

        try:
            adapter = reg.adapter_class(profile=profile)

            resolved_path = Path(adapter.root_dir).resolve()
            exists = "Yes" if resolved_path.exists() else "No"

            stats = validator.validate_dataset(adapter, profile)

            actual_fingerprint = reg.metadata.get("fingerprint")
            expected_fingerprint = getattr(profile, "expected_fingerprint", actual_fingerprint)

            stats["fingerprint_match"] = actual_fingerprint == expected_fingerprint
            if not stats["fingerprint_match"]:
                stats["errors"].append(
                    f"Fingerprint mismatch: expected {expected_fingerprint}, got {actual_fingerprint}"
                )
                stats["outcome"] = "Fail"

            case_count = stats.get("total_items", 0)
            # Image count: sum of all file paths across items
            image_count = sum(
                len(item.get("paths", []))
                for item in (adapter.items if hasattr(adapter, "items") else [])
            )
            class_count = len(stats.get("class_distribution", {}))
            validation_status = stats.get("outcome", "Unknown")

            print(f"{display_name:<20}{resolved_path!s:<60}{exists:<10}{case_count:<14}{image_count:<14}{class_count:<14}{validation_status}")

            report["datasets"][dataset_name] = stats

        except Exception as e:
            resolved_path = Path(profile.data_dir).resolve() if profile.data_dir else Path("N/A")
            exists = "No"
            print(f"{display_name:<20}{resolved_path!s:<60}{exists:<10}{'N/A':<14}{'N/A':<14}{'N/A':<14}Fail")
            report["datasets"][dataset_name] = {"outcome": "Fail", "error": str(e)}

    # Summary output matching expected format
    print("\n")
    for dataset_name, profile, display_name in profiles_to_validate:
        report["datasets"].get(dataset_name, {})
        resolved = Path(profile.data_dir).resolve() if profile.data_dir else Path("N/A")
        exists = "Yes" if resolved.exists() else "No"
        print(f"{display_name}")
        print("Resolved:")
        print(f"{resolved}")
        print("Exists:")
        print(f"{exists}")
        print()

    report_path = Path("dataset_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    run_validation()
