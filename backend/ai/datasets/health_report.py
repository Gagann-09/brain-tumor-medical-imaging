import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ai.config.profiles import DatasetProfile
from ai.datasets.provenance import DatasetProvenanceManager
from ai.datasets.validator import DatasetValidator, ValidationConfig
from ai.datasets.split_manager import SplitStrategy, ManifestSplitStrategy

if TYPE_CHECKING:
    from .base import DatasetAdapter


class DatasetHealthReporter:
    """Generates dataset health reports prior to training."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        dataset: "DatasetAdapter",
        profile: DatasetProfile,
        registry_entry: Any,  # Expected to be RegistryEntry
        split_strategy: SplitStrategy
    ) -> dict[str, Any]:
        """
        Generates and saves the health report.
        """
        # Validate dataset
        validator = DatasetValidator(ValidationConfig())
        try:
            val_stats = validator.validate_dataset(dataset, profile)
        except Exception as e:
             val_stats = {
                 "outcome": "Fail",
                 "errors": [str(e)],
                 "total_items": len(dataset),
                 "duplicates": 0,
                 "missing_modalities": 0,
                 "corrupted_files": 0
             }

        # Summarize splits if applicable
        split_summary = "N/A"
        if isinstance(split_strategy, ManifestSplitStrategy):
            manifest = split_strategy.manifest
            splits = manifest.get("splits", {})
            if split_strategy.fold:
                 splits = splits.get(split_strategy.fold, {})
            
            split_summary = {
                k: len(v) for k, v in splits.items()
            }
            
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "registry_id": registry_entry.dataset_identifier,
            "profile_name": profile.name,
            "dataset_name": registry_entry.dataset_name,
            "dataset_version": registry_entry.dataset_version,
            "dataset_fingerprint": registry_entry.dataset_fingerprint,
            "fingerprint_mode": registry_entry.fingerprint_mode,
            "study_count": val_stats.get("total_items", len(dataset)),
            "split_summary": split_summary,
            "duplicate_identifiers": val_stats.get("duplicates", 0),
            "missing_modalities": val_stats.get("missing_modalities", 0),
            "corrupted_files": val_stats.get("corrupted_files", 0),
            "validation_outcome": val_stats.get("outcome", "Unknown"),
            "errors": val_stats.get("errors", []),
            "warnings": val_stats.get("warnings", []),
            "class_distribution": val_stats.get("class_distribution", {})
        }

        self._save_report(report_data)
        return report_data

    def _save_report(self, report_data: dict[str, Any]) -> None:
        report_path = self.experiment_dir / "dataset_health_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Dataset Health Report\n\n")
            f.write(f"**Generated:** {report_data['timestamp']}\n")
            f.write(f"**Outcome:** {report_data['validation_outcome']}\n\n")
            
            f.write(f"## Dataset Profile\n")
            f.write(f"- Name: {report_data['dataset_name']}\n")
            f.write(f"- Version: {report_data['dataset_version']}\n")
            f.write(f"- Profile ID: {report_data['profile_name']}\n\n")
            
            f.write(f"## Provenance\n")
            f.write(f"- Fingerprint ({report_data['fingerprint_mode']}): `{report_data['dataset_fingerprint']}`\n\n")
            
            f.write(f"## Metrics\n")
            f.write(f"- Total Studies: {report_data['study_count']}\n")
            f.write(f"- Duplicates: {report_data['duplicate_identifiers']}\n")
            f.write(f"- Missing Modalities: {report_data['missing_modalities']}\n")
            f.write(f"- Corrupted Files: {report_data['corrupted_files']}\n\n")
            
            f.write(f"## Split Summary\n")
            f.write(f"```json\n{json.dumps(report_data['split_summary'], indent=2)}\n```\n\n")

            if report_data['class_distribution']:
                f.write(f"## Class Distribution\n")
                f.write(f"```json\n{json.dumps(report_data['class_distribution'], indent=2)}\n```\n\n")
                
            if report_data['errors']:
                f.write(f"## Errors\n")
                for e in report_data['errors']:
                    f.write(f"- {e}\n")
                f.write("\n")
                
            if report_data['warnings']:
                f.write(f"## Warnings\n")
                for w in report_data['warnings']:
                    f.write(f"- {w}\n")
                f.write("\n")
