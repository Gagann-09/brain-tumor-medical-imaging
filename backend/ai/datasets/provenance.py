import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Tracks the origin, transformations, and licensing of a dataset."""

    dataset_name: str = Field(..., description="Name of the dataset")
    source: str = Field(..., description="Source URL or institution")
    version: str = Field(..., description="Dataset version (e.g., v1.0.0)")
    license: str = Field(..., description="License under which the dataset is distributed")
    checksum: str = Field(..., description="SHA-256 checksum of the dataset source files")
    acquisition_metadata: dict[str, Any] = Field(default_factory=dict, description="Scanner/Sequence details if available")
    preprocessing_history: list[str] = Field(default_factory=list, description="List of preprocessing steps applied")
    annotation_version: str | None = Field(None, description="Version of the accompanying annotations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the record was created/updated")

class DatasetProvenanceManager:
    """Manages provenance tracking for medical datasets."""

    def __init__(self):
        # In a real system, this would interact with a database or a registry.
        self.records: dict[str, ProvenanceRecord] = {}

    def register_provenance(self, record: ProvenanceRecord) -> None:
        """Register a new provenance record."""
        self.records[record.dataset_name] = record

    def get_provenance(self, dataset_name: str) -> ProvenanceRecord | None:
        """Retrieve a provenance record by dataset name."""
        return self.records.get(dataset_name)

    @staticmethod
    def compute_checksum_from_dict(data: dict[str, Any]) -> str:
        """Compute a SHA-256 checksum from a dictionary representation of data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
