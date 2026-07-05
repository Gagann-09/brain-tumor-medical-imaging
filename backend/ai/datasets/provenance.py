import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Tracks the origin, transformations, and licensing of a dataset."""

    dataset_name: str = Field(..., description="Name of the dataset")
    source: str = Field(..., description="Source URL or institution")
    version: str = Field(..., description="Dataset version (e.g., v1.0.0)")
    license: str = Field(..., description="License under which the dataset is distributed")
    checksum: str = Field(..., description="SHA-256 checksum or fingerprint of the dataset source files")
    fingerprint_mode: str = Field("fast", description="Mode used to generate checksum (fast or full)")
    acquisition_metadata: dict[str, Any] = Field(default_factory=dict, description="Scanner/Sequence details if available")
    preprocessing_history: list[str] = Field(default_factory=list, description="List of preprocessing steps applied")
    annotation_version: str | None = Field(None, description="Version of the accompanying annotations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the record was created/updated")


class DatasetProvenanceManager:
    """Manages provenance tracking and fingerprinting for medical datasets."""

    def __init__(self):
        self.records: dict[str, ProvenanceRecord] = {}

    def register_provenance(self, record: ProvenanceRecord) -> None:
        """Register a new provenance record."""
        self.records[record.dataset_name] = record

    def get_provenance(self, dataset_name: str) -> ProvenanceRecord | None:
        """Retrieve a provenance record by dataset name."""
        return self.records.get(dataset_name)

    @staticmethod
    def generate_fingerprint(dataset_path: str, mode: str = "fast") -> str:
        """
        Generate a reproducible fingerprint for a dataset directory.
        
        Args:
            dataset_path: Path to the dataset.
            mode: "fast" (size + mtime + relpath) or "full" (SHA-256 content hash).
        """
        root_path = Path(dataset_path)
        if not root_path.exists():
            raise FileNotFoundError(f"Cannot generate fingerprint: path '{dataset_path}' does not exist.")

        # Collect all files, sort to ensure reproducible order
        all_files = sorted(
            [p for p in root_path.rglob("*") if p.is_file()],
            key=lambda p: str(p.relative_to(root_path))
        )
        
        hash_obj = hashlib.sha256()

        for filepath in all_files:
            rel_path = str(filepath.relative_to(root_path))
            stat = filepath.stat()
            
            if mode == "fast":
                # Fast mode: Hash relative path, size, and modification time
                file_signature = f"{rel_path}|{stat.st_size}|{stat.st_mtime}"
                hash_obj.update(file_signature.encode("utf-8"))
            elif mode == "full":
                # Full mode: Hash the actual file contents in chunks
                hash_obj.update(rel_path.encode("utf-8"))
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        hash_obj.update(chunk)
            else:
                raise ValueError(f"Unknown fingerprint mode: {mode}")

        return hash_obj.hexdigest()

    @staticmethod
    def compute_checksum_from_dict(data: dict[str, Any]) -> str:
        """Compute a SHA-256 checksum from a dictionary representation of data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
