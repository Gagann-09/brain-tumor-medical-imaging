import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ai.datasets.provenance import DatasetProvenanceManager
from medical.exceptions import MedicalImagingError

logger = logging.getLogger(__name__)

class RegistryEntry(BaseModel):
    """Configuration and metadata for a registered dataset."""
    dataset_identifier: str = Field(..., description="Unique registry ID (e.g. brats20_dev_v1)")
    dataset_name: str = Field(..., description="Human-readable dataset name")
    dataset_version: str = Field(..., description="Version string")
    dataset_profile: str = Field(..., description="DatasetProfile identifier")
    dataset_path: str = Field(..., description="Path to the dataset directory")
    dataset_fingerprint: str = Field("", description="Pre-computed fingerprint")
    fingerprint_mode: str = Field("fast", description="Mode for fingerprinting (fast/full)")
    split_manifest: str = Field(..., description="Path to the JSON split manifest")
    preprocessing_profile: str = Field(..., description="Preprocessing profile to use")
    label_mapping_version: str = Field(..., description="Label mapping version")
    supported_tasks: list[str] = Field(..., description="List of supported tasks")
    execution_profile: str = Field(..., description="Execution profile (e.g. development, research)")
    created_at: str = Field(..., description="ISO timestamp of creation")
    last_verified: str = Field("", description="ISO timestamp of last verification")

class DatasetRegistry:
    """Central registry for all datasets. Loads lazily and manages fingerprints."""
    
    _instance = None
    
    def __new__(cls, registry_path: str = "backend/ai/datasets/registry.json"):
        if cls._instance is None:
            cls._instance = super(DatasetRegistry, cls).__new__(cls)
            cls._instance.registry_path = Path(registry_path)
            cls._instance._entries: dict[str, RegistryEntry] = {}
            cls._instance._loaded = False
        return cls._instance

    def _load_if_needed(self):
        """Lazily load registry from JSON on first access."""
        if self._loaded:
            return
            
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry file not found at {self.registry_path}")
            
        with open(self.registry_path, "r") as f:
            data = json.load(f)
            
        for d in data.get("datasets", []):
            entry = RegistryEntry(**d)
            self._entries[entry.dataset_identifier] = entry
            
        self._validate_registry()
        self._loaded = True

    def _validate_registry(self):
        """Validate the registry at startup."""
        identifiers = set()
        for entry in self._entries.values():
            if entry.dataset_identifier in identifiers:
                raise MedicalImagingError(f"Duplicate registry identifier found: {entry.dataset_identifier}")
            identifiers.add(entry.dataset_identifier)
            
            # Note: We won't strictly enforce dataset_path existence here to allow 
            # environments without the dataset downloaded to still parse the registry, 
            # but we can warn.
            if not Path(entry.dataset_path).exists():
                logger.warning(f"Dataset path {entry.dataset_path} for {entry.dataset_identifier} does not exist.")
            
            if not Path(entry.split_manifest).exists():
                logger.warning(f"Split manifest {entry.split_manifest} for {entry.dataset_identifier} does not exist.")
                
    def _save_registry(self):
        """Save the registry back to JSON (e.g., after updating fingerprints)."""
        data = {"datasets": [e.model_dump() for e in self._entries.values()]}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_entry(self, registry_id: str) -> RegistryEntry:
        """Retrieve an entry and compute its fingerprint if missing."""
        self._load_if_needed()
        if registry_id not in self._entries:
            raise ValueError(f"Unknown dataset registry identifier: {registry_id}")
            
        entry = self._entries[registry_id]
        
        # Hybrid Fingerprint Strategy
        if not entry.dataset_fingerprint:
            logger.info(f"Computing {entry.fingerprint_mode} fingerprint for {registry_id}...")
            if Path(entry.dataset_path).exists():
                fp = DatasetProvenanceManager.generate_fingerprint(
                    entry.dataset_path, mode=entry.fingerprint_mode
                )
                entry.dataset_fingerprint = fp
                entry.last_verified = datetime.utcnow().isoformat()
                self._save_registry()
            else:
                logger.warning(f"Cannot compute fingerprint, path {entry.dataset_path} is missing.")
                
        return entry
        
    def list_datasets(self) -> list[str]:
        self._load_if_needed()
        return list(self._entries.keys())
