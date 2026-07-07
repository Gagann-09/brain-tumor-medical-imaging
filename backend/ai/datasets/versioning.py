import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, Field

from .provenance import ProvenanceRecord


class DatasetManifest(BaseModel):
    """Manifest representing a specific version of a dataset."""

    dataset_name: str = Field(..., description="Name of the dataset")
    version: str = Field(..., description="Dataset version (e.g., v1.0.0)")
    supported_modalities: list[str] = Field(
        ..., description="Modalities supported by this dataset version"
    )
    checksum: str = Field(..., description="Checksum of the manifest contents or data snapshot")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of version creation"
    )
    provenance: ProvenanceRecord | None = Field(None, description="Provenance tracking information")
    number_of_studies: int = Field(0, description="Total number of studies in this dataset version")


class DatasetVersionManager:
    """Manages dataset version tags and tracks schema changes over time."""

    def __init__(self):
        # Maps dataset_name -> list of manifests
        self._manifests: dict[str, list[DatasetManifest]] = {}

    def generate_manifest(
        self,
        dataset_name: str,
        version: str,
        supported_modalities: list[str],
        number_of_studies: int,
        provenance: ProvenanceRecord | None = None,
    ) -> DatasetManifest:
        """Generate and store a new dataset manifest."""

        # Simple checksum based on the parameters
        data_dict = {
            "dataset_name": dataset_name,
            "version": version,
            "modalities": sorted(supported_modalities),
            "num_studies": number_of_studies,
        }
        checksum = hashlib.sha256(json.dumps(data_dict, sort_keys=True).encode("utf-8")).hexdigest()

        manifest = DatasetManifest(
            dataset_name=dataset_name,
            version=version,
            supported_modalities=supported_modalities,
            checksum=checksum,
            provenance=provenance,
            number_of_studies=number_of_studies,
        )

        if dataset_name not in self._manifests:
            self._manifests[dataset_name] = []

        self._manifests[dataset_name].append(manifest)
        return manifest

    def get_manifest(self, dataset_name: str, version: str) -> DatasetManifest | None:
        """Retrieve a specific dataset manifest by version."""
        if dataset_name not in self._manifests:
            return None

        for manifest in self._manifests[dataset_name]:
            if manifest.version == version:
                return manifest
        return None
