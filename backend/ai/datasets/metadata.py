"""Dataset metadata management."""

from typing import Any


class MetadataManager:
    """Manages parsing and standardization of clinical and image metadata."""

    def __init__(self, raw_metadata: dict[str, Any]):
        self.raw_metadata = raw_metadata
        self.standardized_metadata = self._standardize()

    def _standardize(self) -> dict[str, Any]:
        """Convert raw dataset-specific metadata into a standard schema."""
        # This will be extended by specific adapters or handle known schemas
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve standardized metadata value."""
        return self.standardized_metadata.get(key, default)

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Retrieve raw metadata value."""
        return self.raw_metadata.get(key, default)
