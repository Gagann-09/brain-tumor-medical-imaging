"""Dataset registry implementation."""
from typing import Any

from pydantic import BaseModel


class DatasetRegistration(BaseModel):
    """Metadata for a registered dataset."""
    name: str
    adapter_class: Any
    supported_tasks: list[str]
    dataset_version: str
    available_splits: list[str]
    transform_configuration: dict[str, Any]
    metadata: dict[str, Any]

class DatasetRegistry:
    """Central registry for all AI datasets."""

    _registry: dict[str, DatasetRegistration] = {}

    @classmethod
    def register(cls, name: str, registration: DatasetRegistration) -> None:
        """Register a dataset adapter."""
        if name in cls._registry:
            raise ValueError(f"Dataset {name} is already registered.")
        cls._registry[name] = registration

    @classmethod
    def get(cls, name: str) -> DatasetRegistration:
        """Retrieve a dataset registration by name."""
        if name not in cls._registry:
            raise KeyError(f"Dataset {name} not found in registry.")
        return cls._registry[name]

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all registered dataset names."""
        return list(cls._registry.keys())
