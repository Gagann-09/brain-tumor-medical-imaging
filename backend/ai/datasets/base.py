"""Base dataset abstraction."""
from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @abstractmethod
    def load(self) -> None:
        """Load the dataset into memory or initialize pointers."""
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get dataset level metadata."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of items in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item by index, returning a standardized dictionary."""
        pass

    @abstractmethod
    def preprocess(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply base preprocessing required for this specific dataset format."""
        pass
