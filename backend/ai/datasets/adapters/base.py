from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from medical.domain import MRIStudy


class BaseDatasetAdapter(ABC):
    """
    Abstract adapter for loading external medical datasets into standard MRIStudy domain objects.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

    @abstractmethod
    def load_studies(self) -> Iterator[MRIStudy]:
        """
        Lazily iterate over all studies in the dataset and yield MRIStudy objects.
        """
        pass

    @abstractmethod
    def load_study(self, study_id: str) -> MRIStudy:
        """
        Load a specific study by ID.
        """
        pass
