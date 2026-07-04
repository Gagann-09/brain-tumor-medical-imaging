from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIStudy


class BaseDataset(ABC):
    """Abstract dataset interface for the AI platform."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

class TrainingDataset(BaseDataset):
    """Dataset optimized for training with augmentations."""

    def __init__(self, studies: list[MRIStudy], transforms: Any = None):
        self.studies = studies
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Any:
        study = self.studies[idx]
        # Base implementation, adapters will handle PyTorch conversion
        return {"study": study, "transforms": self.transforms}

class ValidationDataset(BaseDataset):
    """Dataset optimized for validation (no random augmentations)."""

    def __init__(self, studies: list[MRIStudy], transforms: Any = None):
        self.studies = studies
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Any:
        study = self.studies[idx]
        return {"study": study, "transforms": self.transforms}

class InferenceDataset(BaseDataset):
    """Dataset optimized for inference/prediction."""

    def __init__(self, studies: list[MRIStudy], transforms: Any = None):
        self.studies = studies
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Any:
        study = self.studies[idx]
        return {"study": study, "transforms": self.transforms}

class DatasetBuilder(ABC):
    """Builds appropriate dataset objects for different phases."""

    @abstractmethod
    def build_train_dataset(self) -> TrainingDataset:
        pass

    @abstractmethod
    def build_val_dataset(self) -> ValidationDataset:
        pass

    @abstractmethod
    def build_inference_dataset(self) -> InferenceDataset:
        pass

# PyTorch Adapter Interfaces (to be fully realized when connecting models)
try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset

    class PyTorchDatasetAdapter(TorchDataset):
        """Bridges BaseDataset to PyTorch Dataset."""
        def __init__(self, base_dataset: BaseDataset):
            self.base_dataset = base_dataset

        def __len__(self) -> int:
            return len(self.base_dataset)

        def __getitem__(self, idx: int) -> Any:
            # Here, the dictionary would be converted to torch tensors
            # and MONAI transforms applied in a concrete implementation.
            item = self.base_dataset[idx]
            return item

except ImportError:
    pass
