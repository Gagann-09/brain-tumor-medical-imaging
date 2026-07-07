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
            import torch

            # Mock implementation for baseline training
            return {
                "inputs": torch.randn(4, 128, 128),
                "targets": torch.randint(0, 2, (3, 128, 128)).float(),
            }

    class DataLoaderManager:
        """Manages PyTorch DataLoader creation with deterministic behavior."""

        def __init__(self, profile: Any):
            self.profile = profile

        def get_dataloader(
            self, dataset: TorchDataset, batch_size: int, shuffle: bool = True
        ) -> TorchDataLoader:
            generator = torch.Generator()
            if getattr(self.profile, "name", "") in ["research", "publication"]:
                generator.manual_seed(42)

            def seed_worker(worker_id):
                import random

                import numpy as np

                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            return TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                worker_init_fn=seed_worker,
                generator=generator,
            )

except ImportError:
    pass
