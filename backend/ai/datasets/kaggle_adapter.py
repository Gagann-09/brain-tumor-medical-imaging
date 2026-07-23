from pathlib import Path
from typing import Any

from ai.config.profiles import KAGGLE_PROFILE, DatasetProfile

from .base import DatasetAdapter


class KaggleDataset(DatasetAdapter):
    """Adapter for generic Kaggle medical imaging datasets."""

    def __init__(self, data_dir: str | None = None, profile: DatasetProfile | None = None):
        self.profile = profile or KAGGLE_PROFILE

        # Resolve dataset location: explicit data_dir > profile.data_dir
        self.root_dir = data_dir if data_dir else self.profile.data_dir

        if not self.root_dir or not Path(self.root_dir).exists():
            raise ValueError(
                f"Validation Error: Kaggle dataset path not found at '{self.root_dir}'.\n"
                f"Configured via profile '{self.profile.name}'.\n"
                "Please configure the dataset path correctly in the environment or configuration."
            )

        self.items = []
        self.load()

    def load(self) -> None:
        """Load Kaggle dataset index."""
        root_path = Path(self.root_dir)
        if not root_path.exists():
            return
            
        for class_dir in root_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for file_path in class_dir.iterdir():
                    if file_path.is_file():
                        self.items.append(
                            {
                                "class_label": class_name,
                                "paths": [str(file_path)]
                            }
                        )

    def get_metadata(self) -> dict[str, Any]:
        """Expose profile metadata through the adapter."""
        return self.profile.model_dump()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {}

    def preprocess(self, item: dict[str, Any]) -> dict[str, Any]:
        return item
