"""Dataset split management."""


class DatasetSplitManager:
    """Handles train/val/test splits safely."""

    def __init__(self, dataset_size: int):
        self.dataset_size = dataset_size

    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[int]]:
        """Create random, non-overlapping splits."""
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        import random

        indices = list(range(self.dataset_size))
        random.seed(seed)
        random.shuffle(indices)

        train_end = int(train_ratio * self.dataset_size)
        val_end = train_end + int(val_ratio * self.dataset_size)

        return {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

    def load_predefined_splits(self, split_file_path: str) -> dict[str, list[int]]:
        """Load predefined splits from a configuration file."""
        # Placeholder for loading splits
        return {}
