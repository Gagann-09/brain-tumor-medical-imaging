from typing import Any

from torch.utils.data import Dataset

from .sample import ClassificationSample


class BaseClassificationDataset(Dataset):
    """
    Base Dataset for Classification tasks.
    Operates on ClassificationSample objects rather than raw tensors.
    """

    def __init__(self, samples: list[ClassificationSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClassificationSample:
        # Note: the input pipeline/transforms will operate on these samples
        return self.samples[idx]

    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {"num_samples": len(self.samples)}
