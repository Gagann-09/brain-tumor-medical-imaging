import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from collections.abc import Callable
from typing import Any

from ai.training.data import BaseDataset
from medical.domain import MRIStudy


class PyTorchDatasetAdapter(Dataset):
    """
    Adapter bridging BaseDataset (containing MRIStudy objects) to PyTorch.
    It combines multi-modality volumes into a single 4D tensor (C, H, W, D)
    and applies MONAI/torchvision transforms.
    """

    def __init__(
        self,
        base_dataset: BaseDataset,
        required_modalities: list[str] | None = None,
        transforms: Callable | None = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is not installed.")

        self.base_dataset = base_dataset
        self.required_modalities = required_modalities or ["T1", "T1ce", "T2", "FLAIR"]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        study: MRIStudy = item["study"]

        # 1. Stack modalities into a single numpy array (C, H, W, D)
        # Assuming the validator has ensured these modalities exist and have the same shape
        volumes = []
        for mod in self.required_modalities:
            # Fallback to zero if missing but allowed by config, though ideally validator prevents this
            if mod in study.primary_image.volumes:
                volumes.append(study.primary_image.volumes[mod])
            else:
                volumes.append(np.zeros(study.primary_image.shape, dtype=np.float32))

        image_data = np.stack(volumes, axis=0).astype(np.float32)

        # 2. Extract masks if present
        segmentations = study.get_segmentations()
        mask_data = None
        if segmentations:
            # For simplicity, we take the first segmentation mask and expand to (1, H, W, D)
            mask_data = np.expand_dims(segmentations[0].mask, axis=0).astype(np.float32)

        # 3. Create MONAI-compatible dictionary
        data_dict = {"image": image_data}
        if mask_data is not None:
            data_dict["label"] = mask_data

        # 4. Apply transforms
        if self.transforms:
            data_dict = self.transforms(data_dict)

        # Ensure outputs are tensors
        if not isinstance(data_dict["image"], torch.Tensor):
            data_dict["image"] = torch.from_numpy(data_dict["image"])
        if "label" in data_dict and not isinstance(data_dict["label"], torch.Tensor):
            data_dict["label"] = torch.from_numpy(data_dict["label"])

        return data_dict
