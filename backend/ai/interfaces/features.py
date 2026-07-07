import abc
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FeatureMap:
    """Represents an intermediate feature map and its associated metadata."""

    tensor: torch.Tensor
    metadata: dict[str, Any]


class FeatureProvider(abc.ABC):
    """
    Interface for models that can provide intermediate feature maps
    for downstream tasks like Grad-CAM, SHAP, and saliency maps.
    """

    @abc.abstractmethod
    def get_intermediate_features(self) -> dict[str, FeatureMap]:
        """
        Returns a dictionary mapping layer names to their corresponding FeatureMap
        from the most recent forward pass.
        """
        pass
