from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ClassificationSample:
    """
    Domain object representing a single sample for classification.
    Can contain the raw volume, pre-extracted ROI tensor, metadata, and labels.
    """

    sample_id: str

    # Input Data
    image_tensor: torch.Tensor
    mask_tensor: torch.Tensor | None = None

    # Classification Labels
    labels: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
