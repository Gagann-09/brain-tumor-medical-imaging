from collections.abc import Callable
from typing import Any

import torch

from ai.classification.datasets.sample import ClassificationSample
from ai.segmentation.inference.roi_extractor import ROIStrategy


class ClassificationInputPipeline:
    """
    Configurable pipeline for preparing inputs for classification models.
    Supports ROI extraction, transformations, and normalizations independently.
    """

    def __init__(
        self,
        roi_strategy: ROIStrategy | None = None,
        transforms: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        normalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        # If None, implies "Whole-volume input"
        self.roi_strategy = roi_strategy
        self.transforms = transforms or []
        self.normalize_fn = normalize_fn

    def __call__(self, sample: ClassificationSample) -> dict[str, Any]:
        """
        Process a ClassificationSample.
        Returns a dictionary containing the processed tensor and original metadata/labels.
        """
        x = sample.image_tensor
        mask = sample.mask_tensor

        # 1. ROI Extraction Strategy
        if self.roi_strategy is not None and mask is not None:
            x, _ = self.roi_strategy.extract(x, mask)

        # 2. Transformations (Resize, Augmentations)
        for t in self.transforms:
            x = t(x)

        # 3. Normalization
        if self.normalize_fn:
            x = self.normalize_fn(x)

        return {
            "input": x,
            "labels": sample.labels,
            "sample_id": sample.sample_id,
            "metadata": sample.metadata,
        }
