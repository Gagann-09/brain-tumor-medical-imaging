from typing import Any

import torch
from monai.metrics import DiceMetric


class DiceMetricWrapper:
    """Wrapper for the MONAI DiceMetric."""

    def __init__(self, include_background: bool = False, reduction: str = "mean", **kwargs: Any):
        self.metric = DiceMetric(
            include_background=include_background, reduction=reduction, **kwargs
        )

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Computes the Dice metric for a batch and returns the aggregated mean."""
        self.metric(y_pred=preds, y=targets)
        val = self.metric.aggregate().item()
        self.metric.reset()
        return val
