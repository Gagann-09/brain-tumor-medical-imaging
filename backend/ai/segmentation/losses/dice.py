from typing import Any

from monai.losses import DiceLoss

from ai.training.providers import LossProvider


class DiceLossProvider(LossProvider):
    """Provides the MONAI DiceLoss function."""

    def __init__(self, to_onehot_y: bool = False, sigmoid: bool = True, **kwargs: Any):
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.kwargs = kwargs

    def get_loss(self) -> Any:
        return DiceLoss(to_onehot_y=self.to_onehot_y, sigmoid=self.sigmoid, **self.kwargs)
