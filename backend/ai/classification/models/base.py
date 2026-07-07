from abc import abstractmethod
from typing import Any

import torch

from ai.models.base import BaseModel


class BaseClassificationModel(BaseModel):
    """
    Base contract for all Classification models.
    Supports multi-class or binary outputs and includes an extension point for calibration.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from the input tensor.
        Returns:
            torch.Tensor: Logits of shape (B, num_classes) or (B, 1)
        """
        pass

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes probabilities from the input tensor.
        Applies Softmax for multi-class or Sigmoid for binary classification.
        Can be overridden by confidence calibrators.
        """
        logits = self.forward(x)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    @abstractmethod
    def get_calibration_module(self) -> Any:
        """
        Returns the calibration module (e.g. Temperature Scaling) if applied, else None.
        """
        pass
