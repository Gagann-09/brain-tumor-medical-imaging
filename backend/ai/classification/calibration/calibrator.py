from abc import ABC, abstractmethod

import torch


class BaseCalibrator(ABC):
    """
    Base class for Confidence Calibration strategies (e.g. Temperature Scaling).
    These operate on the raw logits output by the classifier.
    """

    @abstractmethod
    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """Fit the calibrator on a validation set."""
        pass

    @abstractmethod
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to raw logits to produce calibrated probabilities."""
        pass


class TemperatureScaler(BaseCalibrator):
    """
    Standard Temperature Scaling for confidence calibration.
    """

    def __init__(self):
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        # In a full implementation, we'd run LBFGS to optimize self.temperature
        # using Negative Log Likelihood loss.
        pass

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return torch.softmax(logits / temperature, dim=1)
