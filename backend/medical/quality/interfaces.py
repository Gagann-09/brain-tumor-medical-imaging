from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIImage


class QualityMetric(ABC):
    """Abstract interface for image quality metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the quality metric."""
        pass

    @abstractmethod
    def calculate(self, image: MRIImage, modality: str, **kwargs: Any) -> float:
        """
        Calculate the metric for a specific modality of the image.
        """
        pass


class QualityAssessmentEngine:
    """
    Engine to run multiple quality metrics on an image and return a report.
    Designed to be extensible for future metrics beyond SNR and Contrast.
    """

    def __init__(self):
        self._metrics: dict[str, QualityMetric] = {}

    def register_metric(self, metric: QualityMetric) -> None:
        """Register a new quality metric."""
        self._metrics[metric.name] = metric

    def assess(self, image: MRIImage, modality: str, **kwargs: Any) -> dict[str, float]:
        """
        Run all registered metrics on the image for the specified modality.
        """
        results = {}
        for name, metric in self._metrics.items():
            results[name] = metric.calculate(image, modality, **kwargs)
        return results


# Basic Interfaces for future implementation
class SignalToNoiseRatio(QualityMetric):
    @property
    def name(self) -> str:
        return "SNR"

    def calculate(self, image: MRIImage, modality: str, **kwargs: Any) -> float:
        # Placeholder for actual SNR calculation
        return 0.0


class ContrastToNoiseRatio(QualityMetric):
    @property
    def name(self) -> str:
        return "CNR"

    def calculate(self, image: MRIImage, modality: str, **kwargs: Any) -> float:
        # Placeholder for actual CNR calculation
        return 0.0
