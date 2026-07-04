from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIImage


class BaseModel(ABC):
    """Abstract base class for all AI models."""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load model weights from a specified path."""
        pass

class SegmentationModel(BaseModel):
    """Abstract interface for segmentation models."""

    @abstractmethod
    def predict(self, image: MRIImage, **kwargs: Any) -> MRIImage:
        """
        Run inference on the input image and return a new MRIImage 
        containing the segmentation masks.
        """
        pass

class ClassificationModel(BaseModel):
    """Abstract interface for classification models."""

    @abstractmethod
    def predict(self, image: MRIImage, **kwargs: Any) -> dict[str, float]:
        """
        Run inference on the input image and return classification 
        probabilities or labels.
        """
        pass

class ExplainabilityEngine(ABC):
    """Abstract interface for explainability and interpretability methods."""

    @abstractmethod
    def explain(self, model: BaseModel, image: MRIImage, **kwargs: Any) -> MRIImage:
        """
        Generate explainability heatmaps (e.g., Grad-CAM, Integrated Gradients) 
        and return them as volumes within an MRIImage.
        """
        pass
