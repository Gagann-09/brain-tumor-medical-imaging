import abc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TrainingOutput:
    loss: torch.Tensor
    metrics: dict[str, float]
    # Useful for GANs (e.g. generator_loss, discriminator_loss)
    auxiliary_losses: dict[str, torch.Tensor] | None = None
    predictions: torch.Tensor | None = None


@dataclass
class ValidationOutput:
    loss: float
    metrics: dict[str, float]
    predictions: torch.Tensor | None = None


@dataclass
class PredictionResult:
    predictions: torch.Tensor
    uncertainty: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


class BaseModel(nn.Module, abc.ABC):
    """
    Strict contract for all AI models in the platform (Segmentation, GAN, etc.).
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def build(self) -> None:
        """Construct the model architecture (layers, modules)."""
        pass

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        pass

    @abc.abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> TrainingOutput:
        """
        Executes a single forward pass and loss computation for a batch.
        Must return a strongly typed TrainingOutput.
        Does NOT perform optimization (loss.backward() or optimizer.step()).
        """
        pass

    @abc.abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> ValidationOutput:
        """Executes a validation pass for a batch."""
        pass

    @abc.abstractmethod
    def prediction_step(self, batch: Any, batch_idx: int) -> PredictionResult:
        """Executes inference/prediction for a batch."""
        pass

    @abc.abstractmethod
    def compute_loss(self, preds: Any, targets: Any) -> torch.Tensor:
        """Computes the primary loss."""
        pass

    @abc.abstractmethod
    def compute_metrics(self, preds: Any, targets: Any) -> dict[str, float]:
        """Computes relevant metrics for the predictions."""
        pass

    @abc.abstractmethod
    def export(self, path: str) -> None:
        """Exports the model (e.g. TorchScript, ONNX)."""
        pass

    @abc.abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Loads weights from a checkpoint."""
        pass


class BaseSegmentationModel(BaseModel, abc.ABC):
    """Base class for all segmentation models."""

    pass


class BaseClassificationModel(BaseModel, abc.ABC):
    """Base class for all classification models."""

    pass


class BaseGAN(BaseModel, abc.ABC):
    """Base class for all GAN models, defining explicit steps for generator and discriminator."""

    @abc.abstractmethod
    def generator_step(self, batch: Any, batch_idx: int) -> TrainingOutput:
        """Executes a single forward pass and loss computation for the generator."""
        pass

    @abc.abstractmethod
    def discriminator_step(self, batch: Any, batch_idx: int) -> TrainingOutput:
        """Executes a single forward pass and loss computation for the discriminator."""
        pass

    # For GANs, the default training_step is often not used directly if the strategy
    # explicitly calls generator_step and discriminator_step. We provide a dummy implementation.
    def training_step(self, batch: Any, batch_idx: int) -> TrainingOutput:
        raise NotImplementedError(
            "BaseGAN uses explicit generator_step and discriminator_step. "
            "Use GANTrainingStrategy to orchestrate training."
        )
