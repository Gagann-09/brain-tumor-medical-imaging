"""Training configuration."""

from typing import Any

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for AI model training."""

    batch_size: int = Field(default=32, ge=1, description="Number of samples per batch")
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    epochs: int = Field(default=100, ge=1, description="Total number of training epochs")
    optimizer_name: str = Field(default="adam", description="Name of the optimizer to use")
    optimizer_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional optimizer arguments"
    )
    loss_function: str = Field(default="cross_entropy", description="Primary loss function")
    early_stopping_patience: int | None = Field(
        default=10, description="Epochs to wait for improvement before stopping"
    )
    gradient_clip_val: float | None = Field(
        default=None, description="Gradient clipping maximum value"
    )
    mixed_precision: bool = Field(
        default=True, description="Use mixed precision training if available"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(default="cuda", description="Hardware device target (cuda/cpu)")
