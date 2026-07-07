"""Inference configuration."""

from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Configuration for AI model inference."""

    batch_size: int = Field(default=1, ge=1, description="Number of samples per inference batch")
    device: str = Field(default="cpu", description="Hardware device target (cuda/cpu)")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence for positive prediction"
    )
    mixed_precision: bool = Field(
        default=False, description="Use mixed precision inference if available"
    )
    num_workers: int = Field(default=0, ge=0, description="Number of data loading workers")
    return_attention_maps: bool = Field(
        default=False, description="Whether to return attention/saliency maps"
    )
    tta_enabled: bool = Field(default=False, description="Enable Test-Time Augmentation")
    tta_steps: int = Field(default=4, description="Number of TTA steps if enabled")
