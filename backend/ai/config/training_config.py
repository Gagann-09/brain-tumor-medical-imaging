from typing import Any

from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    device: str = Field("cuda", description="Primary device (e.g., 'cpu', 'cuda', 'mps', 'rocm')")
    device_ids: list[int] = Field(default_factory=lambda: [0], description="List of device IDs to use")
    mixed_precision: bool = Field(True, description="Enable mixed precision training")
    precision_type: str = Field("fp16", description="Type of mixed precision ('fp16', 'bf16')")
    num_workers: int = Field(4, description="Number of dataloader workers")
    pin_memory: bool = Field(True, description="Pin memory for dataloaders")

class OptimizerConfig(BaseModel):
    name: str = Field("AdamW", description="Name of the optimizer")
    learning_rate: float = Field(1e-4, description="Base learning rate")
    weight_decay: float = Field(1e-5, description="Weight decay factor")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional optimizer arguments")

class SchedulerConfig(BaseModel):
    name: str = Field("CosineAnnealingLR", description="Name of the scheduler")
    warmup_epochs: int = Field(5, description="Number of warmup epochs")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional scheduler arguments")

class LoggingConfig(BaseModel):
    log_dir: str = Field("./logs", description="Base directory for logs")
    experiment_name: str = Field("default_experiment", description="Name of the experiment")
    log_interval: int = Field(10, description="Log metrics every N steps")
    use_wandb: bool = Field(False, description="Enable Weights & Biases logging")
    use_mlflow: bool = Field(False, description="Enable MLflow logging")

class CheckpointConfig(BaseModel):
    save_dir: str = Field("./checkpoints", description="Directory to save checkpoints")
    save_frequency: int = Field(1, description="Save checkpoint every N epochs")
    keep_top_k: int = Field(3, description="Keep top K checkpoints based on monitor metric")
    monitor_metric: str = Field("val_loss", description="Metric to monitor for top K")
    mode: str = Field("min", description="Mode for monitor metric ('min' or 'max')")

class AugmentationConfig(BaseModel):
    train_transforms: list[dict[str, Any]] = Field(default_factory=list, description="Training transforms configuration")
    val_transforms: list[dict[str, Any]] = Field(default_factory=list, description="Validation transforms configuration")

class ModelConfig(BaseModel):
    name: str = Field(..., description="Name of the model architecture")
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels/classes")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Model-specific hyperparameters")

class DatasetConfig(BaseModel):
    registry_id: str = Field(default="brats20_dev_v1", description="Unique identifier for the dataset in the registry")
    batch_size: int = Field(16, description="Batch size per device")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Dataset-specific arguments")

class ExperimentConfig(BaseModel):
    """Root configuration combining all components for a training experiment."""
    seed: int = Field(42, description="Global random seed")
    max_epochs: int = Field(100, description="Maximum number of training epochs")

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
