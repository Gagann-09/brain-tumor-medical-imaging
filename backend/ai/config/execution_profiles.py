from enum import StrEnum

from pydantic import BaseModel, Field


class ProfilerLevel(StrEnum):
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"


class LoggingBehavior(StrEnum):
    LOCAL = "local"
    WANDB_FULL = "wandb_full"
    MLFLOW = "mlflow"


class ExecutionProfile(BaseModel):
    """Configuration for an execution profile specifying training behavior."""

    profile_name: str = Field(..., description="Name of the execution profile")
    random_seed: int = Field(42, description="Global random seed")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    deterministic_training: bool = Field(False, description="Enforce deterministic CUDA operations")
    checkpoint_frequency: int = Field(1, description="Frequency (in epochs) to save checkpoints")
    save_best_only: bool = Field(True, description="Save only the best model checkpoint")
    batch_size: int = Field(16, description="Training batch size")
    epoch_count: int = Field(100, description="Total number of epochs")
    augmentation_policy: str = Field(
        "standard", description="Augmentation policy (minimal, standard, heavy)"
    )
    profiler_level: ProfilerLevel = Field(
        ProfilerLevel.NONE, description="Profiler verbosity level"
    )
    logging_behavior: LoggingBehavior = Field(
        LoggingBehavior.LOCAL, description="Logging sink and verbosity"
    )


DEV_EXECUTION_PROFILE = ExecutionProfile(
    profile_name="development",
    random_seed=42,
    mixed_precision=True,
    deterministic_training=False,
    checkpoint_frequency=1,
    save_best_only=False,
    batch_size=8,
    epoch_count=5,
    augmentation_policy="minimal",
    profiler_level=ProfilerLevel.BASIC,
    logging_behavior=LoggingBehavior.LOCAL,
)

RESEARCH_EXECUTION_PROFILE = ExecutionProfile(
    profile_name="research",
    random_seed=1337,
    mixed_precision=True,
    deterministic_training=True,
    checkpoint_frequency=5,
    save_best_only=True,
    batch_size=32,
    epoch_count=200,
    augmentation_policy="heavy",
    profiler_level=ProfilerLevel.NONE,
    logging_behavior=LoggingBehavior.WANDB_FULL,
)

PROD_VALIDATION_EXECUTION_PROFILE = ExecutionProfile(
    profile_name="production_validation",
    random_seed=42,
    mixed_precision=True,
    deterministic_training=True,
    checkpoint_frequency=1,
    save_best_only=True,
    batch_size=32,
    epoch_count=1,  # typically 1 for validation
    augmentation_policy="none",
    profiler_level=ProfilerLevel.NONE,
    logging_behavior=LoggingBehavior.LOCAL,
)

EXECUTION_PROFILE_REGISTRY = {
    "development": DEV_EXECUTION_PROFILE,
    "research": RESEARCH_EXECUTION_PROFILE,
    "production_validation": PROD_VALIDATION_EXECUTION_PROFILE,
}


def get_execution_profile(name: str) -> ExecutionProfile:
    if name not in EXECUTION_PROFILE_REGISTRY:
        raise ValueError(f"Unknown execution profile: {name}")
    return EXECUTION_PROFILE_REGISTRY[name]
