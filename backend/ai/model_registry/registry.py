"""Model registry implementation."""
from datetime import datetime

from pydantic import BaseModel, Field


class ModelRegistration(BaseModel):
    """Metadata for a registered model."""
    name: str
    version: str
    task: str
    dataset: str
    metrics: dict[str, float] = Field(default_factory=dict)
    checksum: str
    framework: str
    created_date: datetime = Field(default_factory=datetime.utcnow)
    approval_status: str = Field(default="pending")
    production_status: str = Field(default="development")
    artifact_location: str

class ModelRegistry:
    """Central registry for all AI models and their metadata."""

    _registry: dict[str, list[ModelRegistration]] = {}

    @classmethod
    def register(cls, model: ModelRegistration) -> None:
        """Register a new model version."""
        if model.name not in cls._registry:
            cls._registry[model.name] = []
        # Check if version exists
        for existing in cls._registry[model.name]:
            if existing.version == model.version:
                raise ValueError(f"Version {model.version} for model {model.name} already exists.")

        cls._registry[model.name].append(model)

    @classmethod
    def get_model(cls, name: str, version: str | None = None) -> ModelRegistration:
        """Retrieve a model by name and optionally version (returns latest if version is None)."""
        if name not in cls._registry or not cls._registry[name]:
            raise KeyError(f"Model {name} not found in registry.")

        models = cls._registry[name]
        if version is None:
            # Sort by created_date descending and return first
            return sorted(models, key=lambda m: m.created_date, reverse=True)[0]

        for m in models:
            if m.version == version:
                return m

        raise KeyError(f"Version {version} for model {name} not found.")

    @classmethod
    def get_models_by_task(cls, task: str) -> list[ModelRegistration]:
        """Retrieve all registered models for a specific task (e.g., 'segmentation', 'classification')."""
        result = []
        for models in cls._registry.values():
            for m in models:
                if m.task == task:
                    result.append(m)
        return result
