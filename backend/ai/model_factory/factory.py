"""Model factory abstraction."""

from typing import Any


class ModelFactory:
    """Abstract factory for creating AI models dynamically."""

    _builders: dict[str, Any] = {}  # type: ignore  # type: ignore

    @classmethod
    def register_builder(cls, architecture_name: str, builder_func: Any) -> None:
        """Register a function/class capable of building a specific architecture."""
        cls._builders[architecture_name] = builder_func

    @classmethod
    def create_model(cls, architecture_name: str, config: dict[str, Any]) -> Any:
        """Instantiate a model using its registered builder."""
        if architecture_name not in cls._builders:
            raise ValueError(f"Architecture '{architecture_name}' not registered in ModelFactory.")

        builder = cls._builders[architecture_name]
        return builder(**config)
