"""Transform registry."""

from collections.abc import Callable


class TransformRegistry:
    """Central registry for preprocessing and augmentation pipelines."""

    _transforms: dict[str, Callable] = {}  # type: ignore  # type: ignore

    @classmethod
    def register(cls, name: str, transform_func: Callable) -> None:
        """Register a transformation pipeline."""
        cls._transforms[name] = transform_func

    @classmethod
    def get(cls, name: str) -> Callable:
        """Retrieve a transformation pipeline by name."""
        if name not in cls._transforms:
            raise KeyError(f"Transform '{name}' not found.")
        return cls._transforms[name]
