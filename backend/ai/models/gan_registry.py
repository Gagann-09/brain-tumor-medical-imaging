from collections.abc import Callable
from typing import Any


class GANComponentRegistry:
    """Registry for GAN components: generators, discriminators, losses, metrics, etc."""

    _generators: dict[str, Callable[..., Any]] = {}  # type: ignore  # type: ignore
    _discriminators: dict[str, Callable[..., Any]] = {}  # type: ignore  # type: ignore
    _losses: dict[str, Callable[..., Any]] = {}  # type: ignore  # type: ignore
    _metrics: dict[str, Callable[..., Any]] = {}  # type: ignore  # type: ignore
    _regularizers: dict[str, Callable[..., Any]] = {}  # type: ignore  # type: ignore

    @classmethod
    def register_generator(cls, name: str) -> Callable:
        def decorator(factory: Callable[..., Any]) -> Callable:
            cls._generators[name] = factory
            return factory

        return decorator

    @classmethod
    def register_discriminator(cls, name: str) -> Callable:
        def decorator(factory: Callable[..., Any]) -> Callable:
            cls._discriminators[name] = factory
            return factory

        return decorator

    @classmethod
    def register_loss(cls, name: str) -> Callable:
        def decorator(factory: Callable[..., Any]) -> Callable:
            cls._losses[name] = factory
            return factory

        return decorator

    @classmethod
    def register_metric(cls, name: str) -> Callable:
        def decorator(factory: Callable[..., Any]) -> Callable:
            cls._metrics[name] = factory
            return factory

        return decorator

    @classmethod
    def register_regularizer(cls, name: str) -> Callable:
        def decorator(factory: Callable[..., Any]) -> Callable:
            cls._regularizers[name] = factory
            return factory

        return decorator

    @classmethod
    def get_generator(cls, name: str, **kwargs) -> Any:
        if name not in cls._generators:
            raise KeyError(f"Generator {name} not found in registry.")
        return cls._generators[name](**kwargs)

    @classmethod
    def get_discriminator(cls, name: str, **kwargs) -> Any:
        if name not in cls._discriminators:
            raise KeyError(f"Discriminator {name} not found in registry.")
        return cls._discriminators[name](**kwargs)

    @classmethod
    def get_loss(cls, name: str, **kwargs) -> Any:
        if name not in cls._losses:
            raise KeyError(f"Loss {name} not found in registry.")
        return cls._losses[name](**kwargs)
