from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class OptimizerProvider(ABC):
    """Provides the optimizer for the training engine."""

    @abstractmethod
    def get_optimizer(self, model_parameters: Any) -> Any:
        """Instantiate and return the optimizer."""
        pass


class SchedulerProvider(ABC):
    """Provides the learning rate scheduler for the training engine."""

    @abstractmethod
    def get_scheduler(self, optimizer: Any) -> Any:
        """Instantiate and return the scheduler."""
        pass


class LossProvider(ABC):
    """Provides the loss function for the training engine."""

    @abstractmethod
    def get_loss(self) -> Callable[..., Any]:
        """Instantiate and return the loss function."""
        pass
