import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SeedManager:
    """Ensures reproducibility across the platform."""

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for python, numpy, and optionally PyTorch."""
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

class MixedPrecisionManager(ABC):
    """Abstract interface for mixed precision training."""

    @abstractmethod
    def autocast(self) -> Any:
        """Context manager for forward pass casting."""
        pass

    @abstractmethod
    def scale_loss(self, loss: Any) -> Any:
        """Scale the loss for backward pass."""
        pass

    @abstractmethod
    def step_optimizer(self, optimizer: Any) -> None:
        """Step the optimizer with unscaled gradients."""
        pass

class DistributedStrategy(ABC):
    """Abstract interface for handling multi-GPU or multi-node training."""

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        pass

    @abstractmethod
    def prepare_dataloader(self, dataloader: Any) -> Any:
        pass

class MultiOptimizerManager:
    """Manages multiple optimizers (e.g. generator and discriminator) by logical name."""

    def __init__(self, optimizers: dict[str, Any]):
        self.optimizers = optimizers

    def get_optimizer(self, name: str) -> Any:
        return self.optimizers[name]

    def zero_grad(self, name: str | None = None, set_to_none: bool = True) -> None:
        """Zeros gradients for a specific optimizer or all if name is None."""
        if name:
            self.optimizers[name].zero_grad(set_to_none=set_to_none)
        else:
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=set_to_none)

    def step(self, name: str | None = None) -> None:
        """Steps a specific optimizer or all if name is None."""
        if name:
            self.optimizers[name].step()
        else:
            for opt in self.optimizers.values():
                opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, state in state_dict.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)

class AdversarialLossManager:
    """Manages composable weighted generator and discriminator loss components."""

    def __init__(self, g_losses: dict[str, tuple[Any, float]], d_losses: dict[str, tuple[Any, float]]):
        """
        Args:
            g_losses: Dict mapping loss name to a tuple (loss_function, weight).
            d_losses: Dict mapping loss name to a tuple (loss_function, weight).
        """
        self.g_losses = g_losses
        self.d_losses = d_losses

    def compute_generator_loss(self, **kwargs) -> tuple[Any, dict[str, float]]:
        """Computes the total weighted generator loss and individual loss components."""
        total_loss = 0.0
        components = {}
        for name, (loss_fn, weight) in self.g_losses.items():
            loss_val = loss_fn(**kwargs)
            weighted_loss = loss_val * weight
            total_loss += weighted_loss
            components[f"g_loss_{name}"] = weighted_loss.item()
        return total_loss, components

    def compute_discriminator_loss(self, **kwargs) -> tuple[Any, dict[str, float]]:
        """Computes the total weighted discriminator loss and individual loss components."""
        total_loss = 0.0
        components = {}
        for name, (loss_fn, weight) in self.d_losses.items():
            loss_val = loss_fn(**kwargs)
            weighted_loss = loss_val * weight
            total_loss += weighted_loss
            components[f"d_loss_{name}"] = weighted_loss.item()
        return total_loss, components
