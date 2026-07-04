"""Metrics registry."""
from collections.abc import Callable
from typing import Any


class MetricsRegistry:
    """Central registry for evaluation metrics."""

    _metrics: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, metric_func: Callable) -> None:
        """Register an evaluation metric (e.g., dice_score, accuracy)."""
        cls._metrics[name] = metric_func

    @classmethod
    def compute(cls, name: str, y_true: Any, y_pred: Any) -> float:
        """Compute a registered metric."""
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found.")
        return cls._metrics[name](y_true, y_pred)
