"""Experiment tracking abstraction."""
from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking systems (e.g. MLflow)."""

    @abstractmethod
    def start_run(self, experiment_name: str, run_name: str | None = None) -> Any:
        """Start a new tracking run."""
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters."""
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        pass

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a model artifact."""
        pass
