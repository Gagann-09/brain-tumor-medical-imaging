import os
import time
from abc import ABC
from pathlib import Path
from typing import Any

from .events import Event, EventBus, EventType


class Callback(ABC):
    """
    Base class for training callbacks.
    Callbacks register themselves with the EventBus to respond to lifecycle events.
    Priority determines the execution order (higher priority executes first).
    """

    priority: int = 0

    def register(self, event_bus: EventBus) -> None:
        """Register the callback methods to the event bus."""
        event_bus.subscribe(EventType.TRAINING_START, self.on_training_start)
        event_bus.subscribe(EventType.TRAINING_END, self.on_training_end)
        event_bus.subscribe(EventType.EPOCH_START, self.on_epoch_start)
        event_bus.subscribe(EventType.EPOCH_END, self.on_epoch_end)
        event_bus.subscribe(EventType.BATCH_START, self.on_batch_start)
        event_bus.subscribe(EventType.BATCH_END, self.on_batch_end)
        event_bus.subscribe(EventType.EVALUATION_START, self.on_evaluation_start)
        event_bus.subscribe(EventType.EVALUATION_END, self.on_evaluation_end)

    def on_training_start(self, event: Event) -> None:
        pass

    def on_training_end(self, event: Event) -> None:
        pass

    def on_epoch_start(self, event: Event) -> None:
        pass

    def on_epoch_end(self, event: Event) -> None:
        pass

    def on_batch_start(self, event: Event) -> None:
        pass

    def on_batch_end(self, event: Event) -> None:
        pass

    def on_evaluation_start(self, event: Event) -> None:
        pass

    def on_evaluation_end(self, event: Event) -> None:
        pass


class CallbackManager:
    """Manages callback registration to ensure deterministic ordering based on priority."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.callbacks: list[Callback] = []

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        # Sort callbacks by priority descending
        self.callbacks.sort(key=lambda cb: cb.priority, reverse=True)
        # Re-registering here isn't ideal for an active bus, but works for setup phase.
        # Typically, the Manager handles the setup before training starts.

    def setup_bus(self) -> None:
        """Registers all sorted callbacks to the EventBus."""
        for callback in self.callbacks:
            callback.register(self.event_bus)


class CheckpointCallback(Callback):
    """Saves extensive checkpoints at the end of each epoch or if metric improves."""

    priority = 10  # Save after logging

    def __init__(
        self,
        save_dir: str,
        model: Any,
        strategy: Any,
        config: Any,
        monitor_metric: str = "val_loss",
        mode: str = "min",
        keep_top_k: int = 3,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.strategy = strategy
        self.config = config
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.keep_top_k = keep_top_k
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: list[tuple[float, str]] = []

    def on_epoch_end(self, event: Event) -> None:
        event.data.get("metrics", {})
        event.data.get("epoch", 0)

        # We assume validation happens after training and fires EVALUATION_END,
        # but if we just hook into EPOCH_END, we check if validation metrics are passed.
        # Alternatively, hook into EVALUATION_END.
        pass

    def on_evaluation_end(self, event: Event) -> None:
        metrics = event.data.get("metrics", {})
        val_metric = metrics.get(self.monitor_metric)
        if val_metric is None:
            return

        is_best = False
        if (self.mode == "min" and val_metric < self.best_metric) or (
            self.mode == "max" and val_metric > self.best_metric
        ):
            self.best_metric = val_metric
            is_best = True

        if is_best:
            self._save_checkpoint(val_metric, metrics)

    def _save_checkpoint(self, val_metric: float, metrics: dict) -> None:
        import torch

        timestamp = int(time.time())
        filename = f"checkpoint_{timestamp}_{self.monitor_metric}_{val_metric:.4f}.pth"
        filepath = self.save_dir / filename

        # Determine if we have a single optimizer or MultiOptimizerManager
        optimizer_state = {}
        if hasattr(self.strategy.optimizer, "state_dict"):
            optimizer_state = self.strategy.optimizer.state_dict()

        # Extensive serialization
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer_state,
            # "scheduler_state": self.strategy.scheduler.state_dict() if hasattr(self.strategy, 'scheduler') else None,
            "metrics": metrics,
            "experiment_id": getattr(self.config, "experiment_name", "unknown"),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else {},
            # "git_hash": ...
        }

        torch.save(checkpoint, filepath)

        self.saved_checkpoints.append((val_metric, str(filepath)))
        self.saved_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Clean up old checkpoints
        if len(self.saved_checkpoints) > self.keep_top_k:
            _, old_filepath = self.saved_checkpoints.pop(-1)
            if os.path.exists(old_filepath):
                os.remove(old_filepath)


class ModelCardCallback(Callback):
    """Generates a Model Card at the end of training."""

    priority = 0

    def __init__(self, config: Any, model_details: Any, save_dir: str):
        self.config = config
        self.model_details = model_details
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics: dict[str, float] = {}

    def on_evaluation_end(self, event: Event) -> None:
        # Keep track of the latest or best metrics
        metrics = event.data.get("metrics", {})
        # For simplicity, we just keep the latest, or we could track max dice.
        self.best_metrics.update(metrics)

    def on_training_end(self, event: Event) -> None:
        from ai.models.model_card import ModelCard

        card = ModelCard(
            model_details=self.model_details,
            training_config=self.config.model_dump() if hasattr(self.config, "model_dump") else {},
            metrics=self.best_metrics,
            dataset_info={
                "dataset_name": getattr(
                    getattr(self.config, "dataset", None), "registry_id", "unknown"
                )
            },
        )

        card.save(self.save_dir / "model_card.json")
        card.export_markdown(self.save_dir / "model_card.md")


class EarlyStoppingCallback(Callback):
    """Early stopping to terminate training when validation metric stops improving."""

    priority = 20

    def __init__(
        self,
        patience: int = 5,
        monitor_metric: str = "val_loss",
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def on_evaluation_end(self, event: Event) -> None:
        metrics = event.data.get("metrics", {})
        val_metric = metrics.get(self.monitor_metric)
        if val_metric is None:
            return

        improved = False
        if self.mode == "min":
            if val_metric < self.best_metric - self.min_delta:
                improved = True
        else:
            if val_metric > self.best_metric + self.min_delta:
                improved = True

        if improved:
            self.best_metric = val_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                event.data["stop_training"] = True
