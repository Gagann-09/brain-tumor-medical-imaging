from abc import ABC
from typing import Any

from .events import Event, EventBus, EventType


class BaseEngine(ABC):
    """Base class for all engines (Training, Evaluation, Prediction)."""

    def __init__(self, model: Any, event_bus: EventBus):
        self.model = model
        self.event_bus = event_bus

class TrainingEngine(BaseEngine):
    """Handles the core training loop for a single epoch."""

    def __init__(self, model: Any, strategy: Any, event_bus: EventBus):
        super().__init__(model, event_bus)
        self.strategy = strategy

    def train_epoch(self, dataloader: Any, epoch_idx: int) -> dict[str, float]:
        """
        Executes one training epoch.
        Returns a dictionary of metrics for this epoch.
        """
        self.event_bus.publish(Event(EventType.EPOCH_START, {"epoch": epoch_idx, "mode": "train"}))

        # We rely on the adapter for setting train mode (e.g., model.train() in PyTorch)
        if hasattr(self.model, "train"):
            self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            self.event_bus.publish(Event(EventType.BATCH_START, {"batch": batch_idx, "mode": "train"}))

            # Delegate step execution to strategy (handles mixed precision, backward, step)
            step_metrics = self.strategy.execute_step(self.model, batch, batch_idx)

            # Aggregate metrics
            loss_val = step_metrics.get("loss", 0.0)
            epoch_loss += loss_val
            num_batches += 1

            self.event_bus.publish(Event(EventType.BATCH_END, {"batch": batch_idx, "mode": "train", "loss": loss_val, "metrics": step_metrics}))

        avg_loss = epoch_loss / max(1, num_batches)
        metrics = {"loss": avg_loss}

        self.event_bus.publish(Event(EventType.EPOCH_END, {"epoch": epoch_idx, "mode": "train", "metrics": metrics}))
        return metrics

class EvaluationEngine(BaseEngine):
    """Handles the evaluation loop."""

    def __init__(self, model: Any, event_bus: EventBus):
        super().__init__(model, event_bus)

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """
        Executes evaluation over the dataset.
        """
        self.event_bus.publish(Event(EventType.EVALUATION_START, {"mode": "val"}))

        if hasattr(self.model, "eval"):
            self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_metrics = {}

        import torch
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = self.model.validation_step(batch, batch_idx)
                total_loss += output.loss
                num_batches += 1
                for k, v in output.metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v

        avg_loss = total_loss / max(1, num_batches)
        avg_metrics = {k: v / max(1, num_batches) for k, v in all_metrics.items()}
        metrics = {"val_loss": avg_loss, **avg_metrics}
        self.event_bus.publish(Event(EventType.EVALUATION_END, {"mode": "val", "metrics": metrics}))
        return metrics

class PredictionEngine(BaseEngine):
    """Handles inference/prediction."""

    def predict(self, dataloader: Any) -> Any:
        if hasattr(self.model, "eval"):
            self.model.eval()

        predictions = []
        import torch
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = self.model.prediction_step(batch, batch_idx)
                predictions.append(output)
        return predictions
