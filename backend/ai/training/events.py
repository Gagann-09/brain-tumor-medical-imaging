from collections.abc import Callable
from enum import Enum
from typing import Any


class EventType(Enum):
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    GENERATOR_STEP_START = "generator_step_start"
    GENERATOR_STEP_END = "generator_step_end"
    DISCRIMINATOR_STEP_START = "discriminator_step_start"
    DISCRIMINATOR_STEP_END = "discriminator_step_end"
    GAN_TRAINING_STARTED = "gan_training_started"
    GAN_TRAINING_FINISHED = "gan_training_finished"
    GENERATOR_PHASE_STARTED = "generator_phase_started"
    GENERATOR_PHASE_COMPLETED = "generator_phase_completed"
    DISCRIMINATOR_PHASE_STARTED = "discriminator_phase_started"
    DISCRIMINATOR_PHASE_COMPLETED = "discriminator_phase_completed"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ERROR = "error"
    INFERENCE_STARTED = "inference_started"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    INFERENCE_FINISHED = "inference_finished"
    INFERENCE_FAILED = "inference_failed"

class Event:
    """A generic event containing context data."""
    def __init__(self, event_type: EventType, data: dict[str, Any] | None = None):
        self.type = event_type
        self.data = data or {}

class EventBus:
    """Internal EventBus that publishes lifecycle events to registered subscribers."""

    def __init__(self):
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers synchronously."""
        for callback in self._subscribers[event.type]:
            callback(event)
