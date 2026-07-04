"""State management for AI lifecycles."""
from datetime import datetime
from enum import Enum
from typing import Any


class AIModelState(str, Enum):
    """Enumeration of model lifecycle states."""
    TRAINING = "Training"
    EVALUATING = "Evaluating"
    READY = "Ready"
    SERVING = "Serving"
    ARCHIVED = "Archived"
    FAILED = "Failed"

class AIStateManager:
    """Tracks and transitions model lifecycle states."""

    def __init__(self):
        # Placeholder for DB/Redis state storage
        self._states: dict[str, dict[str, Any]] = {}

    def set_state(self, model_id: str, state: AIModelState, metadata: dict[str, Any] | None = None) -> None:
        """Update the state of a model."""
        self._states[model_id] = {
            "state": state,
            "updated_at": datetime.utcnow(),
            "metadata": metadata or {}
        }

    def get_state(self, model_id: str) -> AIModelState:
        """Retrieve the current state of a model."""
        if model_id not in self._states:
            raise KeyError(f"Model ID {model_id} not found in state manager.")
        return self._states[model_id]["state"]
