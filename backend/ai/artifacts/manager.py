"""Artifact management."""

import os
from typing import Any


class ArtifactManager:
    """Manages checkpoints, logs, XAI outputs (Grad-CAM/SHAP), masks, and reports."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _get_path(self, run_id: str, artifact_type: str, filename: str) -> str:
        return os.path.join(self.base_dir, run_id, artifact_type, filename)

    def save_checkpoint(self, run_id: str, checkpoint_data: Any, filename: str) -> str:
        """Save a model checkpoint."""
        path = self._get_path(run_id, "checkpoints", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Checkpoint serialization is handled by callbacks, this just returns the path or could do the save
        return path

    def export_model(self, run_id: str, model: Any, filename: str) -> str:
        """Export the final model (e.g. TorchScript, ONNX) through BaseModel.export()."""
        path = self._get_path(run_id, "models", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.export(path)
        return path

    def save_xai_output(self, run_id: str, output_data: Any, method: str, filename: str) -> str:
        """Save XAI outputs like Grad-CAM or SHAP."""
        return self._get_path(run_id, f"xai/{method}", filename)

    def save_mask(self, run_id: str, mask_data: Any, filename: str) -> str:
        """Save segmentation masks."""
        return self._get_path(run_id, "masks", filename)

    def save_report(self, run_id: str, report_content: str, filename: str) -> str:
        """Save text/JSON/PDF evaluation reports."""
        return self._get_path(run_id, "reports", filename)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Log evaluation or training metrics."""
        pass
