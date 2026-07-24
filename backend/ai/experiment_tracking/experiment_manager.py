import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import uuid
from datetime import datetime
from typing import Any

import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExperimentManager:
    """Generic, model-agnostic and dataset-agnostic Experiment Manager."""

    def __init__(self, base_dir: str = "outputs/experiments"):
        self.base_dir = base_dir
        self.experiment_id = None
        self.experiment_dir = None

        self.dirs = {}

        self.metadata = {}
        self.checkpoints = []
        self.artifacts = []

        self._start_time = None
        self._step_metrics_count = 0
        self._epoch_metrics_count = 0

    def start(self, prefix: str = "exp", experiment_id: str | None = None):
        """Initializes directories, generates ID, captures timestamp and environment info."""
        if experiment_id:
            self.experiment_id = experiment_id
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.experiment_id = f"{prefix}_{timestamp_str}_{unique_id}"

        self.experiment_dir = os.path.join(self.base_dir, self.experiment_id)

        # 4. Fail loudly if duplicate directory exists
        if os.path.exists(self.experiment_dir):
            raise FileExistsError(f"Experiment directory {self.experiment_dir} already exists! Collision detected.")

        self.dirs = {
            "metadata": os.path.join(self.experiment_dir, "metadata"),
            "configs": os.path.join(self.experiment_dir, "configs"),
            "metrics": os.path.join(self.experiment_dir, "metrics"),
            "checkpoints": os.path.join(self.experiment_dir, "checkpoints"),
            "artifacts": os.path.join(self.experiment_dir, "artifacts"),
            "logs": os.path.join(self.experiment_dir, "logs"),
            "report": os.path.join(self.experiment_dir, "report")
        }

        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self._start_time = datetime.now()
        self._capture_environment()

    def _get_package_version(self, pkg_name: str) -> str:
        try:
            return importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            return "N/A"

    def _capture_environment(self):
        """Introspects hardware, ML-relevant runtime environment."""
        env_metadata = {
            "timestamp": self._start_time.isoformat(),
            "experiment_id": self.experiment_id,
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.processor(),
            "cpu_cores": os.cpu_count(),
            "packages": {
                "numpy": self._get_package_version("numpy"),
                "scipy": self._get_package_version("scipy")
            }
        }

        if TORCH_AVAILABLE:
            env_metadata["packages"]["torch"] = self._get_package_version("torch")
            env_metadata["packages"]["torchvision"] = self._get_package_version("torchvision")
            env_metadata["pytorch_version"] = torch.__version__
            env_metadata["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_metadata["cuda_device_count"] = torch.cuda.device_count()
                env_metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
                env_metadata["cudnn_version"] = torch.backends.cudnn.version()

        self.metadata.update({"environment": env_metadata})
        self.save_metadata({})

    def save_config(self, config_dict: dict[str, Any]):
        """Dumps dictionary configuration."""
        config_path = os.path.join(self.dirs["configs"], "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def save_metadata(self, metadata_dict: dict[str, Any]):
        """Merges and saves captured metadata (includes runtime, batch sizes, etc)."""
        self.metadata.update(metadata_dict)
        metadata_path = os.path.join(self.dirs["metadata"], "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def log_step_metrics(self, step: int, metrics: dict[str, float]):
        """Logs step-level metrics."""
        metrics_path = os.path.join(self.dirs["metrics"], "step_metrics.jsonl")
        metrics_copy = metrics.copy()
        metrics_copy["step"] = step
        metrics_copy["timestamp"] = datetime.now().isoformat()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_copy) + "\n")
        self._step_metrics_count += 1

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]):
        """Logs epoch-level metrics."""
        metrics_path = os.path.join(self.dirs["metrics"], "epoch_metrics.jsonl")
        metrics_copy = metrics.copy()
        metrics_copy["epoch"] = epoch
        metrics_copy["timestamp"] = datetime.now().isoformat()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_copy) + "\n")
        self._epoch_metrics_count += 1

    def _calculate_checksum(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def register_checkpoint(self, file_path: str, epoch: int, checksum: str | None = None):
        """Register checkpoint metadata (path, size, timestamp, checksum)."""
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.dirs["checkpoints"], filename)

        # Copy checkpoint if it's not already in the dest_path
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            shutil.copy2(file_path, dest_path)

        if not checksum:
            checksum = self._calculate_checksum(dest_path)

        file_stat = os.stat(dest_path)

        checkpoint_meta = {
            "filename": filename,
            "path": dest_path,
            "epoch": epoch,
            "size_bytes": file_stat.st_size,
            "timestamp": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "checksum": checksum
        }
        self.checkpoints.append(checkpoint_meta)

        idx_path = os.path.join(self.dirs["checkpoints"], "checkpoint_registry.json")
        with open(idx_path, "w") as f:
            json.dump(self.checkpoints, f, indent=4)

    def register_artifact(self, file_path: str, artifact_type: str, description: str, epoch: int | None = None):
        """Register artifact metadata (type, description, originating epoch, timestamp, checksum)."""
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.dirs["artifacts"], filename)

        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            shutil.copy2(file_path, dest_path)

        checksum = self._calculate_checksum(dest_path)
        file_stat = os.stat(dest_path)

        artifact_meta = {
            "filename": filename,
            "path": dest_path,
            "type": artifact_type,
            "description": description,
            "originating_epoch": epoch,
            "size_bytes": file_stat.st_size,
            "timestamp": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "checksum": checksum
        }
        self.artifacts.append(artifact_meta)

        idx_path = os.path.join(self.dirs["artifacts"], "artifact_registry.json")
        with open(idx_path, "w") as f:
            json.dump(self.artifacts, f, indent=4)

    def finish(self):
        """Finalizes the experiment and generates an execution summary."""
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds() if self._start_time else 0

        summary = {
            "status": "completed",
            "start_time": self._start_time.isoformat() if self._start_time else "N/A",
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_step_metrics": self._step_metrics_count,
            "total_epoch_metrics": self._epoch_metrics_count,
            "total_checkpoints": len(self.checkpoints),
            "total_artifacts": len(self.artifacts)
        }
        self.save_metadata({"execution_summary": summary})

        report_path = os.path.join(self.dirs["report"], "experiment_report.md")
        with open(report_path, "w") as f:
            f.write(f"# Experiment Report: {self.experiment_id}\n\n")
            f.write(f"**Status:** {summary['status'].capitalize()}\n")
            f.write(f"**Start Time:** {summary['start_time']}\n")
            f.write(f"**End Time:** {summary['end_time']}\n")
            f.write(f"**Duration:** {summary['duration_seconds']:.2f} seconds\n\n")
            f.write("## Metrics\n")
            f.write(f"- Step Metrics Logs: {summary['total_step_metrics']}\n")
            f.write(f"- Epoch Metrics Logs: {summary['total_epoch_metrics']}\n\n")
            f.write(f"## Artifacts\n{summary['total_artifacts']} registered.\n\n")
            f.write(f"## Checkpoints\n{summary['total_checkpoints']} registered.\n")
