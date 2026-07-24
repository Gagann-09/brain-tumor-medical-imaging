import os

from ai.training.callbacks import Callback
from ai.training.events import Event

from .collector import ArtifactCollector
from .experiment_manager import ExperimentManager


class ExperimentManagerCallback(Callback):
    """
    Integrates ExperimentManager into the training lifecycle via composition.
    Runs with lowest priority (-10) to guarantee execution after CheckpointCallback
    and ModelCardCallback have finished writing to the filesystem.
    """
    priority: int = -10

    def __init__(self, experiment_manager: ExperimentManager, checkpoint_dir: str, artifact_dir: str):
        self.em = experiment_manager
        self.checkpoint_collector = ArtifactCollector(checkpoint_dir)
        self.artifact_collector = ArtifactCollector(artifact_dir)

        # Track if we are inside a training epoch to avoid capturing validation batches
        self._current_epoch = 0

    def on_training_start(self, event: Event) -> None:
        config = event.data.get("config", {})
        config_dict = config.model_dump() if hasattr(config, "model_dump") else (config if isinstance(config, dict) else {})

        # In case the manager wasn't manually started
        if not self.em.experiment_id:
            # We can use the experiment name from config if available
            exp_name = config_dict.get("experiment_name", "exp")
            self.em.start(prefix=exp_name)

        self.em.save_config(config_dict)

        # If dataset details are in config, save them as metadata
        dataset_meta = config_dict.get("dataset", {})
        optimizer_meta = config_dict.get("optimizer", {})
        self.em.save_metadata({
            "dataset_config": dataset_meta,
            "training_params": {
                "batch_size": dataset_meta.get("batch_size"),
                "learning_rate": optimizer_meta.get("learning_rate"),
                "optimizer": optimizer_meta.get("name")
            }
        })

    def on_epoch_start(self, event: Event) -> None:
        if event.data.get("mode") == "train":
            self._current_epoch = event.data.get("epoch", 0)

    def on_batch_end(self, event: Event) -> None:
        if event.data.get("mode") == "train":
            metrics = event.data.get("metrics", {})
            if metrics:
                # Assuming batch_idx is passed as 'batch', we can compute global step
                # or just use batch_idx if that's all we have.
                step = event.data.get("batch", 0)
                self.em.log_step_metrics(step, metrics)

    def on_epoch_end(self, event: Event) -> None:
        if event.data.get("mode") == "train":
            metrics = event.data.get("metrics", {})
            if metrics:
                epoch = event.data.get("epoch", self._current_epoch)
                self.em.log_epoch_metrics(epoch, metrics)

    def on_evaluation_end(self, event: Event) -> None:
        # After evaluation, CheckpointCallback (priority 10) may have saved a checkpoint.
        # Scan and register any new/updated checkpoints.
        new_checkpoints = self.checkpoint_collector.scan()
        for filepath, checksum in new_checkpoints:
            # register_checkpoint handles copying if necessary
            self.em.register_checkpoint(filepath, epoch=self._current_epoch, checksum=checksum)

    def on_training_end(self, event: Event) -> None:
        # After training, ModelCardCallback (priority 0) may have saved artifacts.
        new_artifacts = self.artifact_collector.scan()
        for filepath, _checksum in new_artifacts:
            # Deduce a basic artifact type from extension
            ext = os.path.splitext(filepath)[1].lower()
            a_type = "model_card" if "model_card" in filepath else ("image" if ext in ['.png', '.jpg'] else "document")
            self.em.register_artifact(filepath, artifact_type=a_type, description=f"Artifact {os.path.basename(filepath)}", epoch=self._current_epoch)

        self.em.finish()
