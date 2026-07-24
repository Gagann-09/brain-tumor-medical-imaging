import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.config.training_config import (
    DatasetConfig,
    HardwareConfig,
    ModelConfig,
    OptimizerConfig,
)
from ai.experiment_tracking.callbacks import ExperimentManagerCallback
from ai.experiment_tracking.experiment_manager import ExperimentManager
from ai.training.callbacks import CallbackManager, CheckpointCallback, ModelCardCallback
from ai.training.events import Event, EventBus, EventType


class MockConfig:
    def __init__(self):
        self.experiment_name = "mock_integration"
        self.model = ModelConfig(name="MockModel", in_channels=1, out_channels=1)
        self.dataset = DatasetConfig(name="MockData", data_dir=".", batch_size=2)
        self.optimizer = OptimizerConfig(name="Adam", learning_rate=1e-3)
        self.hardware = HardwareConfig(device="cpu")

    def model_dump(self):
        return {
            "experiment_name": self.experiment_name,
            "dataset": {"data_dir": ".", "batch_size": 2, "name": "MockData"},
            "optimizer": {"name": "Adam", "learning_rate": 1e-3},
            "hardware": {"device": "cpu"}
        }

class MockStrategy:
    def __init__(self):
        self.optimizer = type("Opt", (), {"state_dict": lambda self: {}})()

class MockModel:
    def state_dict(self):
        return {}

def main():
    print("Running mocked training integration...")
    event_bus = EventBus()
    callback_manager = CallbackManager(event_bus)

    config = MockConfig()
    model = MockModel()
    strategy = MockStrategy()

    save_dir = "outputs/mock_integration"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # Existing callbacks
    ckpt_cb = CheckpointCallback(
        save_dir=os.path.join(save_dir, "checkpoints"),
        model=model,
        strategy=strategy,
        config=config,
        monitor_metric="val_loss",
        mode="min"
    )
    callback_manager.add_callback(ckpt_cb)

    mc_cb = ModelCardCallback(
        config=config,
        model_details={"model_name": "Mock", "architecture": "MockArch", "description": "Mock desc"},
        save_dir=save_dir
    )
    callback_manager.add_callback(mc_cb)

    # New ExperimentManagerCallback
    em = ExperimentManager(base_dir="outputs/experiments")
    em_cb = ExperimentManagerCallback(
        experiment_manager=em,
        checkpoint_dir=os.path.join(save_dir, "checkpoints"),
        artifact_dir=save_dir
    )
    callback_manager.add_callback(em_cb)

    # Setup
    callback_manager.setup_bus()

    # Simulate Lifecycle
    event_bus.publish(Event(EventType.TRAINING_START, {"config": config}))
    event_bus.publish(Event(EventType.EPOCH_START, {"mode": "train", "epoch": 1}))
    event_bus.publish(Event(EventType.BATCH_START, {"mode": "train", "batch": 0}))
    event_bus.publish(Event(EventType.BATCH_END, {"mode": "train", "batch": 0, "loss": 0.5, "metrics": {"loss": 0.5}}))
    event_bus.publish(Event(EventType.EPOCH_END, {"mode": "train", "epoch": 1, "metrics": {"loss": 0.5}}))

    # Eval
    event_bus.publish(Event(EventType.EVALUATION_START, {"mode": "val"}))
    event_bus.publish(Event(EventType.EVALUATION_END, {"mode": "val", "metrics": {"val_loss": 0.3}}))

    # Overwrite checkpoint manually for simulating updated checksum
    import time
    time.sleep(1)

    # End
    event_bus.publish(Event(EventType.TRAINING_END, {}))

    print(f"Mock integration passed! Experiment saved at {em.experiment_dir}")

if __name__ == "__main__":
    main()
