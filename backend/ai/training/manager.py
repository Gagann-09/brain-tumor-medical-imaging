from typing import Any

from ai.config.training_config import ExperimentConfig

from .callbacks import Callback, CallbackManager
from .components import DistributedStrategy, SeedManager
from .engine import EvaluationEngine, TrainingEngine
from .events import Event, EventBus, EventType
from .hardware import HardwareManager


class TrainingManager:
    """High-level orchestrator for the AI training platform."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: Any,
        strategy: Any,
        train_loader: Any,
        val_loader: Any,
        hardware_manager: HardwareManager,
        distributed_strategy: DistributedStrategy | None = None
    ):
        self.config = config
        self.model = model
        self.strategy = strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hardware_manager = hardware_manager
        self.distributed_strategy = distributed_strategy

        self.event_bus = EventBus()
        self.callback_manager = CallbackManager(self.event_bus)

        # Setup reproducibility
        SeedManager.set_seed(self.config.seed)

        # Move model to hardware
        self.hardware_manager.set_device(self.config.hardware.device_ids)
        self.model = self.hardware_manager.to_device(self.model)

        if self.distributed_strategy:
            self.distributed_strategy.setup()
            self.model = self.distributed_strategy.prepare_model(self.model)
            self.train_loader = self.distributed_strategy.prepare_dataloader(self.train_loader)
            self.val_loader = self.distributed_strategy.prepare_dataloader(self.val_loader)

        # Initialize engines
        self.train_engine = TrainingEngine(self.model, self.strategy, self.event_bus)
        self.eval_engine = EvaluationEngine(self.model, self.event_bus)

    def register_callback(self, callback: Callback) -> None:
        self.callback_manager.add_callback(callback)

    def start_training(self) -> None:
        """Start the training loop."""
        self.callback_manager.setup_bus()
        self.event_bus.publish(Event(EventType.TRAINING_START, {"config": self.config}))

        from .strategy import GANTrainingStrategy
        is_gan = isinstance(self.strategy, GANTrainingStrategy)
        if is_gan:
            self.event_bus.publish(Event(EventType.GAN_TRAINING_STARTED, {"config": self.config}))

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                # Train
                train_metrics = self.train_engine.train_epoch(self.train_loader, epoch)

                # Evaluate
                val_metrics = self.eval_engine.evaluate(self.val_loader)

        except KeyboardInterrupt:
            self.event_bus.publish(Event(EventType.ERROR, {"error": "KeyboardInterrupt"}))
        except Exception as e:
            self.event_bus.publish(Event(EventType.ERROR, {"error": str(e)}))
            raise e
        finally:
            self.event_bus.publish(Event(EventType.TRAINING_END))
            if is_gan:
                self.event_bus.publish(Event(EventType.GAN_TRAINING_FINISHED))

            if self.distributed_strategy:
                self.distributed_strategy.cleanup()
