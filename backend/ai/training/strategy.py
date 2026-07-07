import abc
from typing import Any

from .events import Event, EventBus, EventType


class TrainingStrategy(abc.ABC):
    """
    Abstract strategy for executing a training step, handling optimization,
    mixed precision scaling, and distributed synchronization.
    """

    def __init__(self, optimizer: Any, mixed_precision: Any, hardware: Any, event_bus: EventBus):
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        self.hardware = hardware
        self.event_bus = event_bus

    @abc.abstractmethod
    def execute_step(self, model: Any, batch: Any, batch_idx: int) -> dict[str, float]:
        """Executes a single step (forward + backward + optimize) and returns metrics."""
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training loop strategy for typical supervised models (e.g. U-Net, ResNet)."""

    def execute_step(self, model: Any, batch: Any, batch_idx: int) -> dict[str, float]:

        self.optimizer.zero_grad(set_to_none=True)

        # We assume the mixed_precision context manager is available
        with self.mixed_precision.autocast():
            output = model.training_step(batch, batch_idx)
            loss = output.loss

        scaled_loss = self.mixed_precision.scale_loss(loss)
        scaled_loss.backward()

        # Gradients are unscaled and optimizer is stepped inside the mixed precision manager
        self.mixed_precision.step_optimizer(self.optimizer)

        return {"loss": loss.item(), **output.metrics}


class GANUpdatePolicy(abc.ABC):
    """Abstract policy for deciding how many discriminator/generator steps to take."""

    @abc.abstractmethod
    def should_update_discriminator(self, batch_idx: int) -> bool:
        pass

    @abc.abstractmethod
    def should_update_generator(self, batch_idx: int) -> bool:
        pass


class AlternatingUpdatePolicy(GANUpdatePolicy):
    """Standard 1:1 alternating update policy."""

    def should_update_discriminator(self, batch_idx: int) -> bool:
        return True

    def should_update_generator(self, batch_idx: int) -> bool:
        return True


class GANTrainingStrategy(TrainingStrategy):
    """Strategy for GAN training (handling multiple optimizers)."""

    def __init__(
        self,
        optimizer: Any,
        mixed_precision: Any,
        hardware: Any,
        event_bus: EventBus,
        update_policy: GANUpdatePolicy | None = None,
    ):
        super().__init__(optimizer, mixed_precision, hardware, event_bus)
        self.update_policy = update_policy or AlternatingUpdatePolicy()

    def execute_step(self, model: Any, batch: Any, batch_idx: int) -> dict[str, float]:

        # We assume self.optimizer is a MultiOptimizerManager
        metrics = {}
        total_loss = 0.0

        # 1. Discriminator Phase
        if self.update_policy.should_update_discriminator(batch_idx):
            self.event_bus.publish(
                Event(EventType.DISCRIMINATOR_PHASE_STARTED, {"batch": batch_idx})
            )
            self.event_bus.publish(Event(EventType.DISCRIMINATOR_STEP_START, {"batch": batch_idx}))

            self.optimizer.zero_grad("discriminator", set_to_none=True)
            with self.mixed_precision.autocast():
                d_output = model.discriminator_step(batch, batch_idx)
                d_loss = d_output.loss

            scaled_d_loss = self.mixed_precision.scale_loss(d_loss)
            scaled_d_loss.backward()
            self.mixed_precision.step_optimizer(self.optimizer.get_optimizer("discriminator"))
            metrics.update(d_output.metrics)
            total_loss += d_loss.item()

            self.event_bus.publish(
                Event(EventType.DISCRIMINATOR_STEP_END, {"batch": batch_idx, "loss": d_loss.item()})
            )
            self.event_bus.publish(
                Event(
                    EventType.DISCRIMINATOR_PHASE_COMPLETED,
                    {"batch": batch_idx, "loss": d_loss.item()},
                )
            )

        # 2. Generator Phase
        if self.update_policy.should_update_generator(batch_idx):
            self.event_bus.publish(Event(EventType.GENERATOR_PHASE_STARTED, {"batch": batch_idx}))
            self.event_bus.publish(Event(EventType.GENERATOR_STEP_START, {"batch": batch_idx}))

            self.optimizer.zero_grad("generator", set_to_none=True)
            with self.mixed_precision.autocast():
                g_output = model.generator_step(batch, batch_idx)
                g_loss = g_output.loss

            scaled_g_loss = self.mixed_precision.scale_loss(g_loss)
            scaled_g_loss.backward()
            self.mixed_precision.step_optimizer(self.optimizer.get_optimizer("generator"))
            metrics.update(g_output.metrics)
            total_loss += g_loss.item()

            self.event_bus.publish(
                Event(EventType.GENERATOR_STEP_END, {"batch": batch_idx, "loss": g_loss.item()})
            )
            self.event_bus.publish(
                Event(
                    EventType.GENERATOR_PHASE_COMPLETED, {"batch": batch_idx, "loss": g_loss.item()}
                )
            )

        # Total loss reported to engine
        metrics["loss"] = total_loss

        return metrics
