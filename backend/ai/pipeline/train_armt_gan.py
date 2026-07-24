import argparse
from pathlib import Path

import torch

# AI Platform Core
from ai.config.training_config import (
    DatasetConfig,
    ExperimentConfig,
    HardwareConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
)
from ai.datasets.adapters.brats import BraTSAdapter
from ai.datasets.adapters.pytorch import PyTorchDatasetAdapter
from ai.datasets.split_manager import DatasetSplitManager, PatientSplitStrategy
from ai.experiment_tracking.callbacks import ExperimentManagerCallback
from ai.experiment_tracking.experiment_manager import ExperimentManager
from ai.segmentation.models.armt_gan import ARMTGANModel
from ai.training.callbacks import (
    CheckpointCallback,
    ModelCardCallback,
    RuntimeMonitorCallback,
    TrainingHealthCallback,
)
from ai.training.components import (
    AdversarialLossManager,
    MixedPrecisionManager,
    MultiOptimizerManager,
)
from ai.training.data import TrainingDataset, ValidationDataset
from ai.training.hardware import PyTorchHardwareManager
from ai.training.manager import TrainingManager
from ai.training.strategy import AlternatingUpdatePolicy, GANTrainingStrategy


class DummyMixedPrecision(MixedPrecisionManager):
    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    def autocast(self):
        from contextlib import nullcontext
        if self.device_type == "cuda":
            return torch.amp.autocast(device_type=self.device_type, dtype=torch.float16)
        return nullcontext()

    def scale_loss(self, loss):
        if self.device_type == "cuda":
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer):
        if self.device_type == "cuda":
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    config = ExperimentConfig(
        model=ModelConfig(name="ARMTGANModel", in_channels=4, out_channels=1),
        dataset=DatasetConfig(registry_id="BraTS", batch_size=2, kwargs={"data_dir": args.data_dir}),
        optimizer=OptimizerConfig(name="AdamW", learning_rate=2e-4),
        hardware=HardwareConfig(device="cpu"), # Defaulting to cpu to allow testing easily if no GPU
        logging=LoggingConfig(experiment_name="brats_armt_gan"),
        max_epochs=args.epochs,
    )
    if torch.cuda.is_available():
        config.hardware.device = "cuda"

    # Dataset
    adapter = BraTSAdapter(root_dir=args.data_dir)
    studies = list(adapter.load_studies())

    split_manager = DatasetSplitManager(PatientSplitStrategy(seed=config.seed))
    train_studies, val_studies, _ = split_manager.create_splits(
        studies, train_ratio=0.8, val_ratio=0.2
    )

    train_loader = torch.utils.data.DataLoader(
        PyTorchDatasetAdapter(TrainingDataset(train_studies)),
        batch_size=config.dataset.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PyTorchDatasetAdapter(ValidationDataset(val_studies)),
        batch_size=config.dataset.batch_size,
        shuffle=False,
    )

    # Loss Manager
    def g_bce(fake_preds, **kwargs):
        import torch.nn.functional as F

        if isinstance(fake_preds, list):
            return sum(F.binary_cross_entropy_with_logits(fp, torch.ones_like(fp)) for fp in fake_preds) / len(fake_preds)
        return F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))

    def g_l1(fake_masks, real_masks, **kwargs):
        import torch.nn.functional as F

        return F.l1_loss(fake_masks, real_masks)

    def d_bce(real_preds, fake_preds, **kwargs):
        import torch.nn.functional as F

        if isinstance(real_preds, list):
            real_loss = sum(F.binary_cross_entropy_with_logits(rp, torch.ones_like(rp)) for rp in real_preds) / len(real_preds)
            fake_loss = sum(F.binary_cross_entropy_with_logits(fp, torch.zeros_like(fp)) for fp in fake_preds) / len(fake_preds)
        else:
            real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
            fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))

        return (real_loss + fake_loss) / 2.0

    loss_mgr = AdversarialLossManager(
        g_losses={"bce": (g_bce, 1.0), "l1": (g_l1, 10.0)}, d_losses={"bce": (d_bce, 1.0)}
    )

    # Model
    model = ARMTGANModel(loss_manager=loss_mgr)

    # Optimizers
    opt_g = torch.optim.AdamW(model.generator.parameters(), lr=config.optimizer.learning_rate)
    opt_d = torch.optim.AdamW(model.discriminator.parameters(), lr=config.optimizer.learning_rate)
    multi_opt = MultiOptimizerManager({"generator": opt_g, "discriminator": opt_d})

    hardware_manager = PyTorchHardwareManager(device_type=config.hardware.device)
    mixed_precision = DummyMixedPrecision(device_type=hardware_manager.device.type)

    start_epoch = 1
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")  # noqa: T201
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            try:
                # Assuming multi_opt state dict is structured correctly
                multi_opt.load_state_dict(checkpoint["optimizer_state"])
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")  # noqa: T201
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

    # Strategy
    strategy = GANTrainingStrategy(
        optimizer=multi_opt,
        mixed_precision=mixed_precision,
        hardware=hardware_manager,
        event_bus=None,
        update_policy=AlternatingUpdatePolicy(),
    )

    manager = TrainingManager(
        config=config,
        model=model,
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        hardware_manager=hardware_manager,
    )
    strategy.event_bus = manager.event_bus

    # Register instrumentation
    manager.register_callback(RuntimeMonitorCallback())
    manager.register_callback(TrainingHealthCallback())

    # ExperimentManager integration
    em = ExperimentManager(base_dir="outputs/experiments")
    em_callback = ExperimentManagerCallback(
        experiment_manager=em,
        checkpoint_dir=f"outputs/{config.logging.experiment_name}/checkpoints",
        artifact_dir=f"outputs/{config.logging.experiment_name}",
    )
    manager.register_callback(em_callback)

    save_dir = Path(args.output_dir) / config.logging.experiment_name
    manager.register_callback(
        CheckpointCallback(
            save_dir=str(save_dir / "checkpoints"),
            model=model,
            strategy=strategy,
            config=config,
            monitor_metric="val_loss",
            mode="min",
        )
    )

    from ai.models.model_card import ModelCardConfig
    mc_config = ModelCardConfig(
        model_name="ARMT-GAN",
        architecture="ARMT-GAN (U-Net Generator, PatchGAN Discriminator)",
        description="Functional verification of ARMT-GAN training",
        author="ARMT-GAN AI System",
    )
    manager.register_callback(
        ModelCardCallback(
            config=config,
            model_details=mc_config,
            save_dir=str(save_dir),
        )
    )

    manager.start_training(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
