import argparse
from pathlib import Path

# PyTorch Integration
import torch
from monai.transforms import SpatialPadd

# AI Platform Core
from ai.config.training_config import (
    DatasetConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    OptimizerConfig,
)
from ai.datasets.adapters.brats import BraTSAdapter
from ai.datasets.adapters.pytorch import PyTorchDatasetAdapter
from ai.datasets.split_manager import DatasetSplitManager, PatientSplitStrategy

# Segmentation Specifics
from ai.segmentation.models.unet import UNetBaseline
from ai.training.callbacks import (
    CheckpointCallback,
    ModelCardCallback,
    RuntimeMonitorCallback,
    TrainingHealthCallback,
)
from ai.training.components import MixedPrecisionManager
from ai.training.data import TrainingDataset, ValidationDataset
from ai.training.hardware import PyTorchHardwareManager
from ai.training.manager import TrainingManager
from ai.training.strategy import StandardTrainingStrategy


class DummyMixedPrecision(MixedPrecisionManager):
    """Simple wrapper for PyTorch AMP."""

    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    def autocast(self):
        return torch.amp.autocast(device_type=self.device_type, dtype=torch.float16)

    def scale_loss(self, loss):
        return self.scaler.scale(loss)

    def step_optimizer(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to BraTS dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 1. Configuration
    config = ExperimentConfig(
        model=ModelConfig(name="UNetBaseline", in_channels=4, out_channels=1),
        dataset=DatasetConfig(registry_id="brats20_full", batch_size=2, kwargs={"data_dir": args.data_dir}),
        optimizer=OptimizerConfig(name="AdamW", learning_rate=1e-4),
        hardware=HardwareConfig(device="cpu"), # Mock data, run on CPU to avoid CUDA initialization overhead
        max_epochs=args.epochs
    )

    # 2. Dataset Integration
    adapter = BraTSAdapter(root_dir=args.data_dir)
    studies = list(adapter.load_studies())

    split_manager = DatasetSplitManager(PatientSplitStrategy(seed=config.seed))
    train_studies, val_studies, _ = split_manager.create_splits(
        studies, train_ratio=0.8, val_ratio=0.2
    )

    pad_transform = SpatialPadd(keys=["image", "label"], spatial_size=(240, 240, 160))

    train_ds = PyTorchDatasetAdapter(TrainingDataset(train_studies), transforms=pad_transform)
    val_ds = PyTorchDatasetAdapter(ValidationDataset(val_studies), transforms=pad_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.dataset.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.dataset.batch_size, shuffle=False
    )

    # 3. Model & Strategy
    model = UNetBaseline(
        in_channels=config.model.in_channels, out_channels=config.model.out_channels
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    hardware_manager = PyTorchHardwareManager(device_type=config.hardware.device)
    mixed_precision = DummyMixedPrecision(device_type=hardware_manager.device.type)

    start_epoch = 1
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")  # noqa: T201
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

    strategy = StandardTrainingStrategy(
        optimizer=optimizer,
        mixed_precision=mixed_precision,
        hardware=hardware_manager,
        event_bus=None,  # Set later by TrainingManager
        scheduler=scheduler
    )

    # 4. Orchestrator
    manager = TrainingManager(
        config=config,
        model=model,
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        hardware_manager=hardware_manager,
    )

    # We must assign the event bus to the strategy now that it's created
    strategy.event_bus = manager.event_bus

    # 5. Callbacks
    save_dir = Path(args.output_dir) / config.logging.experiment_name

    manager.register_callback(RuntimeMonitorCallback())
    manager.register_callback(TrainingHealthCallback())

    manager.register_callback(
        CheckpointCallback(
            save_dir=str(save_dir / "checkpoints"),
            model=model,
            strategy=strategy,
            config=config,
            monitor_metric="val_dice",
            mode="max",
        )
    )

    manager.register_callback(
        ModelCardCallback(
            config=config,
            model_details={
                "model_name": config.model.name,
                "architecture": "3D U-Net",
                "description": "Baseline segmentation model for BraTS.",
                "limitations": ["Trained on small sample, for baseline purposes only."],
            },
            save_dir=str(save_dir),
        )
    )

    # 5.5 Experiment Manager Integration
    from ai.experiment_tracking.callbacks import ExperimentManagerCallback
    from ai.experiment_tracking.experiment_manager import ExperimentManager

    em = ExperimentManager(base_dir=str(Path(args.output_dir) / "experiments"))
    em_callback = ExperimentManagerCallback(
        experiment_manager=em,
        checkpoint_dir=str(save_dir / "checkpoints"),
        artifact_dir=str(save_dir)
    )
    manager.register_callback(em_callback)

    # 6. Start Training
    manager.start_training(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
