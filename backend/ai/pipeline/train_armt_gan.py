import argparse
from pathlib import Path

import torch

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
from ai.segmentation.models.armt_gan import ARMTGANModel
from ai.training.callbacks import CheckpointCallback
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
        return torch.amp.autocast(device_type=self.device_type, dtype=torch.float16)

    def scale_loss(self, loss):
        return self.scaler.scale(loss)

    def step_optimizer(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    config = ExperimentConfig(
        model=ModelConfig(name="ARMTGANModel", in_channels=4, out_channels=1),
        dataset=DatasetConfig(name="BraTS", data_dir=args.data_dir, batch_size=2),
        optimizer=OptimizerConfig(name="AdamW", learning_rate=2e-4),
        hardware=HardwareConfig(device="cuda"),
        experiment_name="brats_armt_gan"
    )

    # Dataset
    adapter = BraTSAdapter(root_dir=config.dataset.data_dir)
    studies = list(adapter.load_studies())

    split_manager = DatasetSplitManager(PatientSplitStrategy(seed=config.seed))
    train_studies, val_studies, _ = split_manager.create_splits(studies, train_ratio=0.8, val_ratio=0.2)

    train_loader = torch.utils.data.DataLoader(PyTorchDatasetAdapter(TrainingDataset(train_studies)), batch_size=config.dataset.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(PyTorchDatasetAdapter(ValidationDataset(val_studies)), batch_size=config.dataset.batch_size, shuffle=False)

    # Loss Manager
    def g_bce(fake_preds, **kwargs):
        import torch.nn.functional as F
        return F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))

    def g_l1(fake_masks, real_masks, **kwargs):
        import torch.nn.functional as F
        return F.l1_loss(fake_masks, real_masks)

    def d_bce(real_preds, fake_preds, **kwargs):
        import torch.nn.functional as F
        real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
        return (real_loss + fake_loss) / 2.0

    loss_mgr = AdversarialLossManager(
        g_losses={"bce": (g_bce, 1.0), "l1": (g_l1, 10.0)},
        d_losses={"bce": (d_bce, 1.0)}
    )

    # Model
    model = ARMTGANModel(loss_manager=loss_mgr)

    # Optimizers
    opt_g = torch.optim.AdamW(model.generator.parameters(), lr=config.optimizer.learning_rate)
    opt_d = torch.optim.AdamW(model.discriminator.parameters(), lr=config.optimizer.learning_rate)
    multi_opt = MultiOptimizerManager({"generator": opt_g, "discriminator": opt_d})

    hardware_manager = PyTorchHardwareManager(device_type=config.hardware.device)
    mixed_precision = DummyMixedPrecision(device_type=hardware_manager.device.type)

    # Strategy
    strategy = GANTrainingStrategy(
        optimizer=multi_opt,
        mixed_precision=mixed_precision,
        hardware=hardware_manager,
        event_bus=None,
        update_policy=AlternatingUpdatePolicy()
    )

    manager = TrainingManager(
        config=config,
        model=model,
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        hardware_manager=hardware_manager
    )
    strategy.event_bus = manager.event_bus

    save_dir = Path(args.output_dir) / config.experiment_name
    manager.register_callback(CheckpointCallback(
        save_dir=str(save_dir / "checkpoints"),
        model=model,
        strategy=strategy,
        config=config,
        monitor_metric="val_dice",
        mode="max"
    ))

    manager.start_training()
    print("ARMT-GAN training finished.")

if __name__ == "__main__":
    main()
