import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from ai.config.profiles import DatasetProfileName, get_profile
from ai.config.training_config import ExperimentConfig
from ai.dataset_registry.registry import DatasetRegistry
from ai.models.unet import UNet3D
from ai.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ModelCardCallback,
)
from ai.training.data import DataLoaderManager, PyTorchDatasetAdapter
from ai.training.manager import TrainingManager
from ai.training.strategy import StandardTrainingStrategy


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train U-Net Baseline")
    parser.add_argument(
        "--profile", type=str, default="development", choices=["development", "research"]
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override profile epochs (req for research)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from scripts.validate_dataset import register_datasets

    register_datasets()

    is_dev = args.profile == "development"
    epochs = args.epochs if args.epochs is not None else (10 if is_dev else 50)

    config = ExperimentConfig(
        seed=42,
        max_epochs=epochs,
        hardware={"device_ids": [0]},
        model={"name": "unet_baseline", "in_channels": 4, "out_channels": 3},
        dataset={"registry_id": "brats2020"},
        logging={"experiment_name": f"unet_baseline_{args.profile}"},
    )

    ds_profile = get_profile(
        DatasetProfileName.DEVELOPMENT if is_dev else DatasetProfileName.RESEARCH
    )

    reg = DatasetRegistry.get("brats2020")
    adapter = reg.adapter_class(profile=ds_profile)
    torch_dataset = PyTorchDatasetAdapter(adapter)

    dl_manager = DataLoaderManager(profile=ds_profile)

    train_size = int(0.8 * len(torch_dataset))
    val_size = len(torch_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        torch_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = dl_manager.get_dataloader(train_ds, batch_size=2, shuffle=True)
    val_loader = dl_manager.get_dataloader(val_ds, batch_size=2, shuffle=False)

    class DummyHardwareManager:
        def empty_cache(self):
            pass

        def set_device(self, x):
            pass

        def to_device(self, t):
            return t

    hardware = DummyHardwareManager()
    model = UNet3D(in_channels=4, out_channels=3)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    class DummyMixedPrecision:
        def autocast(self):
            from contextlib import nullcontext

            if torch.cuda.is_available():
                return torch.amp.autocast("cuda")
            return nullcontext()

        def scale_loss(self, loss):
            return loss

        def unscale_gradients(self, opt):
            pass

        def step_optimizer(self, opt):
            opt.step()

    strategy = StandardTrainingStrategy(
        optimizer=optimizer,
        mixed_precision=DummyMixedPrecision(),
        hardware=hardware,
        event_bus=None,
    )

    manager = TrainingManager(
        config=config,
        model=model,
        strategy=strategy,
        train_loader=train_loader,
        val_loader=val_loader,
        hardware_manager=hardware,
    )

    strategy.event_bus = manager.event_bus

    manager.register_callback(EarlyStoppingCallback(patience=5, monitor_metric="val_dice_mean"))

    manager.register_callback(
        CheckpointCallback(
            save_dir=f"outputs/{config.logging.experiment_name}/checkpoints",
            model=model,
            strategy=strategy,
            config=config,
            monitor_metric="val_dice_mean",
            mode="max",
            keep_top_k=3,
        )
    )

    from ai.models.model_card import ModelCardConfig

    mc_config = ModelCardConfig(
        model_name="U-Net Baseline",
        architecture="UNet3D",
        description="Frozen baseline for BraTS 2020",
        author="ARMT-GAN AI System",
    )
    manager.register_callback(
        ModelCardCallback(
            config=config,
            model_details=mc_config,
            save_dir=f"outputs/{config.logging.experiment_name}",
        )
    )

    from ai.training.visualization import ValidationVisualizationCallback

    manager.register_callback(
        ValidationVisualizationCallback(
            output_dir=f"outputs/{config.logging.experiment_name}", num_cases=5, seed=config.seed
        )
    )

    from ai.evaluation.failure_analysis import FailureAnalyzerCallback

    manager.register_callback(
        FailureAnalyzerCallback(output_dir=f"outputs/{config.logging.experiment_name}")
    )

    print(f"Starting {args.profile} training for {epochs} epochs...")
    manager.start_training()
    print("Training finished. Baseline frozen.")


if __name__ == "__main__":
    main()
