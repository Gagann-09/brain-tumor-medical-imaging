import argparse
from pathlib import Path

import torch

# Platform Core
from ai.config.training_config import (
    DatasetConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    OptimizerConfig,
)
from ai.datasets.adapters.pytorch import PyTorchDatasetAdapter
from ai.models.model_card import ModelCard, ModelCardConfig
from ai.models.promotion import ModelPromotionManager
from ai.pipeline.evaluate_models import run_comparison
from ai.pipeline.profiler import ExperimentProfiler
from ai.pipeline.visualize_comparison import VisualizationManager
from ai.segmentation.models.armt_gan import ARMTGANModel
from ai.segmentation.models.unet import UNetBaseline
from ai.training.components import AdversarialLossManager, MultiOptimizerManager
from ai.training.data import TrainingDataset, ValidationDataset
from ai.training.hardware import PyTorchHardwareManager
from ai.training.manager import TrainingManager
from ai.training.strategy import (
    AlternatingUpdatePolicy,
    GANTrainingStrategy,
    StandardTrainingStrategy,
)
from medical.domain import MRIImage, MRIStudy, SegmentationAnnotation


def generate_mock_dataset(num_samples=10):
    """Generates a tiny synthetic dataset for mock validation to prove infrastructure end-to-end."""
    studies = []
    for i in range(num_samples):
        import numpy as np

        # Attach dummy image numpy arrays
        volumes = {
            "T1": np.random.rand(64, 64, 64).astype(np.float32),
            "T1ce": np.random.rand(64, 64, 64).astype(np.float32),
            "T2": np.random.rand(64, 64, 64).astype(np.float32),
            "FLAIR": np.random.rand(64, 64, 64).astype(np.float32),
        }
        primary_image = MRIImage(volumes=volumes)

        # Attach dummy mask numpy array
        mask_array = (np.random.rand(64, 64, 64) > 0.8).astype(np.float32)
        annotation = SegmentationAnnotation(mask=mask_array, label_map={"tumor": 1})

        study = MRIStudy(
            primary_image=primary_image,
            study_id=f"mock_{i}",
            patient_id=f"patient_{i}",
            annotations=[annotation],
        )

        studies.append(study)
    return studies


def get_reproducibility_meta(args):
    return {
        "profile": args.profile,
        "seed": 42,
        "git_commit": "unknown",  # would fetch dynamically
        "dataset_manifest": "mock_manifest" if args.profile == "mock" else "brats_manifest",
        "pytorch_version": torch.__version__,
    }


def create_model_and_strategy(model_name: str, config: ExperimentConfig, hardware_manager):
    if model_name == "UNetBaseline":
        model = UNetBaseline(
            in_channels=config.model.in_channels, out_channels=config.model.out_channels
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

        # Assuming a dummy mixed precision for now
        class DummyMP:
            def autocast(self):
                return torch.amp.autocast(device_type="cpu", dtype=torch.float32)

            def scale_loss(self, loss):
                return loss

            def step_optimizer(self, opt):
                opt.step()

        strategy = StandardTrainingStrategy(optimizer, DummyMP(), hardware_manager, None)
        return model, strategy

    elif model_name == "ARMTGANModel":

        def g_bce(fake_preds, **kwargs):
            if isinstance(fake_preds, (list, tuple)):
                fake_preds = fake_preds[-1]
            return torch.nn.functional.binary_cross_entropy_with_logits(
                fake_preds, torch.ones_like(fake_preds)
            )

        def g_l1(fake_masks, real_masks, **kwargs):
            return torch.nn.functional.l1_loss(fake_masks, real_masks)

        def d_bce(real_preds, fake_preds, **kwargs):
            if isinstance(real_preds, (list, tuple)):
                real_preds = real_preds[-1]
            if isinstance(fake_preds, (list, tuple)):
                fake_preds = fake_preds[-1]
            real = torch.nn.functional.binary_cross_entropy_with_logits(
                real_preds, torch.ones_like(real_preds)
            )
            fake = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_preds, torch.zeros_like(fake_preds)
            )
            return (real + fake) / 2.0

        loss_mgr = AdversarialLossManager(
            g_losses={"bce": (g_bce, 1.0), "l1": (g_l1, 10.0)}, d_losses={"bce": (d_bce, 1.0)}
        )
        model = ARMTGANModel(
            loss_manager=loss_mgr,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
        )
        opt_g = torch.optim.Adam(model.generator.parameters(), lr=config.optimizer.learning_rate)
        opt_d = torch.optim.Adam(
            model.discriminator.parameters(), lr=config.optimizer.learning_rate
        )
        multi_opt = MultiOptimizerManager({"generator": opt_g, "discriminator": opt_d})

        class DummyMP:
            def autocast(self):
                return torch.amp.autocast(device_type="cpu", dtype=torch.float32)

            def scale_loss(self, loss):
                return loss

            def step_optimizer(self, opt):
                opt.step()

        strategy = GANTrainingStrategy(
            multi_opt, DummyMP(), hardware_manager, None, AlternatingUpdatePolicy()
        )
        return model, strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        choices=["mock", "development", "research", "production"],
        default="mock",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs_val")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Unified configuration logic based on profile
    if args.profile == "mock":
        epochs = 1
        batch_size = 2
        studies = generate_mock_dataset(num_samples=10)
        device = "cpu"
    else:
        # Here we would load actual BraTS data based on profile
        # e.g., development=100 samples, research=1000 samples, production=all
        epochs = 1
        batch_size = 2
        studies = generate_mock_dataset(num_samples=10)
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hardware = PyTorchHardwareManager(device_type=device)

    profiler_mode = "research" if args.profile == "research" else "development"
    profiler = ExperimentProfiler(mode=profiler_mode, output_dir=str(output_dir / "profiler"))

    train_studies = studies[:6]
    val_studies = studies[6:8]
    test_studies = studies[8:]

    train_loader = torch.utils.data.DataLoader(
        PyTorchDatasetAdapter(TrainingDataset(train_studies)), batch_size=batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        PyTorchDatasetAdapter(ValidationDataset(val_studies)), batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        PyTorchDatasetAdapter(ValidationDataset(test_studies)), batch_size=1
    )

    output_dir = Path(args.output_dir)
    registry_dir = output_dir / "registries"

    # Promotion Manager
    promo_manager = ModelPromotionManager(str(registry_dir))
    promo_manager.register_model("UNetBaseline", "v1.0")
    promo_manager.register_model("ARMTGANModel", "v1.0")

    # 1. Train U-Net
    cfg_unet = ExperimentConfig(
        model=ModelConfig(name="UNetBaseline", in_channels=4, out_channels=1),
        dataset=DatasetConfig(name="BraTS", data_dir="", batch_size=batch_size),
        optimizer=OptimizerConfig(name="Adam", learning_rate=1e-3),
        hardware=HardwareConfig(device=device),
        experiment_name="unet_base",
        max_epochs=epochs,
    )
    unet_model, unet_strategy = create_model_and_strategy("UNetBaseline", cfg_unet, hardware)
    mgr_unet = TrainingManager(
        cfg_unet, unet_model, unet_strategy, train_loader, val_loader, hardware
    )
    unet_strategy.event_bus = mgr_unet.event_bus

    # Simple manual hooks for profiler during mock run
    profiler.start_training()
    profiler.start_epoch(1)
    mgr_unet.start_training()
    profiler.end_epoch(1, len(train_studies))
    profiler.stop_training()

    # 2. Train ARMT-GAN
    cfg_armt = ExperimentConfig(
        model=ModelConfig(name="ARMTGANModel", in_channels=4, out_channels=1),
        dataset=DatasetConfig(name="BraTS", data_dir="", batch_size=batch_size),
        optimizer=OptimizerConfig(name="Adam", learning_rate=2e-4),
        hardware=HardwareConfig(device=device),
        experiment_name="armt_gan",
        max_epochs=epochs,
    )
    armt_model, armt_strategy = create_model_and_strategy("ARMTGANModel", cfg_armt, hardware)
    mgr_armt = TrainingManager(
        cfg_armt, armt_model, armt_strategy, train_loader, val_loader, hardware
    )
    armt_strategy.event_bus = mgr_armt.event_bus

    profiler.start_training()
    profiler.start_epoch(1)
    mgr_armt.start_training()
    profiler.end_epoch(1, len(train_studies))
    profiler.stop_training()
    profiler.save_report("performance_report.json")

    # 3. Benchmark
    repro_meta = get_reproducibility_meta(args)
    record, unet_res, armt_res = run_comparison(
        unet_model, armt_model, test_loader, str(registry_dir), repro_meta=repro_meta
    )

    # 4. Visualization Export
    vis_manager = VisualizationManager(str(output_dir / "visualizations"))

    unet_model.eval()
    armt_model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            img = batch["image"].to(hardware.device)
            gt = batch["label"].to(hardware.device)

            unet_pred = unet_model(img)
            armt_pred = armt_model(img)

            unet_pred = (torch.sigmoid(unet_pred) > 0.5).float()
            armt_pred = (torch.sigmoid(armt_pred) > 0.5).float()

            vis_manager.generate_comparison_grid(img, gt, unet_pred, armt_pred, sample_idx=idx)

    # 5. Evaluate Promotion
    promo_manager.mark_as_candidate("ARMTGANModel", "v1.0")
    success = promo_manager.evaluate_promotion(
        "ARMTGANModel", "v1.0", unet_res, armt_res, record.benchmark_id
    )

    if success:
        pass
    else:
        pass

    mc_config = ModelCardConfig(
        model_name="ARMTGANModel",
        architecture="ARMT-GAN (U-Net Generator, PatchGAN Discriminator)",
        description="Adversarially Robust Medical Tumor Generative Adversarial Network for BraTS segmentation.",
    )
    mc = ModelCard(
        model_details=mc_config,
        training_config=cfg_armt.model_dump(mode="json"),
        configuration_checksum="simulated_checksum_123",
        dataset_info={"name": "BraTS Mock" if args.profile == "mock" else "BraTS"},
        dataset_version="v1.0",
        metrics=armt_res,
        benchmark_summary=record.model_dump(mode="json"),
        software_versions={"pytorch": torch.__version__},
        hardware_information=device,
    )
    mc.save(registry_dir / "armt_gan_model_card.json")
    mc.export_markdown(registry_dir / "armt_gan_model_card.md")


if __name__ == "__main__":
    main()
