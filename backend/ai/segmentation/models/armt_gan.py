from typing import Any

import torch
import torch.nn as nn
from monai.networks.nets import PatchDiscriminator, UNet

from ai.models.base import BaseGAN, PredictionResult, TrainingOutput, ValidationOutput
from ai.models.gan_registry import GANComponentRegistry
from ai.training.components import AdversarialLossManager


# Register components in the registry (dummy implementations for demonstration)
@GANComponentRegistry.register_generator("armt_unet")
def create_armt_generator(in_channels: int = 4, out_channels: int = 1) -> nn.Module:
    # In a full implementation, this would be the actual Attention-Residual U-Net
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


@GANComponentRegistry.register_discriminator("patch_gan_3d")
def create_patch_discriminator(in_channels: int = 5) -> nn.Module:
    # 4 image channels + 1 mask channel = 5
    return PatchDiscriminator(
        spatial_dims=3,
        in_channels=in_channels,
        num_layers_d=3,
        channels=32,
    )


class ARMTGANModel(BaseGAN):
    """
    ARMT-GAN concrete model implementing BaseGAN.
    """

    def __init__(self, loss_manager: AdversarialLossManager, **kwargs: Any):
        super().__init__()
        self.loss_manager = loss_manager
        self.kwargs = kwargs
        self.build()

    def build(self) -> None:
        in_c = self.kwargs.get("in_channels", 4)
        out_c = self.kwargs.get("out_channels", 1)
        self.generator = GANComponentRegistry.get_generator(
            "armt_unet", in_channels=in_c, out_channels=out_c
        )
        self.discriminator = GANComponentRegistry.get_discriminator(
            "patch_gan_3d", in_channels=in_c + out_c
        )

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Forward pass through generator for inference
        return self.generator(inputs)

    def generator_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> TrainingOutput:
        images = batch["image"]
        targets = batch["label"]

        # 1. Generate fake masks
        fake_masks = self.generator(images)

        # 2. Score with discriminator (concatenate images and masks)
        fake_concat = torch.cat([images, fake_masks], dim=1)
        fake_preds = self.discriminator(fake_concat)

        # 3. Compute loss
        loss, components = self.loss_manager.compute_generator_loss(
            fake_preds=fake_preds, fake_masks=fake_masks, real_masks=targets
        )

        return TrainingOutput(
            loss=loss, metrics={"g_loss": loss.item(), **components}, predictions=fake_masks
        )

    def discriminator_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> TrainingOutput:
        images = batch["image"]
        targets = batch["label"]

        # 1. Generate fake masks (detached so we don't backward through G)
        with torch.no_grad():
            fake_masks = self.generator(images).detach()

        # 2. Score real and fake
        real_concat = torch.cat([images, targets], dim=1)
        fake_concat = torch.cat([images, fake_masks], dim=1)

        real_preds = self.discriminator(real_concat)
        fake_preds = self.discriminator(fake_concat)

        # 3. Compute loss
        loss, components = self.loss_manager.compute_discriminator_loss(
            real_preds=real_preds, fake_preds=fake_preds
        )

        return TrainingOutput(loss=loss, metrics={"d_loss": loss.item(), **components})

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> ValidationOutput:
        images = batch["image"]
        targets = batch["label"]

        fake_masks = self.generator(images)

        # For validation, we typically just care about generator loss or segmentation metrics
        loss, _components = self.loss_manager.compute_generator_loss(
            fake_preds=self.discriminator(torch.cat([images, fake_masks], dim=1)),
            fake_masks=fake_masks,
            real_masks=targets,
        )

        # Compute val dice roughly
        dice = (2 * (fake_masks * targets).sum()) / ((fake_masks + targets).sum() + 1e-8)

        return ValidationOutput(
            loss=loss.item(),
            metrics={"val_loss": loss.item(), "val_dice": dice.item()},
            predictions=fake_masks,
        )

    def prediction_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> PredictionResult:
        images = batch["image"]
        preds = self.generator(images)
        preds_bin = (torch.sigmoid(preds) > 0.5).float()
        return PredictionResult(predictions=preds_bin)

    def compute_loss(self, preds: Any, targets: Any) -> torch.Tensor:
        # Required by BaseModel, but unused in BaseGAN if we use explicit steps
        return torch.tensor(0.0, device=preds.device)

    def compute_metrics(self, preds: Any, targets: Any) -> dict[str, float]:
        return {}

    def export(self, path: str) -> None:
        scripted = torch.jit.script(self.generator)
        scripted.save(path)

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        if "model_state" in state_dict:
            self.load_state_dict(state_dict["model_state"])
        else:
            self.load_state_dict(state_dict)
