from typing import Any

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from ai.models.base import BaseSegmentationModel, PredictionResult, TrainingOutput, ValidationOutput


class UNetBaseline(BaseSegmentationModel):
    """
    Baseline 3D U-Net for Tumor Segmentation using MONAI.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, **kwargs: Any):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.build()

    def build(self) -> None:
        """Constructs the MONAI U-Net."""
        self.net = UNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            **self.kwargs,
        )
        self.loss_function = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.metric_function = DiceMetric(include_background=False, reduction="mean")

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.net(inputs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> TrainingOutput:
        inputs = batch["image"]
        targets = batch["label"]

        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)

        # Optionally compute metrics during training, but often skipped for speed.
        # We'll just return loss here.
        return TrainingOutput(loss=loss, metrics={"train_loss": loss.item()}, predictions=outputs)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> ValidationOutput:
        inputs = batch["image"]
        targets = batch["label"]

        outputs = self(inputs)
        loss = self.compute_loss(outputs, targets)
        metrics = self.compute_metrics(outputs, targets)

        return ValidationOutput(
            loss=loss.item(), metrics={"val_loss": loss.item(), **metrics}, predictions=outputs
        )

    def prediction_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> PredictionResult:
        inputs = batch["image"]
        outputs = self(inputs)
        # Apply sigmoid and thresholding for binary masks (assuming sigmoid=True in DiceLoss)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return PredictionResult(predictions=preds)

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_function(preds, targets)

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        # Discretize preds and targets for metric computation
        preds_bin = (torch.sigmoid(preds) > 0.5).float()
        self.metric_function(y_pred=preds_bin, y=targets)
        # We aggregate the metric over the batch
        metric = self.metric_function.aggregate().item()
        self.metric_function.reset()
        return {"val_dice": metric}

    def export(self, path: str) -> None:
        """Export model via TorchScript."""
        scripted_model = torch.jit.script(self.net)
        scripted_model.save(path)

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        if "model_state" in state_dict:
            self.load_state_dict(state_dict["model_state"])
        else:
            self.load_state_dict(state_dict)
