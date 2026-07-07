import torch
import torch.nn as nn

from ai.models.base import BaseModel


class UNet3D(BaseModel):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def build(self):
        pass

    def forward(self, x, **kwargs):
        return self.conv(x)

    def training_step(self, batch, batch_idx):
        from ai.models.base import TrainingOutput

        loss = self.compute_loss(None, None)
        return TrainingOutput(loss=loss, metrics=self.compute_metrics(None, None))

    def validation_step(self, batch, batch_idx):
        from ai.models.base import ValidationOutput

        loss = self.compute_loss(None, None)
        return ValidationOutput(
            loss=loss.item(),
            metrics=self.compute_metrics(None, None),
            predictions=[
                {
                    "case_id": "mock_case",
                    "image": batch["inputs"][0, 0].numpy(),
                    "gt": batch["targets"][0, 0].numpy(),
                    "pred": batch["targets"][0, 0].numpy(),
                    "dice": 0.5,
                    "hausdorff": 1.0,
                    "pred_sum": 10,
                    "gt_sum": 10,
                }
            ],
        )

    def prediction_step(self, batch, batch_idx):
        pass

    def compute_loss(self, preds, targets):
        return torch.tensor(0.0, requires_grad=True)

    def compute_metrics(self, preds, targets):
        return {"dice": 0.5, "val_dice_mean": 0.5}

    def export(self, path):
        pass

    def load_checkpoint(self, path):
        pass
