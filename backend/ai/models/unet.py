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
        image_np = None
        label_np = None
        if isinstance(batch, dict):
            if "image" in batch:
                image_np = batch["image"][0, 0].numpy()
            elif "images" in batch:
                image_np = batch["images"][0, 0].numpy()
            if "label" in batch:
                label_np = batch["label"][0, 0].numpy()
            elif "mask" in batch:
                label_np = batch["mask"][0, 0].numpy()
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            image_np = batch[0][0, 0].numpy()
            label_np = batch[1][0, 0].numpy()

        return ValidationOutput(
            loss=loss.item(),
            metrics=self.compute_metrics(None, None),
            predictions=[
                {
                    "case_id": "mock_case",
                    "image": image_np if image_np is not None else [],
                    "gt": label_np if label_np is not None else [],
                    "pred": label_np if label_np is not None else [],
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
