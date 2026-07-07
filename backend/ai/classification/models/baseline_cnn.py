from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

from ai.classification.models.base import BaseClassificationModel
from ai.interfaces.features import FeatureProvider
from ai.models.base import PredictionResult, TrainingOutput, ValidationOutput


class BaselineCNNConfig(BaseModel):
    in_channels: int = 1
    num_classes: int = 2
    base_filters: int = 16
    depth: int = 3
    dropout_rate: float = 0.3


class BaselineCNNClassifier(BaseClassificationModel, FeatureProvider):
    """
    Configuration-driven baseline 3D CNN for classification.
    Implements FeatureProvider to expose intermediate feature maps.
    """

    def __init__(self, config: BaselineCNNConfig):
        super().__init__()
        self.config = config
        self._features: dict[str, torch.Tensor] = {}

        layers = []
        in_c = config.in_channels
        out_c = config.base_filters

        for _i in range(config.depth):
            layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(2))
            in_c = out_c
            out_c *= 2

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate), nn.Linear(in_c, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss() if config.num_classes > 1 else nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        self._features["last_conv"] = feat  # Store for FeatureProvider

        pooled = self.global_pool(feat).view(feat.size(0), -1)
        logits = self.classifier(pooled)
        return logits

    def get_intermediate_features(self) -> dict[str, torch.Tensor]:
        return self._features

    def get_calibration_module(self) -> Any:
        return None

    def build(self) -> None:
        pass

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.num_classes == 1:
            return self.loss_fn(logits.view(-1), target.float())
        return self.loss_fn(logits, target.long())

    def training_step(self, batch: Any, batch_idx: int) -> TrainingOutput:
        x = batch["input"]
        target = batch["labels"].get(
            "target", torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        logits = self(x)
        loss = self._compute_loss(logits, target)
        return TrainingOutput(loss=loss, metrics={"train_loss": loss.item()})

    def validation_step(self, batch: Any, batch_idx: int) -> ValidationOutput:
        x = batch["input"]
        target = batch["labels"].get(
            "target", torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        logits = self(x)
        loss = self._compute_loss(logits, target)
        return ValidationOutput(loss=loss.item(), metrics={"val_loss": loss.item()})

    def prediction_step(self, batch: Any, batch_idx: int) -> PredictionResult:
        x = batch["input"]
        probs = self.predict_proba(x)
        return PredictionResult(predictions=probs)

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return self._compute_loss(outputs, targets)

    def compute_metrics(self, outputs: Any, targets: Any) -> dict[str, float]:
        return {}

    def export(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
