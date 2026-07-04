import abc
import torch
import torch.nn as nn
from typing import Dict, Any, List
from pydantic import BaseModel

from ai.classification.models.base import BaseClassificationModel
from ai.interfaces.features import FeatureProvider
from ai.models.base import TrainingOutput, ValidationOutput, PredictionResult

class ModalityConfig(BaseModel):
    """Configuration for MRI Modality combinations."""
    modalities: List[str] = ["t1", "t1ce", "t2", "flair"]
    
    @property
    def in_channels(self) -> int:
        return len(self.modalities)

class FeatureFusionStrategy(nn.Module, abc.ABC):
    """Abstract strategy for fusing multimodal features."""
    @abc.abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pass

class ConcatenationFusion(FeatureFusionStrategy):
    """Simple concatenation of feature maps along the channel dimension."""
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(features, dim=1)

class HybridCNNConfig(BaseModel):
    modality_config: ModalityConfig = ModalityConfig()
    num_classes: int = 2
    base_filters: int = 16
    depth: int = 4
    dropout_rate: float = 0.4

class HybridCNNClassifier(BaseClassificationModel, FeatureProvider):
    """
    Advanced Hybrid CNN Classification architecture supporting configurable 
    modalities and feature fusion strategies.
    """
    def __init__(self, config: HybridCNNConfig, fusion_strategy: FeatureFusionStrategy = ConcatenationFusion()):
        super().__init__()
        self.config = config
        self.fusion_strategy = fusion_strategy
        self._features: Dict[str, torch.Tensor] = {}
        
        # Single branch for all modalities (since concatenation happens either early or late).
        # For simplicity in this implementation, we assume early fusion (concatenating input channels).
        # If late fusion is desired, the architecture would instantiate multiple branches.
        # We will use early fusion but keep the FusionStrategy for extensibility.
        
        in_c = config.modality_config.in_channels
        out_c = config.base_filters
        
        layers = []
        for i in range(config.depth):
            layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(2))
            in_c = out_c
            out_c *= 2
            
        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(in_c, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss() if config.num_classes > 1 else nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Currently early fusion simply passes the multi-channel input
        # Advanced fusion strategies would split channels and fuse them here.
        fused_x = self.fusion_strategy([x[:, i:i+1] for i in range(x.shape[1])])
        
        feat = self.encoder(fused_x)
        self._features["last_conv"] = feat # Store for FeatureProvider
        
        pooled = self.global_pool(feat).view(feat.size(0), -1)
        logits = self.classifier(pooled)
        return logits

    def get_intermediate_features(self) -> Dict[str, torch.Tensor]:
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
        target = batch["labels"].get("target", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        logits = self(x)
        loss = self._compute_loss(logits, target)
        return TrainingOutput(loss=loss, metrics={"train_loss": loss.item()})

    def validation_step(self, batch: Any, batch_idx: int) -> ValidationOutput:
        x = batch["input"]
        target = batch["labels"].get("target", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
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
