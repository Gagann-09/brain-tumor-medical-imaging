import torch
from typing import Dict, Any, Optional

from ai.classification.models.base import BaseClassificationModel
from ai.classification.pipeline.input_pipeline import ClassificationInputPipeline
from ai.classification.datasets.sample import ClassificationSample
from ai.classification.calibration.calibrator import BaseCalibrator

class ClassificationInference:
    """
    Orchestrates classification inference, bridging the raw input, the input pipeline (ROI extraction),
    the model, and optional confidence calibration.
    """
    def __init__(
        self,
        model: BaseClassificationModel,
        input_pipeline: ClassificationInputPipeline,
        calibrator: Optional[BaseCalibrator] = None
    ):
        self.model = model
        self.input_pipeline = input_pipeline
        self.calibrator = calibrator
        self.model.eval()
        
    @torch.no_grad()
    def __call__(self, sample: ClassificationSample) -> Dict[str, Any]:
        """
        Executes inference on a single ClassificationSample.
        """
        # 1. Pipeline execution (ROI cropping, transforms)
        processed_data = self.input_pipeline(sample)
        x = processed_data["input"].unsqueeze(0) # Add batch dimension (1, C, D, H, W)
        
        # 2. Model Forward Pass
        logits = self.model.forward(x)
        
        # 3. Probabilities and Calibration
        if self.calibrator:
            probs = self.calibrator.calibrate(logits)
        else:
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
                
        # 4. Result Formatting
        preds = torch.argmax(probs, dim=-1) if probs.shape[-1] > 1 else (probs >= 0.5).long()
        
        return {
            "sample_id": sample.sample_id,
            "logits": logits.squeeze(0).cpu(),
            "probabilities": probs.squeeze(0).cpu(),
            "prediction": preds.squeeze(0).cpu().item(),
            "metadata": sample.metadata
        }
