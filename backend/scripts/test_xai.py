import os

# Add backend directory to path if running directly
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.interfaces.features import FeatureMap, FeatureProvider
from ai.xai.shared import (
    ExplainabilityReport,
    ExplainabilitySession,
    ExplainerRegistry,
    OverlayGenerator,
    OverlayLayers,
    XAIBenchmarker,
)


class DummyModel(nn.Module, FeatureProvider):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 2)
        self.last_features = None

    def forward(self, x):
        features = self.conv(x)
        self.last_features = features
        x = features.view(x.size(0), -1)
        return self.fc(x)

    def get_intermediate_features(self):
        return {
            "conv": FeatureMap(
                tensor=self.last_features, metadata={"channels": 16, "resolution": "32x32"}
            )
        }


def run_test():
    model = DummyModel()

    # 1. Instantiate Explainer
    gradcam_cls = ExplainerRegistry.get_explainer_class("grad_cam")
    explainer = gradcam_cls(model=model, target_layer="conv", device=torch.device("cpu"))

    # Dummy input (batch_size=1, channels=1, H=32, W=32)
    input_tensor = torch.randn(1, 1, 32, 32)

    # 2. Benchmark explanation generation
    heatmap, benchmark = XAIBenchmarker.run(
        explainer.generate_explanation, input_tensor=input_tensor, target_class=1
    )

    # 3. Overlays
    mri_dummy = input_tensor.squeeze().detach().numpy()
    mask_dummy = np.zeros_like(mri_dummy)
    mask_dummy[10:20, 10:20] = 1  # Dummy square mask

    layers = OverlayLayers(
        mri=mri_dummy,
        heatmap=heatmap,
        mask=mask_dummy,
        metadata={"Patient": "001", "Prediction": "Tumor"},
    )

    overlay = OverlayGenerator.render_2d_with_metadata(layers)

    # 4. Export Artifacts
    output_dir = Path("test_xai_output")
    output_dir.mkdir(exist_ok=True)

    png_path = output_dir / "overlay.png"
    OverlayGenerator.export_png(overlay, str(png_path))

    # 5. Report & Session
    report = ExplainabilityReport(
        prediction={"class_id": 1, "label": "Tumor"},
        confidence=0.85,
        model_version="v1.0",
        dataset_version="v1.0",
        explanation_method="grad_cam",
        feature_layer="conv",
        generated_artifacts={"overlay_png": str(png_path)},
        benchmark=benchmark,
        software_versions={"torch": torch.__version__, "numpy": np.__version__},
    )

    session = ExplainabilitySession(
        session_id="test_001",
        original_input=input_tensor,
        prediction_result={"class_id": 1, "label": "Tumor"},
        artifacts={"overlay": overlay},
        report=report,
    )

    session.export(str(output_dir))

    print("XAI Test completed successfully. Outputs in test_xai_output/")


if __name__ == "__main__":
    run_test()
