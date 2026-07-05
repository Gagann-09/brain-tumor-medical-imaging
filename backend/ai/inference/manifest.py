from dataclasses import dataclass, field
from typing import List

@dataclass
class PipelineManifest:
    """Defines the order of stages to be executed in the inference pipeline."""
    stages: List[str] = field(default_factory=lambda: [
        "InputValidationStage",
        "MedicalImageLoadingStage",
        "PreprocessingStage",
        "SegmentationStage",
        "ROIExtractionStage",
        "ClassificationStage",
        "ExplainabilityStage",
        "ClinicalReportGenerationStage",
        "ResponseConstructionStage"
    ])
