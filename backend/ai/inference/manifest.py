from dataclasses import dataclass, field


@dataclass
class PipelineManifest:
    """Defines the order of stages to be executed in the inference pipeline."""

    stages: list[str] = field(
        default_factory=lambda: [
            "InputValidationStage",
            "MedicalImageLoadingStage",
            "PreprocessingStage",
            "SegmentationStage",
            "ROIExtractionStage",
            "ClassificationStage",
            "ExplainabilityStage",
            "ClinicalReportGenerationStage",
            "ResponseConstructionStage",
        ]
    )
