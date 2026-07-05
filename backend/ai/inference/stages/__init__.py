from .base import BaseStage
from .validation import InputValidationStage
from .loading import MedicalImageLoadingStage
from .preprocessing import PreprocessingStage
from .segmentation import SegmentationStage
from .roi_extraction import ROIExtractionStage
from .classification import ClassificationStage
from .explainability import ExplainabilityStage
from .report import ClinicalReportGenerationStage
from .response import ResponseConstructionStage

STAGE_CLASSES = {
    "InputValidationStage": InputValidationStage,
    "MedicalImageLoadingStage": MedicalImageLoadingStage,
    "PreprocessingStage": PreprocessingStage,
    "SegmentationStage": SegmentationStage,
    "ROIExtractionStage": ROIExtractionStage,
    "ClassificationStage": ClassificationStage,
    "ExplainabilityStage": ExplainabilityStage,
    "ClinicalReportGenerationStage": ClinicalReportGenerationStage,
    "ResponseConstructionStage": ResponseConstructionStage
}
