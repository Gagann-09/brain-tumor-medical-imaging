from .classification import ClassificationStage
from .explainability import ExplainabilityStage
from .loading import MedicalImageLoadingStage
from .preprocessing import PreprocessingStage
from .report import ClinicalReportGenerationStage
from .response import ResponseConstructionStage
from .roi_extraction import ROIExtractionStage
from .segmentation import SegmentationStage
from .validation import InputValidationStage

STAGE_CLASSES = {
    "InputValidationStage": InputValidationStage,
    "MedicalImageLoadingStage": MedicalImageLoadingStage,
    "PreprocessingStage": PreprocessingStage,
    "SegmentationStage": SegmentationStage,
    "ROIExtractionStage": ROIExtractionStage,
    "ClassificationStage": ClassificationStage,
    "ExplainabilityStage": ExplainabilityStage,
    "ClinicalReportGenerationStage": ClinicalReportGenerationStage,
    "ResponseConstructionStage": ResponseConstructionStage,
}
