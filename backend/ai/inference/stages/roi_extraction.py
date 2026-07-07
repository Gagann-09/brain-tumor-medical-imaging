from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class ROIExtractionStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # Mock ROI extraction
        roi = {"center": (5, 5, 5), "bbox": (0, 10, 0, 10, 0, 10)}
        new_context = context.update(roi_information=roi)
        return new_context, StageDiagnostics(
            self.name, StageStatus.SUCCESS, "ROI extraction complete."
        )
