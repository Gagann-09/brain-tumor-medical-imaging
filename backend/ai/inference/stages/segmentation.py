import numpy as np

from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class SegmentationStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # Mock segmentation
        mask = np.ones((10, 10, 10))
        new_context = context.update(segmentation_results={"mask": mask})
        return new_context, StageDiagnostics(
            self.name, StageStatus.SUCCESS, "Segmentation complete."
        )
