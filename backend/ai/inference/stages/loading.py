import numpy as np

from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class MedicalImageLoadingStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # Mock loading
        vols = {"t1ce": np.zeros((10, 10, 10))}
        new_context = context.update(mri_volumes=vols)
        return new_context, StageDiagnostics(self.name, StageStatus.SUCCESS, "Images loaded.")
