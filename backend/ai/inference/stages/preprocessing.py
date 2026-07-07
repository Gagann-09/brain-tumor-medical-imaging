from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class PreprocessingStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        preprocessed = {k: v / 255.0 for k, v in context.mri_volumes.items()}
        new_context = context.update(preprocessing_outputs=preprocessed)
        return new_context, StageDiagnostics(
            self.name, StageStatus.SUCCESS, "Preprocessing complete."
        )
