from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class ClassificationStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # Mock classification
        clf = {"tumor_type": "glioblastoma", "confidence": 0.95}
        new_context = context.update(classification_results=clf)
        return new_context, StageDiagnostics(
            self.name, StageStatus.SUCCESS, "Classification complete."
        )
