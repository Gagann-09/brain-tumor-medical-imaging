from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class ResponseConstructionStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # This stage doesn't do much, the orchestrator constructs the InferenceResult from the context
        return context, StageDiagnostics(
            self.name,
            StageStatus.SUCCESS,
            "Response construction skipped as it is handled by Orchestrator.",
        )
