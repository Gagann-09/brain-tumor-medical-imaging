import abc
import time

from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus


class BaseStage(abc.ABC):
    """Abstract base class for all inference stages."""

    def __init__(self, name: str):
        self.name = name

    def run(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        """Wrapper to record timing and handle execution."""
        start_time = time.time()
        try:
            new_context, diagnostics = self.execute(context)
            # Update diagnostics with timings
            timings_ms = (time.time() - start_time) * 1000
            new_diagnostics = StageDiagnostics(
                stage_name=self.name,
                status=diagnostics.status,
                message=diagnostics.message,
                timings_ms=timings_ms,
            )
            # Record timing in context
            new_timings = dict(new_context.timing_information)
            new_timings[self.name] = timings_ms
            new_context = new_context.update(timing_information=new_timings)
            return new_context, new_diagnostics
        except Exception as e:
            timings_ms = (time.time() - start_time) * 1000
            diagnostics = StageDiagnostics(
                stage_name=self.name,
                status=StageStatus.FAILED,
                message=f"Unhandled exception: {e!s}",
                timings_ms=timings_ms,
            )
            return context, diagnostics

    @abc.abstractmethod
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        """Core logic of the stage. Returns a new immutable context and diagnostics."""
        pass
