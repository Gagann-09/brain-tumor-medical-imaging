from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class StageStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL_SUCCESS = "partial_success"


@dataclass(frozen=True)
class StageDiagnostics:
    stage_name: str
    status: StageStatus
    message: str = ""
    timings_ms: float = 0.0


@dataclass(frozen=True)
class InferenceContext:
    """Immutable context shared across all pipeline stages."""

    study_metadata: dict[str, Any] = field(default_factory=dict)
    mri_volumes: dict[str, Any] = field(default_factory=dict)
    preprocessing_outputs: dict[str, Any] = field(default_factory=dict)
    segmentation_results: dict[str, Any] = field(default_factory=dict)
    roi_information: dict[str, Any] = field(default_factory=dict)
    classification_results: dict[str, Any] = field(default_factory=dict)
    explainability_artifacts: dict[str, Any] = field(default_factory=dict)
    generated_reports: dict[str, Any] = field(default_factory=dict)
    timing_information: dict[str, float] = field(default_factory=dict)

    def update(self, **kwargs) -> "InferenceContext":
        """Returns a new InferenceContext with updated fields."""
        return replace(self, **kwargs)


@dataclass
class InferenceResult:
    """The final structured result returned by the pipeline for the API layer."""

    status: str
    diagnostics: dict[str, StageDiagnostics]
    segmentation: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    explainability: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    total_latency_ms: float = 0.0
