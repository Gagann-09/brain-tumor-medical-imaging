from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Dict, Any, Optional

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
    study_metadata: Dict[str, Any] = field(default_factory=dict)
    mri_volumes: Dict[str, Any] = field(default_factory=dict)
    preprocessing_outputs: Dict[str, Any] = field(default_factory=dict)
    segmentation_results: Dict[str, Any] = field(default_factory=dict)
    roi_information: Dict[str, Any] = field(default_factory=dict)
    classification_results: Dict[str, Any] = field(default_factory=dict)
    explainability_artifacts: Dict[str, Any] = field(default_factory=dict)
    generated_reports: Dict[str, Any] = field(default_factory=dict)
    timing_information: Dict[str, float] = field(default_factory=dict)
    
    def update(self, **kwargs) -> 'InferenceContext':
        """Returns a new InferenceContext with updated fields."""
        return replace(self, **kwargs)

@dataclass
class InferenceResult:
    """The final structured result returned by the pipeline for the API layer."""
    status: str
    diagnostics: Dict[str, StageDiagnostics]
    segmentation: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    explainability: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    total_latency_ms: float = 0.0
