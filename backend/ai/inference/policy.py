from dataclasses import dataclass
from enum import Enum

class ExecutionMode(Enum):
    RESEARCH = "research"
    CLINICAL_REVIEW = "clinical_review"
    PRODUCTION = "production"

@dataclass
class InferencePolicy:
    """Configures the pipeline based on the execution mode."""
    mode: ExecutionMode
    confidence_threshold: float = 0.8
    generate_explainability: bool = True
    generate_clinical_report: bool = True
    fail_on_missing_metadata: bool = True

    @classmethod
    def get_default_policy(cls, mode: ExecutionMode) -> 'InferencePolicy':
        if mode == ExecutionMode.RESEARCH:
            return cls(mode=mode, confidence_threshold=0.5, fail_on_missing_metadata=False)
        elif mode == ExecutionMode.CLINICAL_REVIEW:
            return cls(mode=mode, confidence_threshold=0.85, fail_on_missing_metadata=True)
        else:
            return cls(mode=mode, confidence_threshold=0.9, generate_explainability=False, generate_clinical_report=False)
