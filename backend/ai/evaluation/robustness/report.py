import json
from dataclasses import asdict, dataclass, field


@dataclass
class GeneralizationReport:
    """Summarizes all generalization metrics and experiment health."""

    experiment_id: str
    overfitting_status: bool
    underfitting_status: bool
    leakage_status: bool
    generalization_gap: dict[str, float]
    cross_validation_summary: dict[str, float]
    calibration_summary: dict[str, float]
    experiment_health_score: float
    promotion_recommendation: str
    curve_analysis: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
