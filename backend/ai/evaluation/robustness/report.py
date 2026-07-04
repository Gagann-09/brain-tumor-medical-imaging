from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json

@dataclass
class GeneralizationReport:
    """Summarizes all generalization metrics and experiment health."""
    experiment_id: str
    overfitting_status: bool
    underfitting_status: bool
    leakage_status: bool
    generalization_gap: Dict[str, float]
    cross_validation_summary: Dict[str, float]
    calibration_summary: Dict[str, float]
    experiment_health_score: float
    promotion_recommendation: str
    curve_analysis: Dict[str, bool] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
