from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional, Any
from pathlib import Path
import json

class ModelLifecycleState(str, Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelPromotionRecord(BaseModel):
    model_name: str
    version: str
    state: ModelLifecycleState
    benchmark_id: Optional[str] = None
    promotion_reason: str = ""

class ModelPromotionManager:
    """Manages model lifecycle states and strict promotion policies."""
    
    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "promotion_state.json"
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                data = json.load(f)
                self.records = {k: ModelPromotionRecord(**v) for k, v in data.items()}
        else:
            self.records: Dict[str, ModelPromotionRecord] = {}
            
    def _save_registry(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in self.records.items()}, f, indent=4)
            
    def register_model(self, model_name: str, version: str) -> None:
        """Registers a new model in EXPERIMENTAL state."""
        record_id = f"{model_name}_{version}"
        if record_id not in self.records:
            self.records[record_id] = ModelPromotionRecord(
                model_name=model_name,
                version=version,
                state=ModelLifecycleState.EXPERIMENTAL
            )
            self._save_registry()

    def mark_as_candidate(self, model_name: str, version: str) -> None:
        """Transitions an experimental model to CANDIDATE."""
        record_id = f"{model_name}_{version}"
        if record_id in self.records:
            self.records[record_id].state = ModelLifecycleState.CANDIDATE
            self._save_registry()

    def evaluate_promotion(
        self, 
        candidate_model: str, 
        candidate_version: str, 
        baseline_metrics: dict, 
        candidate_metrics: dict, 
        benchmark_id: str
    ) -> bool:
        """
        Evaluates if the candidate model surpasses the baseline.
        If successful, transitions to VALIDATED.
        """
        record_id = f"{candidate_model}_{candidate_version}"
        if record_id not in self.records:
            return False
            
        record = self.records[record_id]
        
        # Policy rules
        has_nan = candidate_metrics.get("has_nan", 0.0) > 0.0
        candidate_dice = candidate_metrics.get("dice", 0.0)
        baseline_dice = baseline_metrics.get("dice", 0.0)
        
        candidate_latency = candidate_metrics.get("latency_sec", 0.0)
        baseline_latency = baseline_metrics.get("latency_sec", 0.0)
        
        # Strict rules
        if has_nan:
            record.promotion_reason = "Failed: NaN predictions detected."
            self._save_registry()
            return False
            
        if candidate_dice <= baseline_dice:
            record.promotion_reason = f"Failed: Dice ({candidate_dice:.4f}) not better than Baseline ({baseline_dice:.4f})."
            self._save_registry()
            return False
            
        # Check latency degradation threshold (allow up to 20% degradation)
        if candidate_latency > (baseline_latency * 1.2):
            record.promotion_reason = f"Failed: Latency degraded by more than 20% (Base: {baseline_latency:.3f}s -> Cand: {candidate_latency:.3f}s)."
            self._save_registry()
            return False
            
        # Passed all checks
        record.state = ModelLifecycleState.VALIDATED
        record.benchmark_id = benchmark_id
        record.promotion_reason = "Passed: Met all promotion criteria."
        self._save_registry()
        return True
        
    def promote_to_production(self, model_name: str, version: str) -> None:
        """Promotes a VALIDATED model to PRODUCTION. Archives the old production model."""
        record_id = f"{model_name}_{version}"
        if record_id in self.records and self.records[record_id].state == ModelLifecycleState.VALIDATED:
            
            # Archive existing production models for this model_name
            for r_id, r in self.records.items():
                if r.model_name == model_name and r.state == ModelLifecycleState.PRODUCTION:
                    r.state = ModelLifecycleState.ARCHIVED
                    
            self.records[record_id].state = ModelLifecycleState.PRODUCTION
            self._save_registry()
