import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class ModelLifecycleState(StrEnum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelPromotionRecord(BaseModel):
    model_name: str
    version: str
    state: ModelLifecycleState
    benchmark_id: str | None = None
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
            with open(self.registry_file) as f:
                data = json.load(f)
                self.records = {k: ModelPromotionRecord(**v) for k, v in data.items()}
        else:
            self.records: dict[str, ModelPromotionRecord] = {}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in self.records.items()}, f, indent=4)

    def register_model(self, model_name: str, version: str) -> None:
        """Registers a new model in EXPERIMENTAL state."""
        record_id = f"{model_name}_{version}"
        if record_id not in self.records:
            self.records[record_id] = ModelPromotionRecord(
                model_name=model_name, version=version, state=ModelLifecycleState.EXPERIMENTAL
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
        benchmark_id: str,
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

    def evaluate_classification_candidate(
        self,
        candidate_model: str,
        candidate_version: str,
        baseline_metrics: dict,
        candidate_metrics: dict,
        benchmark_id: str,
    ) -> bool:
        """
        Evaluates a classification candidate against multiple criteria (F1, ROC AUC, Brier Score, Latency).  # noqa: E501
        """
        record_id = f"{candidate_model}_{candidate_version}"
        if record_id not in self.records:
            return False

        record = self.records[record_id]

        c_f1 = candidate_metrics.get("f1_score", 0.0)
        b_f1 = baseline_metrics.get("f1_score", 0.0)

        c_roc = candidate_metrics.get("roc_auc", 0.0)
        b_roc = baseline_metrics.get("roc_auc", 0.0)

        c_brier = candidate_metrics.get("brier_score", float("inf"))
        b_brier = baseline_metrics.get("brier_score", float("inf"))

        c_latency = candidate_metrics.get("avg_inference_latency_sec", 0.0)
        b_latency = baseline_metrics.get("avg_inference_latency_sec", 0.0)

        reasons = []

        if (
            c_f1 < b_f1 - 0.02
        ):  # Allow slight F1 degradation if other metrics are much better, but strict threshold
            reasons.append(f"F1 Score degraded too much (Base: {b_f1:.4f}, Cand: {c_f1:.4f})")

        if c_roc < b_roc - 0.01:
            reasons.append(f"ROC AUC degraded (Base: {b_roc:.4f}, Cand: {c_roc:.4f})")

        if c_brier > b_brier + 0.05:
            reasons.append(
                f"Calibration (Brier) worsened (Base: {b_brier:.4f}, Cand: {c_brier:.4f})"
            )

        if c_latency > b_latency * 1.5:
            reasons.append(
                f"Latency degraded >50% (Base: {b_latency:.3f}s, Cand: {c_latency:.3f}s)"
            )

        # Overall score check
        # Require improvement in at least one primary metric without failing the thresholds above
        improved = (c_f1 > b_f1) or (c_roc > b_roc) or (c_brier < b_brier)

        if reasons:
            record.promotion_reason = "Failed: " + "; ".join(reasons)
            self._save_registry()
            return False

        if not improved:
            record.promotion_reason = (
                "Failed: Did not improve any primary metric (F1, ROC AUC, Brier)."
            )
            self._save_registry()
            return False

        record.state = ModelLifecycleState.VALIDATED
        record.benchmark_id = benchmark_id
        record.promotion_reason = "Passed: Classification metrics surpassed baseline."
        self._save_registry()
        return True

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Promotes a VALIDATED model to PRODUCTION. Archives the old production model."""
        record_id = f"{model_name}_{version}"
        if (
            record_id in self.records
            and self.records[record_id].state == ModelLifecycleState.VALIDATED
        ):
            # Archive existing production models for this model_name
            for _r_id, r in self.records.items():
                if r.model_name == model_name and r.state == ModelLifecycleState.PRODUCTION:
                    r.state = ModelLifecycleState.ARCHIVED

            self.records[record_id].state = ModelLifecycleState.PRODUCTION
            self._save_registry()

    def evaluate_promotion_robust(
        self,
        candidate_model: str,
        candidate_version: str,
        health_score: float,
        critical_failures: list[str],
        benchmark_id: str,
    ) -> bool:
        """
        Evaluates promotion based on the Experiment Health Score and critical failures.
        """
        record_id = f"{candidate_model}_{candidate_version}"
        if record_id not in self.records:
            return False

        record = self.records[record_id]

        if critical_failures:
            record.promotion_reason = "Failed: Critical failures detected: " + ", ".join(
                critical_failures
            )
            self._save_registry()
            return False

        if health_score < 70.0:  # Assuming 70 is passing threshold
            record.promotion_reason = (
                f"Failed: Health Score ({health_score:.2f}) below passing threshold (70.0)."
            )
            self._save_registry()
            return False

        record.state = ModelLifecycleState.VALIDATED
        record.benchmark_id = benchmark_id
        record.promotion_reason = f"Passed: Health Score {health_score:.2f} met criteria."
        self._save_registry()
        return True
