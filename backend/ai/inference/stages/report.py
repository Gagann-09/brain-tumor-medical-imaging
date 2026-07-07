from ai.inference.context import InferenceContext, StageDiagnostics, StageStatus
from ai.inference.stages.base import BaseStage


class ClinicalReportGenerationStage(BaseStage):
    def execute(self, context: InferenceContext) -> tuple[InferenceContext, StageDiagnostics]:
        # Structured report (not narrative)
        report = {
            "patient_id": context.study_metadata.get("patient_id", "unknown"),
            "study_id": context.study_metadata.get("study_id", "unknown"),
            "findings": {
                "tumor_present": bool(context.classification_results),
                "primary_classification": context.classification_results.get(
                    "tumor_type", "unknown"
                )
                if context.classification_results
                else None,
                "confidence_score": context.classification_results.get("confidence", 0.0)
                if context.classification_results
                else 0.0,
                "roi_bbox": context.roi_information.get("bbox", None)
                if context.roi_information
                else None,
            },
            "artifacts_referenced": list(context.explainability_artifacts.keys()),
        }
        new_context = context.update(generated_reports={"clinical_report": report})
        return new_context, StageDiagnostics(
            self.name, StageStatus.SUCCESS, "Structured report generated."
        )
