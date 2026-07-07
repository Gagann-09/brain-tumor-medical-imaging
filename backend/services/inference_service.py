import dataclasses
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ai.inference.context import InferenceContext
from ai.inference.manifest import PipelineManifest
from ai.inference.orchestrator import InferenceOrchestrator
from ai.inference.policy import ExecutionMode, InferencePolicy
from ai.inference.registry import ArtifactRegistry
from ai.inference.stages import STAGE_CLASSES
from ai.training.events import EventBus
from repositories.inference import InferenceRepository
from schemas.inference import InferenceJob, JobState


def _sanitize_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.float32, np.float64)):
        return float(d)
    elif isinstance(d, (np.int32, np.int64)):
        return int(d)
    return d


class InferenceService:
    def __init__(self, db: Session, storage_dir: str = "outputs_val/inference_artifacts"):
        self.db = db
        self.repo = InferenceRepository(db)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus = EventBus()

    def get_job(self, job_id: str) -> InferenceJob | None:
        db_job = self.repo.get_by_job_id(job_id)
        if not db_job:
            return None
        return InferenceJob.model_validate(db_job)

    def submit_inference(
        self,
        study_metadata: dict[str, Any],
        user_id: str | None = None,
        idempotency_key: str | None = None,
        is_async: bool = True,
    ) -> InferenceJob:

        # Check idempotency
        if user_id and idempotency_key:
            existing_job = self.repo.get_by_idempotency_key(user_id, idempotency_key)
            if existing_job:
                return InferenceJob.model_validate(existing_job)

        job_id = str(uuid.uuid4())
        job = InferenceJob(
            job_id=job_id,
            idempotency_key=idempotency_key,
            user_id=user_id,
            status=JobState.QUEUED if is_async else JobState.RUNNING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            study_metadata=study_metadata,
            model_versions={
                "segmentation": study_metadata.get("segmentation_model", "latest"),
                "classification": study_metadata.get("classification_model", "latest"),
            },
            progress={"stage": "Validation", "percent": 0},
            api_version="v1",
        )
        self.repo.create(job)

        if is_async:
            from workers.inference_tasks import execute_inference_async

            execute_inference_async.delay(job_id)
        else:
            self.execute_inference_sync(job_id)

        return self.get_job(job_id)

    def execute_inference_sync(self, job_id: str, retry_attempt: bool = False):
        db_job = self.repo.get_by_job_id(job_id)
        if not db_job:
            return

        def update_progress(stage: str, percent: int):
            curr = self.repo.get_by_job_id(job_id)
            if curr and curr.status not in [JobState.CANCELLED.value, JobState.FAILED.value]:
                self.repo.update(job_id, {"progress": {"stage": stage, "percent": percent}})

        self.repo.update(
            job_id, {"status": JobState.RUNNING.value, "updated_at": datetime.utcnow()}
        )

        try:
            artifact_registry = ArtifactRegistry(output_dir=str(self.storage_dir / job_id))
            policy = InferencePolicy.get_default_policy(ExecutionMode.CLINICAL_REVIEW)
            manifest = PipelineManifest()

            # Using basic orchestration without profiler dependency here for brevity
            orchestrator = InferenceOrchestrator(
                event_bus=self.event_bus,
                artifact_registry=artifact_registry,
                policy=policy,
                manifest=manifest,
                stage_classes=STAGE_CLASSES,
                profiler=None,
                progress_callback=update_progress,
            )

            context = InferenceContext(study_metadata=db_job.study_metadata)
            result = orchestrator.run(context)

            if result.status == "success":
                status = JobState.COMPLETED.value
            else:
                status = JobState.FAILED.value

            result_dict = dataclasses.asdict(result)
            for _stage, diag in result_dict["diagnostics"].items():
                if hasattr(diag["status"], "value"):
                    diag["status"] = diag["status"].value

            # Store only artifact references (handled natively by orchestrator)
            sanitized_result = _sanitize_dict(result_dict)
            self.repo.update(
                job_id,
                {
                    "status": status,
                    "final_result": sanitized_result,
                    "updated_at": datetime.utcnow(),
                },
            )

        except Exception as e:
            # Check if retryable
            # We assume transient DB errors or network issues are transient.
            # In production, check error type.
            is_transient = True
            if "ValidationError" in str(e) or "FileNotFound" in str(e):
                is_transient = False

            if is_transient and not retry_attempt and db_job.retry_count < 3:
                self.repo.update(
                    job_id,
                    {
                        "status": JobState.RETRYING.value,
                        "retry_count": db_job.retry_count + 1,
                        "error_message": str(e),
                        "updated_at": datetime.utcnow(),
                    },
                )
                # In real system, this would trigger celery delay again
                # execute_inference_async.delay(job_id, retry_attempt=True)
            else:
                self.repo.update(
                    job_id,
                    {
                        "status": JobState.FAILED.value,
                        "error_message": str(e),
                        "updated_at": datetime.utcnow(),
                    },
                )

    def cancel_job(self, job_id: str) -> bool:
        db_job = self.repo.get_by_job_id(job_id)
        if not db_job or db_job.status not in [
            JobState.QUEUED.value,
            JobState.RUNNING.value,
            JobState.RETRYING.value,
        ]:
            return False

        self.repo.update(
            job_id, {"status": JobState.CANCELLED.value, "updated_at": datetime.utcnow()}
        )
        return True
