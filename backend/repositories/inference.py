from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional, List
from models.inference import InferenceJobModel
from schemas.inference import InferenceJob, JobState

class InferenceRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, job_data: InferenceJob) -> InferenceJobModel:
        db_job = InferenceJobModel(
            job_id=job_data.job_id,
            idempotency_key=job_data.idempotency_key,
            user_id=job_data.user_id,
            status=job_data.status.value,
            study_metadata=job_data.study_metadata,
            model_versions=job_data.model_versions,
            progress=job_data.progress,
            api_version=job_data.api_version
        )
        self.db.add(db_job)
        self.db.commit()
        self.db.refresh(db_job)
        return db_job

    def get_by_job_id(self, job_id: str) -> Optional[InferenceJobModel]:
        return self.db.execute(select(InferenceJobModel).where(InferenceJobModel.job_id == job_id)).scalars().first()

    def get_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Optional[InferenceJobModel]:
        if not user_id or not idempotency_key:
            return None
        return self.db.execute(
            select(InferenceJobModel).where(
                InferenceJobModel.user_id == user_id, 
                InferenceJobModel.idempotency_key == idempotency_key
            )
        ).scalars().first()

    def update(self, job_id: str, update_data: dict) -> Optional[InferenceJobModel]:
        db_job = self.get_by_job_id(job_id)
        if db_job:
            for key, value in update_data.items():
                if hasattr(db_job, key):
                    setattr(db_job, key, value)
            self.db.commit()
            self.db.refresh(db_job)
        return db_job
