from sqlalchemy import JSON, Column, DateTime, Index, Integer, String
from sqlalchemy.sql import func

from schemas.inference import JobState

from .base import Base


class InferenceJobModel(Base):
    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    idempotency_key = Column(String, index=True, nullable=True)
    user_id = Column(String, index=True, nullable=True)
    status = Column(String, default=JobState.QUEUED.value, index=True)
    study_metadata = Column(JSON, default={})
    model_versions = Column(JSON, default={})
    final_result = Column(JSON, nullable=True)
    progress = Column(JSON, default={})
    error_message = Column(String, nullable=True)
    retry_count = Column(Integer, default=0)
    api_version = Column(String, default="v1")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Ensure idempotency scoped by user
    __table_args__ = (Index("ix_user_idempotency", "user_id", "idempotency_key", unique=True),)
