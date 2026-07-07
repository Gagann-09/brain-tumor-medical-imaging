from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class InferenceJob(BaseModel):
    job_id: str
    idempotency_key: str | None = None
    user_id: str | None = None
    status: JobState
    created_at: datetime
    updated_at: datetime
    study_metadata: dict[str, Any]
    model_versions: dict[str, str] = Field(default_factory=dict)
    final_result: dict[str, Any] | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    retry_count: int = 0
    api_version: str = "v1"

    class Config:
        from_attributes = True
