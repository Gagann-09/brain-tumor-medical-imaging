from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class InferenceJob(BaseModel):
    job_id: str
    idempotency_key: Optional[str] = None
    user_id: Optional[str] = None
    status: JobState
    created_at: datetime
    updated_at: datetime
    study_metadata: Dict[str, Any]
    model_versions: Dict[str, str] = Field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    progress: Dict[str, str] = Field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    api_version: str = "v1"

    class Config:
        from_attributes = True
