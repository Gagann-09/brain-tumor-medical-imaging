from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class UploadState(StrEnum):
    UPLOADED = "uploaded"
    QUARANTINE = "quarantine"
    VALIDATING = "validating"
    READY = "ready"
    PROCESSING = "processing"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class UploadRecord(BaseModel):
    upload_id: str
    filename: str
    status: UploadState
    user_id: str | None = None
    storage_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
