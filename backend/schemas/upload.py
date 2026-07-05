from enum import Enum
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UploadState(str, Enum):
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
    user_id: Optional[str] = None
    storage_path: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
