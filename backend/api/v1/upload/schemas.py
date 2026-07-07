"""Pydantic schemas for medical image upload and processing."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UploadCreate(BaseModel):
    """Request to create a upload."""

    patient_id: UUID
    study_type: str
    description: str | None = None


class UploadResponse(BaseModel):
    """Response for upload data."""

    id: UUID
    patient_id: UUID
    study_type: str
    file_path: str
    file_size_bytes: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
