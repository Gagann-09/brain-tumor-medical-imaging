"""Pydantic schemas for tumor region segmentation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SegmentationCreate(BaseModel):
    """Request to create a segmentation."""

    study_id: UUID
    model_version: str = "v1.0.0"


class SegmentationResponse(BaseModel):
    """Response for segmentation data."""

    id: UUID
    study_id: UUID
    model_version: str
    mask_path: str | None = None
    tumor_volume_mm3: float | None = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
