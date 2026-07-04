"""Pydantic schemas for explainable ai visualizations."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class XaiCreate(BaseModel):
    """Request to create a xai."""
    prediction_id: UUID


class XaiResponse(BaseModel):
    """Response for xai data."""
    id: UUID
    prediction_id: UUID
    explanation_type: str
    artifact_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
