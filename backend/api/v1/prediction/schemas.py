"""Pydantic schemas for tumor classification prediction."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PredictionCreate(BaseModel):
    """Request to create a prediction."""
    study_id: UUID
    model_version: str = "v1.0.0"


class PredictionResponse(BaseModel):
    """Response for prediction data."""
    id: UUID
    study_id: UUID
    model_version: str
    prediction_class: str | None = None
    confidence: float | None = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
