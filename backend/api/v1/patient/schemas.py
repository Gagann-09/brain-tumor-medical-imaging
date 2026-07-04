"""Pydantic schemas for patient management."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel


class PatientCreate(BaseModel):
    """Create a new patient record."""
    medical_record_number: str
    full_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    referring_physician: str | None = None
    notes: str | None = None


class PatientUpdate(BaseModel):
    """Partial update for a patient record."""
    full_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    referring_physician: str | None = None
    notes: str | None = None


class PatientResponse(BaseModel):
    """Patient representation."""
    id: UUID
    medical_record_number: str
    full_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    referring_physician: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
