"""Pydantic schemas for administrative operations."""

from pydantic import BaseModel


class AdminCreate(BaseModel):
    """Request to create a admin."""

    pass


class AdminResponse(BaseModel):
    """Response for admin data."""

    total_users: int = 0
    total_patients: int = 0
    total_predictions: int = 0

    model_config = {"from_attributes": True}
