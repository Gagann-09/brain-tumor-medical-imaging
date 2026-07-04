"""Pydantic schemas for audit trail and compliance logging."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AuditCreate(BaseModel):
    """Request to create a audit."""
    pass


class AuditResponse(BaseModel):
    """Response for audit data."""
    id: UUID
    user_id: UUID | None = None
    action: str
    resource_type: str
    resource_id: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
