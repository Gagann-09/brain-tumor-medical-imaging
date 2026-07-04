"""Audit trail and compliance logging endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from schemas.common import APIResponse
from services.database import get_db

from .schemas import AuditResponse

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("/", response_model=APIResponse[list[AuditResponse]])
async def list_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List audit log entries."""
    return APIResponse(data=[])
