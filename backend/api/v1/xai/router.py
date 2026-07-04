"""Explainable AI visualizations endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from schemas.common import APIResponse
from services.database import get_db

from .schemas import XaiCreate, XaiResponse

router = APIRouter(prefix="/xai", tags=["XAI"])


@router.post("/", response_model=APIResponse[XaiResponse], status_code=201)
async def create_xai(
    data: XaiCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new xai request."""
    return APIResponse(data=None, message="Not implemented yet")


@router.get("/{item_id}", response_model=APIResponse[XaiResponse])
async def get_xai(
    item_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve xai by ID."""
    return APIResponse(data=None, message="Not implemented yet")
