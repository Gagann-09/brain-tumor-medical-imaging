"""Tumor region segmentation endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from schemas.common import APIResponse
from services.database import get_db

from .schemas import SegmentationCreate, SegmentationResponse

router = APIRouter(prefix="/segmentations", tags=["Segmentation"])


@router.post("/", response_model=APIResponse[SegmentationResponse], status_code=201)
async def create_segmentation(
    data: SegmentationCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new segmentation request."""
    return APIResponse(data=None, message="Not implemented yet")


@router.get("/{item_id}", response_model=APIResponse[SegmentationResponse])
async def get_segmentation(
    item_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve segmentation by ID."""
    return APIResponse(data=None, message="Not implemented yet")
