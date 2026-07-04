"""Tumor classification prediction endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from schemas.common import APIResponse
from services.database import get_db

from .schemas import PredictionCreate, PredictionResponse

router = APIRouter(prefix="/predictions", tags=["Prediction"])


@router.post("/", response_model=APIResponse[PredictionResponse], status_code=201)
async def create_prediction(
    data: PredictionCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new prediction request."""
    return APIResponse(data=None, message="Not implemented yet")


@router.get("/{item_id}", response_model=APIResponse[PredictionResponse])
async def get_prediction(
    item_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve prediction by ID."""
    return APIResponse(data=None, message="Not implemented yet")
