"""Medical image upload and processing endpoints."""

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from schemas.common import APIResponse
from services.database import get_db

from .schemas import UploadResponse
from .service import UploadService

router = APIRouter(prefix="/uploads", tags=["Upload"])


@router.post("/", response_model=APIResponse[dict], status_code=201)
async def create_upload(
    file: UploadFile = File(...),
    patient_id: uuid.UUID = Form(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new upload request."""
    service = UploadService(db)
    result = await service.process_upload(file, patient_id, current_user.id)
    return APIResponse(data=result, message="Upload processed successfully")


@router.get("/{item_id}", response_model=APIResponse[UploadResponse])
async def get_upload(
    item_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve upload by ID."""
    return APIResponse(data=None, message="Not implemented yet")
