"""Patient management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from schemas.common import APIResponse
from services.database import get_db

from .schemas import PatientCreate, PatientResponse, PatientUpdate
from .service import PatientService

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("/", response_model=APIResponse[PatientResponse], status_code=201)
async def create_patient(
    data: PatientCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new patient record."""
    service = PatientService(db)
    patient = service.create(data, created_by=current_user.id)
    return APIResponse(data=patient)


@router.get("/", response_model=APIResponse[list[PatientResponse]])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List patients with pagination."""
    service = PatientService(db)
    patients = service.list_all(skip=skip, limit=limit)
    return APIResponse(data=patients)


@router.get("/{patient_id}", response_model=APIResponse[PatientResponse])
async def get_patient(
    patient_id: UUID,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get a patient by ID."""
    service = PatientService(db)
    return APIResponse(data=service.get_by_id(patient_id))


@router.put("/{patient_id}", response_model=APIResponse[PatientResponse])
async def update_patient(
    patient_id: UUID,
    data: PatientUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update a patient record."""
    service = PatientService(db)
    return APIResponse(data=service.update(patient_id, data))
