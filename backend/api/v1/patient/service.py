"""Patient business logic."""

from uuid import UUID

from sqlalchemy.orm import Session

from api.errors import APIError
from core.constants import ERR_RESOURCE_CONFLICT, ERR_RESOURCE_NOT_FOUND
from repositories.patient import PatientRepository

from .schemas import PatientCreate, PatientResponse, PatientUpdate


class PatientService:
    """Handles patient CRUD operations."""

    def __init__(self, db: Session) -> None:
        self.repo = PatientRepository(db)

    def create(self, data: PatientCreate, created_by: UUID) -> PatientResponse:
        """Create a new patient."""
        existing = self.repo.get_by_mrn(data.medical_record_number)
        if existing:
            raise APIError(409, ERR_RESOURCE_CONFLICT, "MRN already exists")
        patient = self.repo.create(created_by=created_by, **data.model_dump())
        return PatientResponse.model_validate(patient)

    def get_by_id(self, patient_id: UUID) -> PatientResponse:
        """Retrieve a single patient by ID."""
        patient = self.repo.get_by_id(patient_id)
        if not patient:
            raise APIError(404, ERR_RESOURCE_NOT_FOUND, "Patient not found")
        return PatientResponse.model_validate(patient)

    def list_all(self, skip: int = 0, limit: int = 20) -> list[PatientResponse]:
        """List patients with pagination."""
        patients = self.repo.list_all(skip=skip, limit=limit)
        return [PatientResponse.model_validate(p) for p in patients]

    def update(self, patient_id: UUID, data: PatientUpdate) -> PatientResponse:
        """Update patient fields."""
        patient = self.repo.get_by_id(patient_id)
        if not patient:
            raise APIError(404, ERR_RESOURCE_NOT_FOUND, "Patient not found")
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(patient, field, value)
        self.repo.db.commit()
        self.repo.db.refresh(patient)
        return PatientResponse.model_validate(patient)
