"""Patient repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.patient import Patient


class PatientRepository:
    """Data access layer for Patient entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, patient_id: UUID) -> Patient | None:
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_by_mrn(self, mrn: str) -> Patient | None:
        return self.db.query(Patient).filter(Patient.medical_record_number == mrn).first()

    def create(
        self,
        *,
        medical_record_number: str,
        full_name: str,
        created_by: UUID,
        **kwargs,
    ) -> Patient:
        patient = Patient(
            medical_record_number=medical_record_number,
            full_name=full_name,
            created_by=created_by,
            **kwargs,
        )
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        return patient

    def list_all(self, skip: int = 0, limit: int = 20) -> list[Patient]:
        return self.db.query(Patient).offset(skip).limit(limit).all()

    def count(self) -> int:
        return self.db.query(Patient).count()

    def search(self, query: str, skip: int = 0, limit: int = 20) -> list[Patient]:
        return (
            self.db.query(Patient)
            .filter(Patient.full_name.ilike(f"%{query}%"))
            .offset(skip)
            .limit(limit)
            .all()
        )
