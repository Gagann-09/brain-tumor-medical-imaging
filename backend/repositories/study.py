"""Study repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.study import Study


class StudyRepository:
    """Data access layer for Study entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, study_id: UUID) -> Study | None:
        return self.db.query(Study).filter(Study.id == study_id).first()

    def list_by_patient(self, patient_id: UUID) -> list[Study]:
        return self.db.query(Study).filter(Study.patient_id == patient_id).all()

    def create(
        self,
        *,
        patient_id: UUID,
        study_type: str,
        file_path: str,
        uploaded_by: UUID,
        **kwargs,
    ) -> Study:
        study = Study(
            patient_id=patient_id,
            study_type=study_type,
            file_path=file_path,
            uploaded_by=uploaded_by,
            **kwargs,
        )
        self.db.add(study)
        self.db.commit()
        self.db.refresh(study)
        return study
