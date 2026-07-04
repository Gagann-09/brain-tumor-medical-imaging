"""Prediction repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.prediction import Prediction


class PredictionRepository:
    """Data access layer for Prediction entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, prediction_id: UUID) -> Prediction | None:
        return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()

    def list_by_study(self, study_id: UUID) -> list[Prediction]:
        return self.db.query(Prediction).filter(Prediction.study_id == study_id).all()

    def create(self, *, study_id: UUID, model_version: str, requested_by: UUID) -> Prediction:
        pred = Prediction(
            study_id=study_id,
            model_version=model_version,
            requested_by=requested_by,
        )
        self.db.add(pred)
        self.db.commit()
        self.db.refresh(pred)
        return pred

    def update_status(self, prediction_id: UUID, status: str, **kwargs) -> Prediction | None:
        pred = self.get_by_id(prediction_id)
        if pred:
            pred.status = status
            for k, v in kwargs.items():
                setattr(pred, k, v)
            self.db.commit()
            self.db.refresh(pred)
        return pred
