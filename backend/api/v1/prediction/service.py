"""Business logic for tumor classification prediction."""

from sqlalchemy.orm import Session


class PredictionService:
    """Prediction business logic layer."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # TODO: Implement domain methods in Phase 2
