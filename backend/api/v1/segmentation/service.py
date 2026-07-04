"""Business logic for tumor region segmentation."""

from sqlalchemy.orm import Session


class SegmentationService:
    """Segmentation business logic layer."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # TODO: Implement domain methods in Phase 2
