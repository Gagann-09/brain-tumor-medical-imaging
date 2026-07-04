"""Segmentation repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.segmentation import SegmentationResult


class SegmentationRepository:
    """Data access layer for SegmentationResult entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, seg_id: UUID) -> SegmentationResult | None:
        return self.db.query(SegmentationResult).filter(SegmentationResult.id == seg_id).first()

    def list_by_study(self, study_id: UUID) -> list[SegmentationResult]:
        return (
            self.db.query(SegmentationResult)
            .filter(SegmentationResult.study_id == study_id)
            .all()
        )

    def create(
        self, *, study_id: UUID, model_version: str, requested_by: UUID, **kwargs,
    ) -> SegmentationResult:
        seg = SegmentationResult(
            study_id=study_id,
            model_version=model_version,
            requested_by=requested_by,
            **kwargs,
        )
        self.db.add(seg)
        self.db.commit()
        self.db.refresh(seg)
        return seg
