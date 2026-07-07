"""Segmentation result ORM model."""

import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .base import Base
from .mixins import OptimisticLockingMixin, SoftDeleteMixin


class SegmentationResult(SoftDeleteMixin, OptimisticLockingMixin, Base):
    __tablename__ = "segmentation_results"
    __table_args__ = (
        Index("ix_segmentation_results_study_id", "study_id"),
        Index("ix_segmentation_results_prediction_id", "prediction_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(
        UUID(as_uuid=True), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False
    )
    prediction_id = Column(
        UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True
    )
    model_version = Column(String(50), nullable=False)
    mask_path = Column(String(500), nullable=True)
    tumor_volume_mm3 = Column(Float, nullable=True)
    dice_score = Column(Float, nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    requested_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
