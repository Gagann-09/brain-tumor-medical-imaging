"""Imaging study ORM model."""

import uuid

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, String, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .base import Base
from .mixins import SoftDeleteMixin, OptimisticLockingMixin


class Study(SoftDeleteMixin, OptimisticLockingMixin, Base):
    __tablename__ = "studies"
    __table_args__ = (
        Index("ix_studies_patient_id", "patient_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    study_type = Column(String(50), nullable=False)
    modality = Column(String(50), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    description = Column(Text, nullable=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
