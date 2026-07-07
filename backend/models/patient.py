"""Patient ORM model."""

import uuid

from sqlalchemy import Column, Date, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .base import Base
from .mixins import OptimisticLockingMixin, SoftDeleteMixin


class Patient(SoftDeleteMixin, OptimisticLockingMixin, Base):
    __tablename__ = "patients"
    __table_args__ = (
        Index(
            "ix_patients_medical_record_number_not_deleted",
            "medical_record_number",
            unique=True,
            postgresql_where="(is_deleted IS FALSE)",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    medical_record_number = Column(String(100), nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)
    referring_physician = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
