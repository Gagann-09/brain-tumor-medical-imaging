"""SQLAlchemy mixins for common patterns."""

from sqlalchemy import Boolean, Column, DateTime, Integer


class SoftDeleteMixin:
    """Provides soft-delete capability to models."""

    is_deleted = Column(Boolean, nullable=False, default=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)


class OptimisticLockingMixin:
    """Provides optimistic concurrency control via version_id."""

    version_id = Column(Integer, nullable=False, default=1)

    __mapper_args__ = {  # type: ignore
        "version_id_col": version_id,
        "version_id_generator": False,
    }
