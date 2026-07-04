"""Audit repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.audit import AuditLog


class AuditRepository:
    """Data access layer for AuditLog entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, *, action: str, resource_type: str, **kwargs) -> AuditLog:
        log = AuditLog(action=action, resource_type=resource_type, **kwargs)
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def list_by_user(self, user_id: UUID, skip: int = 0, limit: int = 50) -> list[AuditLog]:
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def list_by_resource(self, resource_type: str, resource_id: str) -> list[AuditLog]:
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.resource_type == resource_type, AuditLog.resource_id == resource_id)
            .order_by(AuditLog.created_at.desc())
            .all()
        )
