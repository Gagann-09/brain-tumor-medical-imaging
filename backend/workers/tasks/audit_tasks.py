"""Audit logging tasks."""

import structlog

from workers.celery_app import celery_app

logger = structlog.get_logger()


# Exponential backoff: 2s, 4s, 8s... max 5 retries.
@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5},
    retry_backoff=2,
    retry_backoff_max=60,
    retry_jitter=True,
)
def publish_audit_event(self, event_data: dict):
    """Publish an audit event without blocking the API response."""
    try:
        # Placeholder for writing to database or external audit system
        logger.info("audit_event_published", data=event_data)
        # from services.database import SessionLocal
        # from models.audit import AuditLog
        # with SessionLocal() as db:
        #     audit = AuditLog(**event_data)
        #     db.add(audit)
        #     db.commit()
    except Exception as exc:
        logger.error("audit_event_publish_failed", error=str(exc))
        raise self.retry(exc=exc) from exc
