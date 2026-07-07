"""Celery application initialization."""

import os

from celery import Celery

from core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "armt_gan_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.tasks.audit_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "4")),
    # Exponential backoff settings can be configured per task
)
