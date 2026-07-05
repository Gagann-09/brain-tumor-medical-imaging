import os
import sys
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import select
from models.inference import InferenceJobModel
from models.upload import UploadRecordModel
from schemas.inference import JobState
from schemas.upload import UploadState
from services.database import SessionLocal
from core.config import get_settings

settings = get_settings()

try:
    from celery import Celery
    from celery.schedules import crontab
    
    celery_app = Celery('cleanup', broker=settings.REDIS_URL)
    
    @celery_app.on_after_configure.connect
    def setup_periodic_tasks(sender, **kwargs):
        # Run daily at midnight
        sender.add_periodic_task(crontab(hour=0, minute=0), cleanup_expired_uploads.s())
        sender.add_periodic_task(crontab(hour=0, minute=0), cleanup_expired_jobs.s())

    @celery_app.task
    def cleanup_expired_uploads():
        db = SessionLocal()
        try:
            expiration_date = datetime.utcnow() - timedelta(days=settings.UPLOAD_RETENTION_DAYS)
            
            expired = db.execute(
                select(UploadRecordModel).where(
                    UploadRecordModel.created_at < expiration_date,
                    UploadRecordModel.status != UploadState.ARCHIVED.value
                )
            ).scalars().all()
            
            for upload in expired:
                if os.path.exists(upload.storage_path):
                    os.remove(upload.storage_path)
                upload.status = UploadState.ARCHIVED.value
                upload.updated_at = datetime.utcnow()
                
            db.commit()
        finally:
            db.close()
            
    @celery_app.task
    def cleanup_expired_jobs():
        db = SessionLocal()
        try:
            expiration_date = datetime.utcnow() - timedelta(days=settings.JOB_RETENTION_DAYS)
            
            expired = db.execute(
                select(InferenceJobModel).where(
                    InferenceJobModel.created_at < expiration_date,
                    InferenceJobModel.status != JobState.EXPIRED.value
                )
            ).scalars().all()
            
            for job in expired:
                artifact_path = f"outputs_val/inference_artifacts/{job.job_id}"
                import shutil
                if os.path.exists(artifact_path):
                    shutil.rmtree(artifact_path)
                
                job.status = JobState.EXPIRED.value
                job.updated_at = datetime.utcnow()
                
            db.commit()
        finally:
            db.close()

except ImportError:
    pass
