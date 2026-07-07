class MockTask:
    def delay(self, job_id: str):
        from services.database import SessionLocal
        from services.inference_service import InferenceService

        db = SessionLocal()
        try:
            service = InferenceService(db=db)
            service.execute_inference_sync(job_id)
        finally:
            db.close()


try:
    from celery import Celery

    celery_app = Celery("inference", broker="redis://localhost:6379/0")

    @celery_app.task(bind=True, max_retries=3)
    def execute_inference_async(self, job_id: str):
        from services.database import SessionLocal
        from services.inference_service import InferenceService

        db = SessionLocal()
        try:
            service = InferenceService(db=db)
            service.execute_inference_sync(job_id, retry_attempt=self.request.retries > 0)
        finally:
            db.close()
except ImportError:
    # Fallback mock for testing if celery is missing
    execute_inference_async = MockTask()
