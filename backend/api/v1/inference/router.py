from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional
import json

from sqlalchemy.orm import Session
from schemas.inference import InferenceJob
from services.inference_service import InferenceService
from services.database import get_db

router = APIRouter(prefix="/inference", tags=["inference"])

def get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    return InferenceService(db=db)

@router.post("", response_model=InferenceJob)
async def submit_inference(
    study_metadata: str = Form(...), # JSON string
    is_async: bool = Form(True),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    files: List[UploadFile] = File(...),
    service: InferenceService = Depends(get_inference_service)
    # user = Depends(get_current_active_user)
):
    try:
        metadata_dict = json.loads(study_metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid study_metadata JSON")
        
    metadata_dict["uploaded_files"] = [f.filename for f in files]
    
    # Normally user_id comes from auth token
    user_id = "test_user" 
    
    job = service.submit_inference(
        study_metadata=metadata_dict, 
        user_id=user_id,
        idempotency_key=idempotency_key,
        is_async=is_async
    )
    return job

@router.get("/{job_id}", response_model=InferenceJob)
async def get_inference_status(
    job_id: str, 
    service: InferenceService = Depends(get_inference_service)
):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/{job_id}/artifacts")
async def get_inference_artifacts(
    job_id: str, 
    service: InferenceService = Depends(get_inference_service)
):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.final_result:
        raise HTTPException(status_code=400, detail="Job has no final result yet")
        
    return {"explainability": job.final_result.get("explainability", {}), "report": job.final_result.get("report", {})}

@router.delete("/{job_id}")
async def cancel_inference(
    job_id: str, 
    service: InferenceService = Depends(get_inference_service)
):
    success = service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Unable to cancel job")
    return {"message": "Job cancelled"}
