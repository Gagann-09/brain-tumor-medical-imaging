"""API v1 aggregation router - includes all feature routers."""

from fastapi import APIRouter

from .admin.router import router as admin_router
from .audit.router import router as audit_router
from .auth.router import router as auth_router
from .inference.router import router as inference_router
from .patient.router import router as patient_router
from .prediction.router import router as prediction_router
from .segmentation.router import router as segmentation_router
from .upload.router import router as upload_router
from .xai.router import router as xai_router

api_v1_router = APIRouter()

api_v1_router.include_router(auth_router)
api_v1_router.include_router(patient_router)
api_v1_router.include_router(prediction_router)
api_v1_router.include_router(segmentation_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(xai_router)
api_v1_router.include_router(admin_router)
api_v1_router.include_router(audit_router)
api_v1_router.include_router(inference_router)
