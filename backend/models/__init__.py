"""ORM model registry - import all models here for Alembic auto-detection."""

from .audit import AuditLog
from .base import Base
from .inference import InferenceJobModel
from .patient import Patient
from .prediction import Prediction
from .segmentation import SegmentationResult
from .study import Study
from .upload import UploadRecordModel
from .user import User

__all__ = [
    "AuditLog",
    "Base",
    "InferenceJobModel",
    "Patient",
    "Prediction",
    "SegmentationResult",
    "Study",
    "UploadRecordModel",
    "User",
]
