"""ORM model registry - import all models here for Alembic auto-detection."""

from .audit import AuditLog
from .base import Base
from .patient import Patient
from .prediction import Prediction
from .segmentation import SegmentationResult
from .study import Study
from .user import User
from .inference import InferenceJobModel
from .upload import UploadRecordModel

__all__ = [
    "AuditLog",
    "Base",
    "Patient",
    "Prediction",
    "SegmentationResult",
    "Study",
    "User",
    "InferenceJobModel",
    "UploadRecordModel",
]
