"""Centralized project constants - import from here, not from magic strings."""

# -- API --------------------------------------------------
API_V1_PREFIX = "/api/v1"

# -- Auth -------------------------------------------------
AUTH_TOKEN_TYPE = "bearer"
AUTH_SCHEME_NAME = "BearerAuth"

# -- Roles ------------------------------------------------
ROLE_ADMIN = "admin"
ROLE_PHYSICIAN = "physician"
ROLE_RADIOLOGIST = "radiologist"
ROLE_RESEARCHER = "researcher"
ROLE_VIEWER = "viewer"
ALL_ROLES = [ROLE_ADMIN, ROLE_PHYSICIAN, ROLE_RADIOLOGIST, ROLE_RESEARCHER, ROLE_VIEWER]

# -- Pagination -------------------------------------------
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# -- Error Codes ------------------------------------------
ERR_AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
ERR_AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
ERR_AUTH_FORBIDDEN = "AUTH_FORBIDDEN"
ERR_RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
ERR_RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
ERR_VALIDATION_ERROR = "VALIDATION_ERROR"
ERR_INTERNAL_ERROR = "INTERNAL_ERROR"
ERR_SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

# -- Prediction / Segmentation Status --------------------
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# -- Tumor Types ------------------------------------------
TUMOR_GLIOMA = "glioma"
TUMOR_MENINGIOMA = "meningioma"
TUMOR_PITUITARY = "pituitary"
TUMOR_NO_TUMOR = "no_tumor"
TUMOR_TYPES = [TUMOR_GLIOMA, TUMOR_MENINGIOMA, TUMOR_PITUITARY, TUMOR_NO_TUMOR]

# -- File / Storage ---------------------------------------
STORAGE_BACKEND_LOCAL = "local"
STORAGE_BACKEND_S3 = "s3"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz"}
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

# -- Correlation ID Header -------------------------------
CORRELATION_ID_HEADER = "X-Correlation-ID"

# -- Model Defaults ---------------------------------------
DEFAULT_MODEL_VERSION = "v1.0.0"
