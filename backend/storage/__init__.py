"""Storage factory - returns the configured backend."""

from core.config import get_settings
from core.constants import STORAGE_BACKEND_S3

from .local_storage import LocalStorage
from .object_storage import ObjectStorage
from .s3_storage import S3Storage


def get_storage() -> ObjectStorage:
    """Return a storage backend based on the current configuration."""
    settings = get_settings()
    if settings.STORAGE_BACKEND == STORAGE_BACKEND_S3:
        return S3Storage(
            bucket=settings.S3_BUCKET,
            region=settings.S3_REGION,
            access_key=settings.S3_ACCESS_KEY_ID,
            secret_key=settings.S3_SECRET_ACCESS_KEY,
        )
    return LocalStorage(base_path=settings.STORAGE_LOCAL_PATH)
