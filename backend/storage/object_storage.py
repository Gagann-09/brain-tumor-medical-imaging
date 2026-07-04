"""Abstract base class for object storage backends."""

from abc import ABC, abstractmethod
from typing import BinaryIO


class ObjectStorage(ABC):
    """Interface for all storage backends (local, S3, GCS, etc.)."""

    @abstractmethod
    async def upload(self, key: str, data: BinaryIO, content_type: str = "") -> str:
        """Upload a file and return its storage path/URL."""

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download a file by key."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a file by key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a file exists."""

    @abstractmethod
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a time-limited URL for direct access."""
