"""Local filesystem storage backend."""

from pathlib import Path
from typing import BinaryIO

from .object_storage import ObjectStorage


class LocalStorage(ObjectStorage):
    """Store files on the local filesystem."""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload(self, key: str, data: BinaryIO, content_type: str = "") -> str:
        target = self.base_path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.read())
        return str(target)

    async def download(self, key: str) -> bytes:
        return (self.base_path / key).read_bytes()

    async def delete(self, key: str) -> None:
        target = self.base_path / key
        if target.exists():
            target.unlink()

    async def exists(self, key: str) -> bool:
        return (self.base_path / key).exists()

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return f"/files/{key}"
