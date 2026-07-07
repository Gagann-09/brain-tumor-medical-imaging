"""Business logic for medical image upload and processing."""

import hashlib
import uuid

import filetype  # type: ignore[import-untyped]
import pydicom
from fastapi import HTTPException, UploadFile
from pydicom.errors import InvalidDicomError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from storage.security import ClamAVScanner, NoOpScanner, VirusScanner


class UploadService:
    """Upload business logic layer."""

    def __init__(self, db: AsyncSession = None) -> None:
        self.db = db
        self.settings = get_settings()
        # Dependency injection for scanner could be better, but direct instantiation for now
        self.scanner: VirusScanner = (
            NoOpScanner() if self.settings.ENVIRONMENT == "development" else ClamAVScanner()
        )

    async def process_upload(
        self, file: UploadFile, patient_id: uuid.UUID, user_id: uuid.UUID
    ) -> dict:
        """Process, validate, and store an uploaded medical image."""
        content = await file.read()

        # 1. Antivirus Scan
        is_safe = await self.scanner.scan(content)
        if not is_safe:
            raise HTTPException(
                status_code=400, detail="File failed security scan (malware detected)."
            )

        # 2. Content Validation
        kind = filetype.guess(content)

        # Check dev image bypass
        is_dev_image = False
        if (
            self.settings.ALLOW_IMAGE_UPLOADS_FOR_DEV
            and kind
            and kind.mime in ["image/png", "image/jpeg"]
        ):
            is_dev_image = True

        if not is_dev_image:
            # If not a dev image, must be DICOM or NIfTI
            try:
                from io import BytesIO

                pydicom.dcmread(BytesIO(content))
                # It's a valid DICOM
            except InvalidDicomError:
                # Check for NIfTI (simple check, full NIfTI parsing needs nibabel)
                # GZIP NIfTI often starts with gzip magic bytes, uncompressed NIfTI has specific header
                if not (kind and kind.extension == "gz"):  # Very loose check, refine later
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid file format. Only DICOM/NIfTI allowed in production.",
                    ) from None

        # 3. SHA-256 Checksum
        sha256_hash = hashlib.sha256(content).hexdigest()

        # 4. Storage with UUID Key
        object_key = f"{patient_id}/{uuid.uuid4()}_{file.filename}"

        # (Storage implementation goes here, placeholder for now)
        # await storage_backend.upload(object_key, content)

        # 5. Database record creation (Placeholder)
        # study = Study(id=uuid.uuid4(), patient_id=patient_id, file_path=object_key, ...)
        # self.db.add(study)

        return {
            "file_name": file.filename,
            "object_key": object_key,
            "checksum": sha256_hash,
            "size": len(content),
        }
