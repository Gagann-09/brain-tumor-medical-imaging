from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func

from schemas.upload import UploadState

from .base import Base


class UploadRecordModel(Base):
    __tablename__ = "upload_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_id = Column(String, unique=True, index=True, nullable=False)
    filename = Column(String, nullable=False)
    status = Column(String, default=UploadState.UPLOADED.value, index=True)
    user_id = Column(String, index=True, nullable=True)
    storage_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
