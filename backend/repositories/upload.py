from sqlalchemy import select
from sqlalchemy.orm import Session

from models.upload import UploadRecordModel
from schemas.upload import UploadRecord


class UploadRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, upload_data: UploadRecord) -> UploadRecordModel:
        db_upload = UploadRecordModel(
            upload_id=upload_data.upload_id,
            filename=upload_data.filename,
            status=upload_data.status.value,
            user_id=upload_data.user_id,
            storage_path=upload_data.storage_path,
        )
        self.db.add(db_upload)
        self.db.commit()
        self.db.refresh(db_upload)
        return db_upload

    def get_by_upload_id(self, upload_id: str) -> UploadRecordModel | None:
        return (
            self.db.execute(
                select(UploadRecordModel).where(UploadRecordModel.upload_id == upload_id)
            )
            .scalars()
            .first()
        )

    def update(self, upload_id: str, update_data: dict) -> UploadRecordModel | None:
        db_upload = self.get_by_upload_id(upload_id)
        if db_upload:
            for key, value in update_data.items():
                if hasattr(db_upload, key):
                    setattr(db_upload, key, value)
            self.db.commit()
            self.db.refresh(db_upload)
        return db_upload
