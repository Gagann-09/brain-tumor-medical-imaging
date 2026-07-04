"""AWS S3 storage backend."""

from typing import BinaryIO

from .object_storage import ObjectStorage


class S3Storage(ObjectStorage):
    """Store files on AWS S3."""

    def __init__(
        self, bucket: str, region: str, access_key: str = "", secret_key: str = "",
    ) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )

    async def upload(self, key: str, data: BinaryIO, content_type: str = "") -> str:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.upload_fileobj(data, self.bucket, key, ExtraArgs=extra)
        return f"s3://{self.bucket}/{key}"

    async def download(self, key: str) -> bytes:
        import io

        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buf)
        return buf.getvalue()

    async def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    async def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
