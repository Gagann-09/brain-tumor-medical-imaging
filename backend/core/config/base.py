"""Base configuration with pydantic-settings - all defaults live here."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Shared settings across all environments."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ── Application ──────────────────────────────────────
    PROJECT_NAME: str = "ARMT-GAN Platform"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/armt_gan"
    
    # ── Redis ────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Auth / JWT ───────────────────────────────────────
    SECRET_KEY: str = "change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── Storage ──────────────────────────────────────────
    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_PATH: str = "./uploads"
    S3_BUCKET: str = ""
    S3_REGION: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""

    # ── Rate Limiting ────────────────────────────────────
    RATE_LIMIT_GLOBAL: str = "100/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_AUTH: str = "5/minute"

    # ── Feature Flags ────────────────────────────────────
    EXPERIMENTAL_MODELS: bool = False
    CLINICAL_FEATURES: bool = False
    ALLOW_IMAGE_UPLOADS_FOR_DEV: bool = False
    AI_MODEL_VERSION: str = "v1.0.0"

    @model_validator(mode="after")
    def validate_config(self) -> "BaseConfig":
        """Centralized config validation."""
        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND is 's3'")
        return self

