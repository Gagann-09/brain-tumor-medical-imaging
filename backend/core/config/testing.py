"""Testing environment overrides."""

from .base import BaseConfig


class TestingConfig(BaseConfig):
    """Test suite defaults - uses SQLite in-memory."""

    ENVIRONMENT: str = "testing"
    DATABASE_URL: str = "sqlite:///./test.db"
    LOG_LEVEL: str = "DEBUG"
    SECRET_KEY: str = "test-secret-key-not-for-production"
