"""Production environment overrides."""

from .base import BaseConfig


class ProductionConfig(BaseConfig):
    """Production hardened settings."""

    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "WARNING"
