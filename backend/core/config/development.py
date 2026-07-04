"""Development environment overrides."""

from .base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Local development defaults."""

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"
