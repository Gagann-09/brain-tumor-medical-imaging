"""Configuration factory - returns environment-specific settings."""

import os
from functools import lru_cache

from .base import BaseConfig
from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig

_CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


@lru_cache(maxsize=1)
def get_settings() -> BaseConfig:
    """Return cached settings instance for the current environment."""
    env = os.getenv("ENVIRONMENT", "development")
    config_cls = _CONFIG_MAP.get(env, DevelopmentConfig)
    return config_cls()
