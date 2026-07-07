"""Unit tests for configuration loading."""

import os


def test_testing_config_loads():
    """Verify TestingConfig loads with ENVIRONMENT=testing."""
    os.environ["ENVIRONMENT"] = "testing"
    from core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.ENVIRONMENT == "testing"
    assert "test" in settings.SECRET_KEY.lower() or settings.DATABASE_URL.endswith("test.db")


def test_constants_defined():
    """Verify core constants are accessible."""
    from core.constants import ALL_ROLES, API_V1_PREFIX, TUMOR_TYPES

    assert API_V1_PREFIX == "/api/v1"
    assert len(TUMOR_TYPES) >= 4
    assert "admin" in ALL_ROLES
