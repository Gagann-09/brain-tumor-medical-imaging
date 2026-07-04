"""Feature flags utilities."""

from . import get_settings

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific boolean feature flag is enabled in the configuration."""
    settings = get_settings()
    return getattr(settings, feature_name, False)

def get_feature_flag(feature_name: str, default: any = None) -> any:
    """Get the value of a feature flag, supporting non-boolean flags (like AI_MODEL_VERSION)."""
    settings = get_settings()
    return getattr(settings, feature_name, default)
