"""API tests for health endpoints."""
from unittest.mock import AsyncMock, patch


def test_health(client):
    """GET /health returns 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_liveness(client):
    """GET /health/live returns 200."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


@patch("api.health.router.HAS_REDIS", True)
@patch("api.health.router.redis")
def test_readiness(mock_redis, client):
    mock_r = AsyncMock()
    mock_redis.from_url.return_value = mock_r
    """GET /health/ready returns 200 when DB is reachable."""
    response = client.get("/health/ready")
    assert response.status_code == 200
