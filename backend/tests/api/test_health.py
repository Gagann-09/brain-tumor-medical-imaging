"""API tests for health endpoints."""


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


def test_readiness(client):
    """GET /health/ready returns 200 when DB is reachable."""
    response = client.get("/health/ready")
    assert response.status_code == 200
