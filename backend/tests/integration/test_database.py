"""Integration tests for database connectivity."""

from sqlalchemy import text


def test_db_connection(db_session):
    """Verify the test database is reachable."""
    result = db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
