"""Database session middleware with proper lifecycle and rollback."""

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from services.database import SessionLocal

logger = structlog.get_logger()


class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    """Middleware for database session lifecycle management."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Inject database session into request state."""
        session = SessionLocal()
        request.state.db = session
        try:
            response = await call_next(request)
            session.commit()
            return response
        except Exception as e:
            logger.error("database_session_error", error=str(e))
            session.rollback()
            raise
        finally:
            session.close()
