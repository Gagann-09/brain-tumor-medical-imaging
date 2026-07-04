"""API Version response header middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from core.config import get_settings

class APIVersionMiddleware(BaseHTTPMiddleware):
    """Injects X-API-Version header into responses."""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add X-API-Version to response."""
        response = await call_next(request)
        response.headers["X-API-Version"] = self.settings.VERSION
        return response
