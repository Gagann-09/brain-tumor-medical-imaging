"""Request timeout middleware."""

import asyncio
import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforces a maximum execution time for requests."""

    def __init__(self, app, timeout_seconds: int = 30):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply timeout to request processing."""
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.error("request_timeout", path=request.url.path, timeout_seconds=self.timeout_seconds)
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"error": {"type": "GatewayTimeout", "message": "Request took too long to process"}}
            )
