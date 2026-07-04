"""Request logging middleware - structured log per request with timing."""

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with duration, status, user, and client metadata."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        user_id = getattr(request.state, "user_id", None)
        correlation_id = getattr(request.state, "correlation_id", None)

        logger.info(
            "http_request",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            user_id=str(user_id) if user_id else None,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        return response
