"""Observability middleware for metrics and tracing."""

import time

import structlog
from fastapi import Request, Response
from prometheus_client import Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for request latency histograms and basic telemetry."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Record request duration."""
        start_time = time.time()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
            logger.info(
                "request_completed", method=method, path=path, duration_ms=round(duration * 1000, 2)
            )
