"""Redis-backed rate limiting middleware."""

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import redis.asyncio as redis
from core.config import get_settings

logger = structlog.get_logger()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding window rate limiter using Redis."""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.redis = redis.from_url(self.settings.REDIS_URL, decode_responses=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Enforce rate limits based on client IP or authenticated user."""
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        limit_str = self.settings.RATE_LIMIT_GLOBAL
        if path.startswith("/api/v1/auth"):
            limit_str = self.settings.RATE_LIMIT_AUTH
        elif path.startswith("/api/v1/upload"):
            limit_str = self.settings.RATE_LIMIT_UPLOAD
            
        try:
            req_count, time_window = limit_str.split("/")
            max_requests = int(req_count)
            window_seconds = 60 if time_window == "minute" else 3600
        except ValueError:
            max_requests = 100
            window_seconds = 60

        key = f"rate_limit:{client_ip}:{path}"
        try:
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, window_seconds)
            
            if current > max_requests:
                logger.warning("rate_limit_exceeded", ip=client_ip, path=path)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"error": {"type": "RateLimitExceeded", "message": "Too many requests"}}
                )
        except Exception as e:
            logger.error("redis_rate_limit_error", error=str(e))
            # Fail open if Redis is down
            pass

        return await call_next(request)
