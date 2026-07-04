"""Idempotency-key middleware."""

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import redis.asyncio as redis
from core.config import get_settings

logger = structlog.get_logger()

class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Ensures POST requests with Idempotency-Key are only processed once."""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.redis = redis.from_url(self.settings.REDIS_URL, decode_responses=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check for Idempotency-Key and handle duplicate requests."""
        if request.method not in ["POST", "PUT", "PATCH"]:
            return await call_next(request)

        idempotency_key = request.headers.get("Idempotency-Key")
        if not idempotency_key:
            return await call_next(request)

        client_id = request.headers.get("Authorization", "anonymous")
        cache_key = f"idempotency:{client_id}:{idempotency_key}"

        try:
            # Simple approach: Check if key exists (in progress or done)
            # In a real implementation, you'd store the full response and return it.
            is_duplicate = await self.redis.setnx(cache_key, "in_progress")
            if not is_duplicate:
                logger.info("idempotency_key_duplicate", key=idempotency_key)
                return JSONResponse(
                    status_code=409,
                    content={"error": {"type": "Conflict", "message": "Duplicate request detected"}}
                )
            
            # Expire after 24 hours to avoid indefinite storage
            await self.redis.expire(cache_key, 86400)
            
            response = await call_next(request)
            # Future enhancement: save the response here so subsequent requests get the same response
            return response
        except Exception as e:
            logger.error("redis_idempotency_error", error=str(e))
            return await call_next(request)
