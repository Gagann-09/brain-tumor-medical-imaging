"""Request body size limit middleware."""

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limits the size of incoming requests (especially uploads)."""

    def __init__(self, app, max_upload_size: int = 100 * 1024 * 1024):  # Default 100MB
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check Content-Length header against configured limit."""
        if request.method in ["POST", "PUT"]:
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_upload_size:
                logger.warning(
                    "upload_size_limit_exceeded", size=content_length, limit=self.max_upload_size
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": {
                            "type": "PayloadTooLarge",
                            "message": "Request body exceeds maximum allowed size",
                        }
                    },
                )
        return await call_next(request)
