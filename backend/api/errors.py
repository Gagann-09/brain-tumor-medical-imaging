"""Centralized exception classes and FastAPI exception handlers."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.constants import ERR_INTERNAL_ERROR, ERR_VALIDATION_ERROR


class APIError(Exception):
    """Raise inside services / routers for controlled error responses."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError instances with RFC 7807 support."""
    cid = getattr(request.state, "correlation_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            # Standardized envelope
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "correlation_id": cid,
            # RFC 7807 fields
            "type": f"https://api.armt-gan.local/errors/{exc.error_code}",
            "title": exc.error_code.replace("_", " ").title(),
            "status": exc.status_code,
            "detail": exc.message,
            "instance": request.url.path,
        },
        headers={"Content-Type": "application/problem+json"},
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic / FastAPI validation errors with RFC 7807 support."""
    cid = getattr(request.state, "correlation_id", "unknown")
    return JSONResponse(
        status_code=422,
        content={
            # Standardized envelope
            "success": False,
            "error_code": ERR_VALIDATION_ERROR,
            "message": "Validation Error",
            "correlation_id": cid,
            # RFC 7807 fields
            "type": f"https://api.armt-gan.local/errors/{ERR_VALIDATION_ERROR}",
            "title": "Validation Error",
            "status": 422,
            "detail": str(exc.errors()),
            "instance": request.url.path,
        },
        headers={"Content-Type": "application/problem+json"},
    )


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions with RFC 7807 support."""
    cid = getattr(request.state, "correlation_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={
            # Standardized envelope
            "success": False,
            "error_code": ERR_INTERNAL_ERROR,
            "message": "An unexpected error occurred.",
            "correlation_id": cid,
            # RFC 7807 fields
            "type": f"https://api.armt-gan.local/errors/{ERR_INTERNAL_ERROR}",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred.",
            "instance": request.url.path,
        },
        headers={"Content-Type": "application/problem+json"},
    )
