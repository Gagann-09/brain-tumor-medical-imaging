"""Shared Pydantic response envelopes used across all endpoints."""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class APIResponse[T](BaseModel):
    """Standard success response."""

    success: bool = True
    data: T | None = None
    message: str | None = None


class APIErrorResponse(BaseModel):
    """Standard error response - returned by all error handlers."""

    success: bool = False
    error_code: str
    message: str
    correlation_id: str


class PaginatedData[T](BaseModel):
    """Paginated list wrapper."""

    items: list[T] = []
    total: int
    page: int
    page_size: int
    total_pages: int
