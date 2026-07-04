"""ARMT-GAN Platform - Medical Imaging API entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.errors import APIError, api_error_handler, internal_error_handler, validation_error_handler
from api.health.router import router as health_router
from api.v1.router import api_v1_router
from core.config import get_settings
from core.constants import API_V1_PREFIX
from core.logging.config import setup_logging
from core.middleware.correlation import CorrelationIdMiddleware
from core.middleware.logging import RequestLoggingMiddleware
from core.middleware.security_headers import SecurityHeadersMiddleware
from core.middleware.database import DatabaseSessionMiddleware
from core.middleware.rate_limit import RateLimitMiddleware
from core.middleware.idempotency import IdempotencyMiddleware
from core.middleware.upload_size import UploadSizeLimitMiddleware
from core.middleware.observability import ObservabilityMiddleware
from core.middleware.timeout import TimeoutMiddleware
from core.middleware.api_version import APIVersionMiddleware
from prometheus_client import make_asgi_app

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    setup_logging(settings.LOG_LEVEL)
    logger = structlog.get_logger()
    logger.info("application_startup", version=settings.VERSION, environment=settings.ENVIRONMENT)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Medical imaging platform for brain tumor detection, segmentation, "
        "and analysis using ARMT-GAN and Hybrid CNN architectures. "
        "Provides XAI-driven diagnostic insights for clinical decision support."
    ),
    version=settings.VERSION,
    openapi_url=f"{API_V1_PREFIX}/openapi.json",
    docs_url=f"{API_V1_PREFIX}/docs",
    redoc_url=f"{API_V1_PREFIX}/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "System health and readiness probes"},
        {"name": "Auth", "description": "Authentication and authorization"},
        {"name": "Patients", "description": "Patient record management"},
        {"name": "Upload", "description": "Medical image upload and processing"},
        {"name": "Prediction", "description": "Tumor classification predictions"},
        {"name": "Segmentation", "description": "Tumor region segmentation"},
        {"name": "XAI", "description": "Explainable AI visualizations"},
        {"name": "Audit", "description": "Audit trail and compliance logging"},
        {"name": "Admin", "description": "Administrative operations"},
    ],
)

# ── Middleware (last added = outermost = runs first) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIVersionMiddleware)
app.add_middleware(TimeoutMiddleware, timeout_seconds=30)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(IdempotencyMiddleware)
app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(DatabaseSessionMiddleware)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIdMiddleware)

# ── Exception handlers ───────────────────────────────────────
app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, internal_error_handler)  # type: ignore[arg-type]

# ── Routers ──────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(api_v1_router, prefix=API_V1_PREFIX)

# ── Metrics ──────────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
