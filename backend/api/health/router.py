"""Health check endpoints for Kubernetes probes."""

try:
    import redis.asyncio as redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.config import get_settings
from services.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict:
    """Overall system health."""
    return {"status": "healthy", "service": "armt-gan-backend"}


@router.get("/health/live")
async def liveness() -> dict:
    """Liveness probe - returns 200 if the process is running."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(db: Session = Depends(get_db)) -> dict:
    """Readiness probe - returns 200 only if dependencies are reachable."""
    settings = get_settings()

    # 1. Database Check
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    # 2. Redis Check
    try:
        if HAS_REDIS:
            r = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await r.ping()
            await r.aclose()
            redis_ok = True
        else:
            redis_ok = False
    except Exception:
        redis_ok = False

    # 3. Storage Check (Placeholder - can be S3 ping or local dir check)
    storage_ok = True

    is_ready = db_ok and redis_ok and storage_ok
    status = "ready" if is_ready else "not_ready"
    status_code = 200 if is_ready else 503
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "checks": {"database": db_ok, "redis": redis_ok, "storage": storage_ok},
        },
    )
