"""Administrative operations endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import require_admin
from schemas.common import APIResponse

from .schemas import AdminResponse

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats", response_model=APIResponse[AdminResponse])
async def get_stats(current_user=Depends(require_admin)):
    """Retrieve platform statistics (admin only)."""
    return APIResponse(data=AdminResponse())
