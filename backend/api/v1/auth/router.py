"""Authentication endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from schemas.common import APIResponse
from services.database import get_db

from .schemas import TokenRequest, TokenResponse, UserCreate, UserResponse
from .service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=APIResponse[UserResponse], status_code=201)
async def register(data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    service = AuthService(db)
    user = service.register(data)
    return APIResponse(data=user, message="User registered successfully")


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(data: TokenRequest, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT."""
    service = AuthService(db)
    token = service.authenticate(data.email, data.password)
    return APIResponse(data=token)


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_me(current_user=Depends(get_current_user)):
    """Return the current authenticated user."""
    return APIResponse(data=UserResponse.model_validate(current_user))
