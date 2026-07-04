"""Auth business logic."""

from sqlalchemy.orm import Session

from api.errors import APIError
from core.constants import ERR_AUTH_INVALID_CREDENTIALS, ERR_RESOURCE_CONFLICT
from core.security.hashing import hash_password, verify_password
from core.security.jwt import create_access_token
from repositories.user import UserRepository

from .schemas import TokenResponse, UserCreate, UserResponse


class AuthService:
    """Handles registration and authentication."""

    def __init__(self, db: Session) -> None:
        self.user_repo = UserRepository(db)

    def register(self, data: UserCreate) -> UserResponse:
        """Register a new user account."""
        existing = self.user_repo.get_by_email(data.email)
        if existing:
            raise APIError(409, ERR_RESOURCE_CONFLICT, "Email already registered")

        user = self.user_repo.create(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )
        return UserResponse.model_validate(user)

    def authenticate(self, email: str, password: str) -> TokenResponse:
        """Verify credentials and return a JWT."""
        user = self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise APIError(401, ERR_AUTH_INVALID_CREDENTIALS, "Invalid email or password")

        token = create_access_token({"sub": str(user.id)})
        return TokenResponse(access_token=token)
