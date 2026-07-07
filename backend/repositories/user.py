"""User repository - domain-specific data access."""

from uuid import UUID

from sqlalchemy.orm import Session

from models.user import User


class UserRepository:
    """Data access layer for User entities."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, user_id: UUID) -> User | None:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> User | None:
        return self.db.query(User).filter(User.email == email).first()

    def create(
        self,
        email: str,
        hashed_password: str,
        full_name: str,
        role: str = "viewer",
    ) -> User:
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def list_active(self, skip: int = 0, limit: int = 20) -> list[User]:
        return self.db.query(User).filter(User.is_active.is_(True)).offset(skip).limit(limit).all()

    def deactivate(self, user_id: UUID) -> User | None:
        user = self.get_by_id(user_id)
        if user:
            user.is_active = False
            self.db.commit()
            self.db.refresh(user)
        return user
