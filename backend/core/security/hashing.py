"""Password hashing using pwdlib with Argon2id."""

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

_hasher = PasswordHash((Argon2Hasher(),))


def hash_password(plain: str) -> str:
    """Hash a plaintext password with Argon2id."""
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its Argon2id hash."""
    return _hasher.verify(plain, hashed)
