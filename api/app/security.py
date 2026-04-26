import logging

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.models import User


logger = logging.getLogger(__name__)

password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.

    Never store plain-text passwords.
    """
    return password_hasher.hash(password)


def verify_password(password: str | None, password_hash: str | None) -> bool:
    """
    Verify a plain-text password against the stored Argon2 hash.

    Returns False on any argon2 failure (mismatch, malformed/legacy hash,
    etc.) so a corrupted DB row can never crash login with a 500. Genuine
    bugs are still surfaced via the warning log.
    """
    if not password or not password_hash:
        return False
    try:
        return password_hasher.verify(password_hash, password)
    except Argon2Error:
        return False
    except Exception:
        logger.exception("verify_password: unexpected error during verification")
        return False


def get_current_user(request: Request, db: Session) -> User:
    """
    Read the current logged-in user from the signed session cookie.
    """
    raw_user_id = request.session.get("user_id")
    if not raw_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        # Tampered or otherwise malformed session: clear it and force re-login
        # instead of 500-ing.
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    user = db.get(User, user_id)
    if not user:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    return user
