import logging

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import User


logger = logging.getLogger(__name__)

password_hasher = PasswordHasher()
AUTH_TOKEN_SALT = "anisync-auth-token-v1"
AUTH_TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 14


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.

    Never store plain-text passwords.
    """
    return password_hasher.hash(password)


def verify_password(password: str | None, password_hash: str | None) -> bool:
    """
    Verify a plain-text password against the stored Argon2 hash.
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


def _auth_token_signer() -> TimestampSigner:
    settings = get_settings()
    return TimestampSigner(settings.session_secret, salt=AUTH_TOKEN_SALT)


def create_auth_token(user_id: int) -> str:
    """
    Create a signed fallback auth token.

    This keeps production auth working even when a browser does not preserve
    the cross-site session cookie between the Vercel frontend and Render API.
    """
    signed = _auth_token_signer().sign(str(int(user_id)))
    return signed.decode("utf-8")


def get_user_id_from_auth_token(token: str | None) -> int | None:
    """
    Validate a signed auth token and return its user id.

    Returns None for missing, expired, malformed, or tampered tokens.
    """
    if not token:
        return None

    try:
        raw_user_id = _auth_token_signer().unsign(
            token,
            max_age=AUTH_TOKEN_MAX_AGE_SECONDS,
        )
        return int(raw_user_id.decode("utf-8"))
    except (BadSignature, SignatureExpired, TypeError, ValueError):
        return None


def _get_bearer_token(request: Request) -> str | None:
    headers = getattr(request, "headers", {}) or {}
    auth_header = headers.get("authorization", "")

    if not auth_header.lower().startswith("bearer "):
        return None

    return auth_header.split(" ", 1)[1].strip()


def get_current_user(request: Request, db: Session) -> User:
    """
    Read the current logged-in user.

    Preferred path:
    - signed session cookie

    Production fallback path:
    - signed Authorization: Bearer token
    """
    raw_user_id = request.session.get("user_id")

    if not raw_user_id:
        token_user_id = get_user_id_from_auth_token(_get_bearer_token(request))
        raw_user_id = token_user_id

    if not raw_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    user = db.get(User, user_id)
    if not user:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    return user