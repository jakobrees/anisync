"""
Unit tests for app.security.

Focus:
- verify_password must NEVER 500 on a corrupted hash; it returns False.
- get_current_user must clear a session that contains a non-int user_id
  instead of crashing (avoids 500s from tampered cookies).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.security import get_current_user, hash_password, verify_password


def test_verify_password_correct():
    h = hash_password("AniSyncDemo123!")
    assert verify_password("AniSyncDemo123!", h) is True


def test_verify_password_wrong_returns_false():
    h = hash_password("AniSyncDemo123!")
    assert verify_password("not the password", h) is False


def test_verify_password_empty_inputs_return_false():
    """Empty/None inputs must short-circuit to False, not raise."""
    assert verify_password("", "any_hash") is False
    assert verify_password("password", "") is False
    assert verify_password(None, "any_hash") is False
    assert verify_password("password", None) is False


def test_verify_password_corrupted_hash_returns_false():
    """
    Regression: the original `except VerifyMismatchError` only handled the
    mismatch case. A malformed Argon2 hash (e.g. truncated DB row) raises
    InvalidHashError and used to bubble up as a 500. Now it must return False.
    """
    assert verify_password("password", "this-is-not-a-real-argon2-hash") is False
    assert verify_password("password", "$argon2id$v=19$m=garbage") is False


class _FakeSession:
    """Minimal fake of starlette.requests.Request.session for testing."""

    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def clear(self) -> None:
        self._data.clear()


class _FakeRequest:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session


class _FakeDB:
    """Stand-in for a SQLAlchemy Session that always returns None on .get."""

    def __init__(self, user=None) -> None:
        self._user = user

    def get(self, model, pk):  # noqa: ARG002 - signature matches SQLA
        return self._user


def test_get_current_user_unauthenticated_raises_401():
    request = _FakeRequest(_FakeSession({}))
    with pytest.raises(HTTPException) as excinfo:
        get_current_user(request, _FakeDB())
    assert excinfo.value.status_code == 401


def test_get_current_user_with_garbage_session_clears_and_raises_401():
    """Tampered session should not 500; we wipe it and force re-login."""
    session = _FakeSession({"user_id": "not-an-int"})
    request = _FakeRequest(session)
    with pytest.raises(HTTPException) as excinfo:
        get_current_user(request, _FakeDB())
    assert excinfo.value.status_code == 401
    # Session was cleared as a side-effect.
    assert session.get("user_id") is None


def test_get_current_user_missing_user_clears_session():
    """A valid-looking int that no longer maps to a row must clear the session."""
    session = _FakeSession({"user_id": 999})
    request = _FakeRequest(session)
    with pytest.raises(HTTPException) as excinfo:
        get_current_user(request, _FakeDB(user=None))
    assert excinfo.value.status_code == 401
    assert session.get("user_id") is None


def test_get_current_user_happy_path():
    session = _FakeSession({"user_id": 42})
    request = _FakeRequest(session)
    fake_user = SimpleNamespace(id=42, email="x@y.com", display_name="X")
    user = get_current_user(request, _FakeDB(user=fake_user))
    assert user is fake_user
