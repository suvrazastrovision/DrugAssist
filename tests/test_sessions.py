"""Expiry, revocation, independent sessions and SQLite constraints."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.auth.database import connect
from src.auth.service import AuthError


def login(auth):
    return auth.login("person@example.com", "test-password-123")["session_token"]


def test_expired_session_is_rejected_before_cleanup(auth_service):
    auth_service.clock = lambda: 1000
    auth_service.signup("person@example.com", "test-password-123")
    token = login(auth_service)
    auth_service.clock = lambda: 4599
    assert auth_service.current_user(token)["email"] == "person@example.com"
    auth_service.clock = lambda: 4600
    with pytest.raises(AuthError) as error:
        auth_service.current_user(token)
    assert error.value.status == 401
    with connect(auth_service.path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    assert auth_service.cleanup_expired() == 1


def test_logout_revokes_only_current_session(auth_service):
    auth_service.signup("person@example.com", "test-password-123")
    first, second = login(auth_service), login(auth_service)
    assert first != second
    auth_service.logout(first)
    with pytest.raises(AuthError):
        auth_service.current_user(first)
    assert auth_service.current_user(second)["email"] == "person@example.com"


def test_http_logout_and_expiry_reject_protected_requests(auth_service):
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    auth_service.clock = lambda: 1000
    auth_service.signup("person@example.com", "test-password-123")
    revoked = {"Authorization": "Bearer " + login(auth_service)}
    expired = {"Authorization": "Bearer " + login(auth_service)}
    with TestClient(create_app(auth_factory=lambda: auth_service)) as client:
        response = client.post("/auth/logout", headers=revoked)
        assert response.status_code == 204 and not response.content
        assert client.get("/auth/me", headers=revoked).status_code == 401
        auth_service.clock = lambda: 4600
        for headers in (revoked, expired):
            assert client.get("/auth/me", headers=headers).status_code == 401
            assert client.post("/search", json={"query": "metformin"}, headers=headers).status_code == 401


def test_foreign_keys_and_user_deletion_revoke_sessions(auth_service):
    auth_service.signup("person@example.com", "test-password-123")
    token = login(auth_service)
    with connect(auth_service.path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO sessions(user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (999, "missing-user", 1, 2),
            )
        db.execute("DELETE FROM users")
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    with pytest.raises(AuthError):
        auth_service.current_user(token)


def test_concurrent_duplicate_signup_has_one_winner(auth_service):
    def signup(_):
        try:
            auth_service.signup("person@example.com", "test-password-123")
            return 201
        except AuthError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(signup, range(2))) == [201, 409]
