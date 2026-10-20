"""Real temporary authentication storage; retrieval is injected and offline."""

import sqlite3
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.auth.database import connect

ACCOUNT = {"email": "person@example.com", "password": "a-secure-test-password"}


@pytest.fixture
def auth_client(auth_service):
    search = Mock()
    search.health.return_value = {
        "status": "ready",
        "chromadb": "connected",
        "indexed_documents": 1,
        "indexed_chunks": 2,
    }
    search.search.side_effect = lambda query, k: {
        "query": query,
        "parsed": {"drug": None, "condition": None, "information_type": None},
        "no_match": True,
        "no_match_reason": "no_candidates",
        "method": "hybrid_rerank",
        "snapshot_id": "offline-fixture",
        "applied_filters": {},
        "score_kind": "cross_encoder_score",
        "score_note": "Raw relevance scores, not probabilities.",
        "threshold": {"method": "hybrid_rerank", "min_score": 0.5, "status": "experimental"},
        "results": [],
        "disclaimer": "Engineering fixture; not medical advice.",
    }
    factory = Mock(return_value=search)
    with TestClient(create_app(factory, lambda: auth_service)) as client:
        yield client, factory


def register_and_login(client):
    assert client.post("/auth/signup", json=ACCOUNT).status_code == 201
    response = client.post("/auth/login", json=ACCOUNT)
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["session_token"]}


def test_signup_normalizes_email_and_stores_only_argon2(auth_client, auth_service):
    client, factory = auth_client
    response = client.post("/auth/signup", json={**ACCOUNT, "email": "  PERSON@Example.com  "})
    assert response.status_code == 201 and response.json()["email"] == ACCOUNT["email"]
    assert set(response.json()) == {"id", "email", "created_at"}
    assert response.json()["created_at"].endswith("Z")
    with connect(auth_service.path) as db:
        row = db.execute("SELECT * FROM users").fetchone()
        assert row["password_hash"].startswith("$argon2id$")
        assert auth_service.passwords.verify(ACCOUNT["password"], row["password_hash"])
        assert ACCOUNT["password"] not in str(tuple(row))
    assert client.post("/auth/signup", json=ACCOUNT).status_code == 409
    factory.assert_not_called()


def test_login_and_me_use_persistent_hashed_session(auth_client, auth_service):
    from src.auth.service import AuthService, token_hash

    client, factory = auth_client
    headers = register_and_login(client)
    token = headers["Authorization"].split()[1]
    with connect(auth_service.path) as db:
        row = db.execute("SELECT * FROM sessions").fetchone()
        assert row["token_hash"] == token_hash(token)
        assert token not in str(tuple(row))
        assert row["expires_at"] - row["created_at"] == 3600
    reopened = AuthService(auth_service.path)
    assert reopened.current_user(token)["email"] == ACCOUNT["email"]
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200 and response.json()["email"] == ACCOUNT["email"]
    assert response.headers["cache-control"] == "no-store"
    assert "password" not in response.text and token not in response.text
    factory.assert_not_called()


def test_incorrect_credentials_have_same_generic_error(auth_client):
    client, factory = auth_client
    client.post("/auth/signup", json=ACCOUNT)
    wrong_password = client.post("/auth/login", json={**ACCOUNT, "password": "incorrect-password"})
    missing_user = client.post("/auth/login", json={**ACCOUNT, "email": "absent@example.com"})
    assert wrong_password.status_code == missing_user.status_code == 401
    assert wrong_password.json() == missing_user.json() == {"detail": "Incorrect email or password."}
    assert wrong_password.headers["www-authenticate"] == "Bearer"
    factory.assert_not_called()


@pytest.mark.parametrize("header", [None, "Bearer invalid-token", "Basic invalid", "Bearer"])
def test_search_rejects_unauthenticated_requests_before_retrieval(auth_client, header):
    client, factory = auth_client
    headers = {"Authorization": header} if header else {}
    assert client.get("/auth/me", headers=headers).status_code == 401
    assert client.post("/auth/logout", headers=headers).status_code == 401
    assert client.post("/search", json={"query": "metformin"}, headers=headers).status_code == 401
    factory.assert_not_called()


def test_authenticated_no_match_and_public_health(auth_client):
    client, factory = auth_client
    assert client.get("/health").status_code == 200
    headers = register_and_login(client)
    response = client.post("/search", json={"query": "unlistedcompound"}, headers=headers)
    assert response.status_code == 200 and response.json()["no_match"]
    assert response.headers["cache-control"] == "no-store"
    factory.return_value.search.assert_called_once_with("unlistedcompound", 5)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"email": "invalid", "password": "valid-password"},
        {"email": None, "password": "valid-password"},
        {"email": "person@example.com", "password": "short"},
        {"email": "person@example.com", "password": " " * 8},
        {"email": "person@example.com", "password": "x" * 129},
        {"email": "person@example.com", "password": 12345678},
        {**ACCOUNT, "unexpected": True},
    ],
)
def test_invalid_credentials_are_422_and_not_echoed(auth_client, payload):
    client, factory = auth_client
    for path in ("/auth/signup", "/auth/login"):
        response = client.post(path, json=payload)
        assert response.status_code == 422
        assert all("input" not in error for error in response.json()["detail"])
        assert ACCOUNT["password"] not in response.text
    factory.assert_not_called()


def test_rate_limit_recovers_after_window(auth_client, auth_service):
    client, factory = auth_client
    auth_service.attempt_limit = 2
    auth_service.clock = lambda: 1000
    for _ in range(2):
        assert client.post("/auth/login", json=ACCOUNT).status_code == 401
    blocked = client.post("/auth/login", json=ACCOUNT)
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
    assert client.post("/auth/signup", json=ACCOUNT).status_code == 429
    auth_service.clock = lambda: 1601
    assert client.post("/auth/signup", json=ACCOUNT).status_code == 201
    factory.assert_not_called()


def test_database_failures_are_sanitized(auth_service):
    def unavailable():
        raise sqlite3.OperationalError("private-account-database-path")

    with TestClient(create_app(auth_factory=unavailable)) as client:
        response = client.post("/auth/login", json=ACCOUNT)
        assert response.status_code == 503
        assert "private" not in response.text and "Traceback" not in response.text
