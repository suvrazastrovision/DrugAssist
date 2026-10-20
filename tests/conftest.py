"""Synthetic engineering fixtures only; never used as regulatory data/benchmark."""

import numpy as np
import pytest

from src.preprocessing.normalization import normalize_record


@pytest.fixture(autouse=True)
def offline_providers(monkeypatch):
    """Tests neither read .env nor call an external HTTP provider."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FDA_API_KEY", raising=False)
    monkeypatch.setattr("src.embeddings.embedder.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.ingestion.api_client.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.auth.service.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.services.answer_service.load_dotenv", lambda *args, **kwargs: False)

    def no_http(*args, **kwargs):
        raise AssertionError("External HTTP is disabled during tests")

    monkeypatch.setattr("requests.sessions.Session.request", no_http)
    monkeypatch.setattr("httpx.HTTPTransport.handle_request", no_http)


@pytest.fixture
def auth_service(tmp_path):
    from src.auth.service import AuthService

    return AuthService(tmp_path / "accounts.sqlite3")


@pytest.fixture
def auth_headers(auth_service):
    auth_service.signup("reviewer@example.com", "test-password-123")
    token = auth_service.login("reviewer@example.com", "test-password-123")["session_token"]
    return {"Authorization": f"Bearer {token}"}


class FakeEmbedder:
    """Deterministic engineering-only vectors, never a production fallback."""

    fingerprint = "offline-api-test-space-v1"
    dimensions = 4

    def __init__(self):
        self.calls, self.fail = 0, False

    def encode(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake-provider-secret-and-internal-details")
        return np.array(
            [
                [float(term in text.casefold()) for term in ("metformin", "warfarin", "warnings")] + [1.0]
                for text in texts
            ],
            dtype=np.float32,
        ).reshape(-1, self.dimensions)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def record():
    return {
        "set_id": "11111111-1111-4111-8111-111111111111",
        "id": "test-label-version",
        "effective_time": "20240101",
        "version": "1",
        "openfda": {
            "generic_name": ["TEST COMPOUND"],
            "brand_name": ["Engineering fixture"],
            "substance_name": ["TEST COMPOUND"],
        },
        "warnings": ["Do not administer 2.5 mg/kg. Keep the safety qualifier intact. " * 15],
        "dosage_and_administration": ["Engineering test text only. No clinical content."],
    }


@pytest.fixture
def document(record):
    return normalize_record(
        {
            "record": record,
            "kind": "fda_full",
            "provenance": {"body_sha256": "test-body-hash", "retrieved_at": "2024-01-01T00:00:00+00:00"},
        },
        "test-snapshot",
    )


@pytest.fixture
def chunks(document):
    from src.preprocessing.chunking import chunk_documents

    return chunk_documents([document], size=32, overlap=4)
