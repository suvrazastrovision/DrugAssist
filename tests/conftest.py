"""Offline, isolated engineering fixtures; no prepared data or user caches required."""

import atexit
import hashlib
import ipaddress
import os
import socket
import sys
import tempfile
from contextvars import ContextVar
from pathlib import Path

# Set cache locations before test collection imports third-party model libraries.
_session_directory = tempfile.TemporaryDirectory(prefix="drugassist-tests-")
atexit.register(_session_directory.cleanup)
_CACHE_FOLDERS = {
    "TIKTOKEN_CACHE_DIR": "tokenizer", "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub",
    "TRANSFORMERS_CACHE": "hf/transformers", "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg", "STREAMLIT_HOME": "streamlit",
}
for _name, _folder in _CACHE_FOLDERS.items():
    os.environ[_name] = str(Path(_session_directory.name) / _folder)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import numpy as np  # noqa: E402 -- Set model cache locations before importing providers.
import pytest  # noqa: E402 -- Keep test imports after isolated environment setup.

from src.preprocessing.normalization import normalize_record  # noqa: E402 -- Requires the isolated environment.

_ROOT = Path(__file__).resolve().parents[1]
_protected = [_ROOT / name for name in ("data", "results", "reports", ".env")]
_protected.extend([
    Path.home() / ".cache/huggingface", Path.home() / ".cache/torch",
    Path(tempfile.gettempdir()) / "data-gym-cache",
])
_guard_active = False
_opening_path = ContextVar("opening_path", default=None)


def _directory_path(descriptor):
    # Linux rmtree opens children relative to an already-open directory. The
    # open audit event omits dir_fd, so resolve it before calling os.open.
    for folder in ("/proc/self/fd", "/dev/fd"):
        try:
            return Path(os.readlink(f"{folder}/{descriptor}"))
        except OSError:
            continue
    raise AssertionError("Cannot resolve directory descriptor for the test filesystem guard")


def _protect_runtime_files(event, args):
    if _guard_active and event in {"open", "os.mkdir", "os.listdir", "os.scandir", "sqlite3.connect"}:
        path = _opening_path.get() if event == "open" and _opening_path.get() is not None else args[0]
        if event == "os.mkdir" and args[2] != -1 and not Path(os.fsdecode(path)).is_absolute():
            path = _directory_path(args[2]) / os.fsdecode(path)
        if isinstance(path, (str, bytes, os.PathLike)):
            target = Path(os.fsdecode(path)).resolve()
            if any(target == parent or target.is_relative_to(parent) for parent in _protected):
                raise AssertionError(f"Tests must use temporary stores, not runtime path: {target}")


sys.addaudithook(_protect_runtime_files)


@pytest.fixture(autouse=True)
def offline_providers(monkeypatch, tmp_path, request):
    """Prevent offline tests from reading .env or calling external HTTP providers."""
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FDA_API_KEY", raising=False)
    monkeypatch.setattr("src.embeddings.embedder.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.ingestion.api_client.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.auth.service.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.services.answer_service.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.setup.dotenv_values", lambda *args, **kwargs: {})
    # Chroma's Pydantic settings independently inspect .env when first imported.
    monkeypatch.setattr(
        "pydantic_settings.sources.providers.dotenv.dotenv_values", lambda *args, **kwargs: {}
    )

    def no_http(*args, **kwargs):
        raise AssertionError("External HTTP is disabled during tests")

    monkeypatch.setattr("requests.sessions.Session.request", no_http)
    monkeypatch.setattr("httpx.HTTPTransport.handle_request", no_http)

    original_connect = socket.socket.connect

    def no_connect(connection, address):
        # Windows asyncio creates a loopback socket pair for its own event loop.
        if connection.family not in {socket.AF_INET, socket.AF_INET6}:
            return original_connect(connection, address)
        try:
            loopback = ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            loopback = address[0] == "localhost"
        if loopback:
            return original_connect(connection, address)
        raise AssertionError("External sockets are disabled during tests")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    original_open = os.open

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        target = path
        if dir_fd is not None and not Path(os.fsdecode(path)).is_absolute():
            target = _directory_path(dir_fd) / os.fsdecode(path)
        token = _opening_path.set(target)
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            _opening_path.reset(token)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {guarded_open}
                        if original_open in os.supports_dir_fd else os.supports_dir_fd)
    monkeypatch.setenv("DRUGASSIST_AUTH_DB", str(tmp_path / "accounts.sqlite3"))
    for name, folder in _CACHE_FOLDERS.items():
        monkeypatch.setenv(name, str(tmp_path / "caches" / folder))
    import tiktoken.load

    def vocabulary(path):
        if path != "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken":
            raise AssertionError(f"No checked-in tokenizer asset for {path}")
        body = (_ROOT / "tests/assets/cl100k_base.tiktoken").read_bytes()
        assert hashlib.sha256(body).hexdigest() == (
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
        )
        return body

    monkeypatch.setattr(tiktoken.load, "read_file", vocabulary)
    global _guard_active
    _guard_active = True
    try:
        yield
    finally:
        _guard_active = False


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
