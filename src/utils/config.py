"""Configuration, canonical serialization, hashes and credential-safe paths."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
# A separate process uses this only while preparing an isolated source installation.
ROOT = Path(os.getenv("DRUGASSIST_WORKSPACE", CODE_ROOT)).resolve()


def canonical(value):
    """Serialize a JSON-compatible value with stable key ordering and spacing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    """Return the SHA-256 digest of a value's canonical JSON representation."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utcnow():
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def read_json(path):
    """Read and decode a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write strict UTF-8 JSON through a temporary file and replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_config(path=None):
    """Load an explicit configuration file or the repository default."""
    return read_json(path or CODE_ROOT / "config/default.json")
