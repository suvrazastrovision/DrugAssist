import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.ingestion.api_client import APIClient, APIError
from src.preprocessing.normalization import normalize_record
from src.quality.validators import validate_record


def test_normalization_preserves_original(record):
    before = deepcopy(record)
    doc = normalize_record({"record": record, "kind": "fda_full", "provenance": {}}, "snapshot")
    assert record == before
    assert doc["sections"]["warnings"] == record["warnings"][0]
    assert "Do not" in doc["sections"]["warnings"] and "2.5 mg/kg" in doc["sections"]["warnings"]
    assert doc["content_status"] == "authoritative_full"


@pytest.mark.parametrize(
    "field,value", [("set_id", "broken"), ("effective_time", "20241301"), ("warnings", [42]), ("openfda", [])]
)
def test_invalid_schema(record, field, value):
    record[field] = value
    assert any(i["severity"] == "error" for i in validate_record(record, "fda_full"))


def test_future_dates_are_observed_not_corrected(record):
    record["effective_time"] = "20990101"
    issues = validate_record(record, "fda_full")
    assert any(i["issue"] == "future effective date" for i in issues)
    assert record["effective_time"] == "20990101"


def test_summary_is_never_marked_complete():
    d = normalize_record(
        {
            "record": {"set_id": "a", "indications_snippet": "snippet"},
            "kind": "recent_summary",
            "provenance": {},
        },
        "snap",
    )
    assert d["truncation_possible"] and d["sections"] == {"indications_snippet": "snippet"}


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self.content = status, body
        self.ok, self.headers = status < 400, headers or {"Content-Type": "application/json"}

    def json(self):
        return json.loads(self.content)


def fake_session(responses):
    items = iter(responses)
    return SimpleNamespace(headers={}, get=lambda *args, **kwargs: next(items))


def test_raw_preserved_before_parse_failure(tmp_path):
    body = b"not valid JSON \xff"
    client = APIClient(raw_dir=tmp_path, session=fake_session([FakeResponse(200, body)]))
    with pytest.raises(APIError, match="expected JSON"):
        client.get("https://example.invalid")
    assert (tmp_path / (hashlib.sha256(body).hexdigest() + ".body")).read_bytes() == body


def test_retry_after_and_no_secret_in_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("FDA_API_KEY", "FAKE-SECRET-MUST-NOT-BE-RECORDED")
    sleeps = []
    client = APIClient(
        raw_dir=tmp_path,
        sleep=sleeps.append,
        session=fake_session(
            [FakeResponse(429, b"{}", {"Retry-After": "2"}), FakeResponse(200, b'{"results":[]}')]
        ),
    )
    result, meta = client.get("https://api.fda.gov/drug/label.json", provider="fda")
    assert result == {"results": []} and sleeps == [2]
    assert "FAKE-SECRET" not in str(meta)
    assert all("FAKE-SECRET" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_unsupported_pagination_limit(tmp_path):
    client = APIClient(raw_dir=tmp_path)
    with pytest.raises(ValueError, match="pagination"):
        client.recent(201)
