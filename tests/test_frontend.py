"""Streamlit screens exercise the real auth API through an in-process HTTP client."""

from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api.main import create_app
from src.auth.service import AuthError

APP = Path(__file__).resolve().parents[1] / "frontend/app.py"
ACCOUNT = {"email": "reviewer@example.com", "password": "test-password-123"}


def search_response(query, k):
    missing = query == "unknown drug"
    return {
        "query": query,
        "parsed": {"drug": None, "condition": None, "information_type": None},
        "no_match": missing,
        "no_match_reason": "outside_label_scope" if missing else None,
        "method": "hybrid_rerank",
        "snapshot_id": "ui-fixture",
        "applied_filters": {},
        "score_kind": "cross_encoder_score",
        "score_note": "Raw relevance scores, not probabilities.",
        "threshold": {"method": "hybrid_rerank", "min_score": 0.5, "status": "experimental"},
        "disclaimer": "Engineering fixture; not medical advice.",
        "results": []
        if missing
        else [
            {
                "rank": 1,
                "score": 7.25,
                "drug_name": "metformin",
                "label_id": "fixture-label",
                "set_id": "fixture-set",
                "document_id": "fixture-document",
                "chunk_id": "fixture-chunk",
                "snapshot_id": "ui-fixture",
                "section": "contraindications",
                "text": "Engineering FDA excerpt for UI testing only.",
                "start": 0,
                "end": 10,
                "effective_time": "20240101",
                "source_url": "https://api.fda.gov/drug/label.json",
                "source_record_hash": "fixture-record-hash",
                "source_body_sha256": "fixture-body-hash",
            }
        ],
    }


@pytest.fixture
def ui(auth_service, monkeypatch):
    service = Mock()
    service.search.side_effect = search_response
    monkeypatch.setenv("DRUGASSIST_API_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("DRUGASSIST_REQUEST_TIMEOUT", "120")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    calls = []
    with TestClient(create_app(lambda: service, lambda: auth_service)) as client:

        def request(method, url, **kwargs):
            path = urlsplit(url).path
            calls.append((method, path))
            return client.request(method, path, json=kwargs.get("json"), headers=kwargs.get("headers"))

        monkeypatch.setattr(httpx, "request", request)
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        assert not app.exception
        yield app, calls, service


def click(app, label):
    return next(button for button in app.button if button.label == label).click().run()


def log_in(app, auth_service):
    auth_service.signup(**ACCOUNT)
    app.text_input[0].set_value(ACCOUNT["email"])
    app.text_input[1].set_value(ACCOUNT["password"])
    click(app, "Log in")
    assert not app.exception


def test_signup_login_search_and_logout(ui, auth_service):
    app, calls, service = ui
    app.radio[0].set_value("Sign up").run()
    app.text_input[0].set_value(ACCOUNT["email"])
    app.text_input[1].set_value(ACCOUNT["password"])
    app.text_input[2].set_value(ACCOUNT["password"])
    click(app, "Sign up")
    assert app.success and not app.exception
    app.radio[0].set_value("Log in").run()
    app.text_input[0].set_value(ACCOUNT["email"])
    app.text_input[1].set_value(ACCOUNT["password"])
    click(app, "Log in")
    token = app.session_state["token"]
    assert ("GET", "/auth/me") in calls
    assert not app.exception
    app.text_area[0].set_value("metformin contraindications")
    app.number_input[0].set_value(2)
    click(app, "Search FDA labels")
    service.search.assert_called_once_with("metformin contraindications", 2)
    assert any("cross_encoder_score: 7.250" in element.value for element in app.caption)
    assert any("Engineering FDA excerpt" in element.value for element in app.text)
    assert app.get("link_button")
    assert not app.exception
    for elements in (app.text, app.caption, app.markdown):
        assert all(token not in element.value for element in elements)
    click(app, "Log out")
    assert app.radio[0].value == "Log in"
    with pytest.raises(KeyError):
        app.session_state["token"]
    with pytest.raises(AuthError):
        auth_service.current_user(token)


def test_password_confirmation_is_checked_before_api_call(ui):
    app, calls, _service = ui
    app.radio[0].set_value("Sign up").run()
    app.text_input[0].set_value(ACCOUNT["email"])
    app.text_input[1].set_value(ACCOUNT["password"])
    app.text_input[2].set_value("different-password")
    click(app, "Sign up")
    assert "do not match" in app.error[0].value
    assert not calls and not app.exception


def test_no_match_and_expired_session_return_to_login(ui, auth_service):
    app, _calls, _service = ui
    log_in(app, auth_service)
    app.text_area[0].set_value("unknown drug")
    click(app, "Search FDA labels")
    assert any("No good match found" in element.value for element in app.info)
    assert not app.error and not app.exception
    original_clock = auth_service.clock
    auth_service.clock = lambda: original_clock() + 3601
    app.run()
    assert app.radio[0].value == "Log in"
    assert any("session has ended" in element.value for element in app.info)
    with pytest.raises(KeyError):
        app.session_state["results"]
    assert not app.exception


def test_backend_connection_error_is_friendly(ui, monkeypatch):
    app, _calls, _service = ui

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("private-network-details")

    monkeypatch.setattr(httpx, "request", unavailable)
    app.text_input[0].set_value(ACCOUNT["email"])
    app.text_input[1].set_value(ACCOUNT["password"])
    click(app, "Log in")
    assert "Cannot reach DrugAssist" in app.error[0].value
    assert "private" not in app.error[0].value and not app.exception


def test_ranked_passages_summary_and_source_details_persist_without_searching_again(ui, auth_service):
    app, _calls, service = ui
    log_in(app, auth_service)
    response = search_response("metformin contraindications", 2)
    response["results"].append(
        {
            **response["results"][0],
            "rank": 2,
            "score": 6.5,
            "set_id": "second-label-set",
            "chunk_id": "second-passage",
        }
    )
    service.search.side_effect = None
    service.search.return_value = response
    app.text_area[0].set_value("metformin contraindications")
    click(app, "Search FDA labels")
    assert any("2 passages from 2 FDA label sets | Search time:" in element.value for element in app.caption)
    assert any("Label date: 01 Jan 2024" in element.value for element in app.caption)
    passages = [element for element in app.expander if element.label == "Read FDA passage"]
    assert len(passages) == 2 and passages[0].proto.expanded and not passages[1].proto.expanded
    assert app.json and "fixture-record-hash" in app.json[0].value
    elapsed = app.session_state["search_seconds"]
    app.run()
    assert app.session_state["search_seconds"] == elapsed
    service.search.assert_called_once_with("metformin contraindications", 5)
    assert not app.exception


def test_failed_search_clears_old_results_and_shows_error_status(ui, auth_service):
    from src.services.search_service import ServiceUnavailable

    app, _calls, service = ui
    log_in(app, auth_service)
    app.text_area[0].set_value("metformin contraindications")
    click(app, "Search FDA labels")
    assert app.session_state["results"]["results"]
    service.search.side_effect = ServiceUnavailable
    app.text_area[0].set_value("metformin warnings")
    click(app, "Search FDA labels")
    assert any("temporarily unavailable" in element.value for element in app.error)
    assert app.status and app.status[0].state == "error"
    assert not any("Engineering FDA excerpt" in element.value for element in app.text)
    with pytest.raises(KeyError):
        app.session_state["results"]
    with pytest.raises(KeyError):
        app.session_state["search_seconds"]
    assert not app.exception

