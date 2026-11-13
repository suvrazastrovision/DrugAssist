"""Metadata-only tracing with no exporter or live credentials."""

import sys
import json
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import observability as obs
from src.services.answer_service import AnswerSettings
from src.services.answer_service import AnswerService, OpenAIAnswerWriter
from test_api import api_service as api_service


@pytest.fixture
def traces(monkeypatch):
    records = []
    active = ContextVar("test_parent", default=None)

    @contextmanager
    def start(**fields):
        record = {**fields, "parent": active.get()}
        records.append(record)
        token = active.set(fields["name"])
        try:
            yield SimpleNamespace(update=lambda **updates: record.update(updates))
        finally:
            active.reset(token)

    client = SimpleNamespace(start_as_current_observation=start, shutdown=Mock())
    monkeypatch.setattr(obs, "_get_client", lambda: client)
    return records


def test_nesting_usage_and_private_content(traces):
    @obs.traced("retrieval")
    def retrieve(query, k=5):
        return {"query": query, "results": [{"text": "PRIVATE PASSAGE"}], "method": "bm25"}

    @obs.traced("generation")
    def generate(payload, settings):
        obs.record_generation({"model": "test-model", "usage": {
            "input_tokens": 12, "output_tokens": 3, "total_tokens": 15, "private": "SECRET",
        }})
        return {"answer": payload}

    @obs.traced("answer")
    def answer(query, k=5, audience="patient"):
        retrieve(query, k)
        generate("PRIVATE ANSWER", AnswerSettings())
        return {"answer_status": "generated", "results": [], "answer": "PRIVATE ANSWER"}

    answer("PRIVATE QUESTION", 3, "clinician")
    assert [(row["name"], row["parent"]) for row in traces] == [
        ("answer", None), ("retrieval", "answer"), ("generation", "answer"),
    ]
    assert traces[0]["metadata"]["audience"] == "clinician"
    assert traces[1]["metadata"]["result_count"] == 1
    assert traces[2]["usage_details"] == {"input": 12, "output": 3, "total": 15}
    assert traces[2]["model"] == "test-model"
    assert "PRIVATE" not in repr(traces) and "SECRET" not in repr(traces)
    generate("PRIVATE", AnswerSettings())
    assert traces[-1]["parent"] is None


def test_failures_are_sanitized_and_not_swallowed(traces):
    @obs.traced("retrieval")
    def fail(query):
        raise RuntimeError("PRIVATE QUESTION AND KEY")

    with pytest.raises(RuntimeError, match="PRIVATE"):
        fail("PRIVATE")
    assert traces[0]["level"] == "ERROR"
    assert traces[0]["status_message"] == "operation_failed"
    assert "PRIVATE" not in repr(traces)


def test_caught_answer_failure_is_visible(traces):
    @obs.traced("answer")
    def answer(query):
        return {"answer_status": "unavailable", "results": []}

    assert answer("PRIVATE")["answer_status"] == "unavailable"
    assert traces[0]["level"] == "ERROR"


@pytest.mark.parametrize("failure", ["start", "enter", "update", "exit"])
def test_telemetry_failures_do_not_repeat_or_break_work(monkeypatch, failure):
    manager = Mock()
    manager.__enter__ = Mock(return_value=Mock())
    manager.__exit__ = Mock(return_value=False)
    client = Mock()
    client.start_as_current_observation.return_value = manager
    target = {
        "start": client.start_as_current_observation,
        "enter": manager.__enter__, "update": manager.__enter__.return_value.update,
        "exit": manager.__exit__,
    }[failure]
    target.side_effect = RuntimeError("telemetry failure")
    monkeypatch.setattr(obs, "_get_client", lambda: client)
    work = Mock(return_value={"results": []})
    assert obs.traced("retrieval")(work)("PRIVATE") == {"results": []}
    work.assert_called_once_with("PRIVATE")


@pytest.mark.parametrize("enabled,keys", [(False, False), (True, False), (True, True)])
def test_disabled_missing_keys_or_sdk_are_noops(monkeypatch, enabled, keys):
    monkeypatch.setenv("DRUGASSIST_OBSERVABILITY_ENABLED", str(enabled).lower())
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(key, "test" if keys else "")
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert obs._get_client() is None


def test_lazy_initialization_and_shutdown(monkeypatch):
    monkeypatch.setenv("DRUGASSIST_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    factory = Mock()
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=factory))
    assert obs._get_client() is obs._get_client()
    factory.assert_called_once()
    obs.shutdown()
    factory.return_value.shutdown.assert_called_once()
    obs.shutdown()
    factory.return_value.shutdown.assert_called_once()


@pytest.mark.parametrize("outcome", ["generated", "insufficient_evidence", "unavailable"])
def test_real_service_flow(api_service, traces, monkeypatch, outcome):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    payload = {
        "insufficient_evidence": False, "evidence_note": None,
        "statements": [{"section": "answer", "text": "Synthetic response.", "source_ids": ["S1"]}],
    }
    body = {"status": "completed", "model": "synthetic-model", "usage": {"input_tokens": 8},
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(payload)},
            ]}]}
    post = Mock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: body))
    if outcome == "unavailable":
        post.side_effect = RuntimeError("PRIVATE PROVIDER FAILURE")
    monkeypatch.setattr("requests.post", post)
    question = "weather" if outcome == "insufficient_evidence" else "metformin contraindications"
    result = AnswerService(api_service).answer(question)
    assert result["answer_status"] == outcome
    assert [(row["name"], row["parent"]) for row in traces][:2] == [
        ("answer", None), ("retrieval", "answer"),
    ]
    if outcome == "insufficient_evidence":
        post.assert_not_called()
        assert len(traces) == 2
    else:
        assert traces[2]["parent"] == "answer" and traces[2]["name"] == "generation"
    assert "Synthetic response" not in repr(traces) and "PRIVATE" not in repr(traces)


def test_writer_usage_is_per_call(traces, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    bodies = [{"status": "completed", "usage": {"input_tokens": count}, "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "{}"}]},
    ]} for count in (4, 9)]
    monkeypatch.setattr("requests.post", Mock(side_effect=[
        SimpleNamespace(raise_for_status=lambda: None, json=lambda body=body: body) for body in bodies
    ]))
    writer = OpenAIAnswerWriter()
    writer.write("PRIVATE", AnswerSettings(), ["S1"])
    writer.write("PRIVATE", AnswerSettings(), ["S1"])
    assert [row["usage_details"] for row in traces] == [{"input": 4}, {"input": 9}]


def test_installed_sdk_with_in_memory_exporter(monkeypatch):
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(
        public_key="pk-lf-offline-inmemory", secret_key="offline-test-only",
        tracer_provider=TracerProvider(), span_exporter=exporter, sample_rate=1,
    )
    monkeypatch.setattr(obs, "_get_client", lambda: client)
    try:
        with obs._observation("answer", {"requested_k": 3}):
            with obs._observation("generation", {}, "test-model"):
                obs.record_generation({"usage": {"input_tokens": 7}})
        client.flush()
        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == ["generation", "answer"]
        assert spans[0].parent.span_id == spans[1].context.span_id
        assert "7" in repr(spans[0].attributes)
        assert all("input" not in key or "usage" in key for span in spans for key in span.attributes)
    finally:
        client.shutdown()
