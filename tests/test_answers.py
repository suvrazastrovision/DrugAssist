"""Offline answer-generation checks using synthetic evidence, never clinical validation."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.preprocessing.chunking import token_count
from src.services.answer_service import (
    INSTRUCTIONS,
    AnswerService,
    AnswerSettings,
    AnswerUnavailable,
    OpenAIAnswerWriter,
    section_window,
)


@pytest.fixture
def answer_search(document):
    text = document["sections"]["warnings"]
    hit = {
        "rank": 1, "score": 5.0, "drug_name": "TEST COMPOUND", "label_id": document["label_id"],
        "set_id": document["set_id"], "document_id": document["document_id"], "chunk_id": "test-chunk",
        "snapshot_id": document["snapshot_id"], "section": "warnings", "text": text[:60],
        "start": 0, "end": 60, "effective_time": document["effective_time"],
        "source_url": document["authority_url"], "source_record_hash": document["source_record_hash"],
        "source_body_sha256": document["provenance"]["body_sha256"],
    }
    result = {
        "query": "TEST COMPOUND warnings", "parsed": {"drug": None, "condition": None, "information_type": None},
        "no_match": False, "no_match_reason": None, "method": "hybrid_rerank", "snapshot_id": "test-snapshot",
        "applied_filters": {}, "score_kind": "cross_encoder_score", "score_note": "Engineering fixture.",
        "threshold": {"method": "hybrid_rerank", "min_score": 0.5, "status": "experimental"},
        "results": [hit], "disclaimer": "Engineering fixture, not medical advice.",
    }
    return SimpleNamespace(
        search=Mock(return_value=result), documents={document["document_id"]: document}, close=Mock()
    )


def good_answer(source="S1"):
    return {"insufficient_evidence": False, "evidence_note": None, "statements": [
        {"section": "answer", "text": "Do not administer 2.5 mg/kg. Keep the safety qualifier intact.",
         "source_ids": [source]},
    ]}


def test_answer_expands_exact_saved_context_and_keeps_citations(answer_search):
    writer = Mock()
    writer.write.return_value = good_answer()
    service = AnswerService(answer_search, writer)
    result = service.answer("original question with NOT and 2.5 mg/kg", 3, "clinician")
    assert result["answer_status"] == "generated" and result["insufficient_evidence"] is False
    assert "[S1]" in result["answer"]
    assert result["results"] == answer_search.search.return_value["results"]
    source = result["citations"][0]
    section = answer_search.documents[source["document_id"]]["sections"][source["section"]]
    assert source["text"] == section and source["end"] == len(section)
    assert not source["context_partial"]
    assert len(source["text"]) > len(result["results"][0]["text"])
    answer_search.search.assert_called_once_with("original question with NOT and 2.5 mg/kg", 3)
    payload, settings, ids = writer.write.call_args.args
    assert ids == ["S1"] and json.loads(payload)["audience"] == "clinician"
    assert token_count(INSTRUCTIONS) + token_count(payload) <= settings.max_context_tokens


def test_missing_evidence_skips_the_model(answer_search):
    answer_search.search.return_value.update(no_match=True, results=[])
    writer = Mock()
    result = AnswerService(answer_search, writer).answer("unsupported question")
    assert result["answer_status"] == "insufficient_evidence" and result["answer"] is None
    assert "No matching FDA passages" in result["evidence_note"]
    writer.write.assert_not_called()


def test_writer_can_abstain_when_retrieved_passages_do_not_answer_question(answer_search):
    writer = Mock()
    writer.write.return_value = {
        "insufficient_evidence": True, "statements": [],
        "evidence_note": "The supplied excerpts do not describe this requested detail.",
    }
    result = AnswerService(answer_search, writer).answer("unsupported medical detail")
    assert result["answer_status"] == "insufficient_evidence"
    assert result["results"] and result["citations"] == []
    assert result["evidence_note"] == writer.write.return_value["evidence_note"]


def test_missing_side_effects_section_is_explained_without_relaxing_filters(answer_search):
    document_id = next(iter(answer_search.documents))
    answer_search.search.return_value.update(
        no_match=True, no_match_reason="no_candidates", results=[],
        applied_filters={"document_id": [document_id], "section": ["adverse_reactions"]},
    )
    writer = Mock()
    result = AnswerService(answer_search, writer).answer("test compound side effects")
    assert "no separate side-effects section" in result["evidence_note"]
    assert "does not mean the drug has no side effects" in result["evidence_note"]
    assert "Warnings are available" in result["evidence_note"]
    assert result["results"] == [] and result["applied_filters"]["section"] == ["adverse_reactions"]
    writer.write.assert_not_called()


@pytest.mark.parametrize("output", [
    good_answer("invented-source"),
    {**good_answer(), "statements": []},
    {**good_answer(), "insufficient_evidence": True},
    {**good_answer(), "statements": [{"section": "answer", "text": "uncited claim", "source_ids": []}]},
    {**good_answer(), "statements": [{"section": "answer", "text": "   ", "source_ids": ["S1"]}]},
    {**good_answer(), "insufficient_evidence": "false"},
    {"insufficient_evidence": True, "statements": [], "evidence_note": None},
    {**good_answer(), "evidence_note": "   "},
    {**good_answer(), "statements": [{"section": "details", "text": "No direct answer", "source_ids": ["S1"]}]},
    {**good_answer(), "statements": good_answer()["statements"] * 2},
])
def test_invalid_or_untraceable_output_falls_back_to_evidence(answer_search, output):
    writer = Mock()
    writer.write.return_value = output
    result = AnswerService(answer_search, writer).answer("question")
    assert result["answer_status"] == "unavailable" and result["answer"] is None
    assert result["insufficient_evidence"] is None and result["results"]
    assert not result["statements"] and not result["citations"]


def test_changed_saved_text_prevents_generation(answer_search):
    answer_search.search.return_value["results"][0]["text"] = "Changed evidence"
    writer = Mock()
    result = AnswerService(answer_search, writer).answer("question")
    assert result["answer_status"] == "unavailable"
    writer.write.assert_not_called()


def test_context_deduplicates_sections_but_preserves_label_versions(answer_search):
    hit = answer_search.search.return_value["results"][0]
    doc = deepcopy(answer_search.documents[hit["document_id"]])
    doc["document_id"] = "other-label-version"
    answer_search.documents[doc["document_id"]] = doc
    answer_search.search.return_value["results"] += [
        deepcopy(hit), {**hit, "document_id": doc["document_id"], "chunk_id": "other-chunk"},
    ]
    writer = Mock()
    writer.write.return_value = good_answer()
    AnswerService(answer_search, writer).answer("question")
    evidence = json.loads(writer.write.call_args.args[0])["evidence"]
    assert len(evidence) == 2 and evidence[0]["document_id"] != evidence[1]["document_id"]


def test_distinct_passages_in_one_long_section_are_kept_with_exact_offsets(answer_search):
    first, second = "First engineering detail.", "Second engineering detail."
    section = "Earlier context. " * 2000 + first + "Middle context. " * 2000 + second + "Later context. " * 2000
    hit = answer_search.search.return_value["results"][0]
    answer_search.documents[hit["document_id"]]["sections"][hit["section"]] = section
    hits = [
        {**hit, "chunk_id": f"anchor-{i}", "start": section.index(text),
         "end": section.index(text) + len(text), "text": text}
        for i, text in enumerate((first, second))
    ]
    answer_search.search.return_value["results"] = [hits[0], hits[0], hits[1]]
    service = AnswerService(answer_search, settings={"max_source_tokens": 256, "max_sources": 2})
    sources, payload = service.context(answer_search.search.return_value, "patient")
    assert len(sources) == 2 and sources[0]["document_id"] == sources[1]["document_id"]
    assert [source["source_id"] for source in sources] == ["S1", "S2"]
    for source, anchor in zip(sources, hits, strict=True):
        assert source["text"] == section[source["start"]:source["end"]]
        assert anchor["text"] in source["text"] and source["context_partial"]
        assert token_count(source["text"]) <= service.settings.max_source_tokens
    assert token_count(INSTRUCTIONS) + token_count(payload) <= service.settings.max_context_tokens


def test_overlapping_context_merges_even_at_source_limit(answer_search, monkeypatch):
    hit = answer_search.search.return_value["results"][0]
    section = "Engineering context only. " * 100
    answer_search.documents[hit["document_id"]]["sections"][hit["section"]] = section
    answer_search.search.return_value["results"] = [
        {**hit, "start": start, "end": start + 10, "text": section[start:start + 10]}
        for start in (10, 90)
    ]
    monkeypatch.setattr(
        "src.services.answer_service.section_window",
        lambda text, start, end, limit: (0, 80) if start == 10 else (60, 150),
    )
    service = AnswerService(answer_search, settings={"max_sources": 1})
    sources, _ = service.context(answer_search.search.return_value, "patient")
    assert len(sources) == 1 and sources[0]["source_id"] == "S1"
    assert sources[0]["text"] == section[:150] and sources[0]["end"] == 150


def test_budget_rejects_new_context_without_losing_accepted_evidence(answer_search):
    hit = answer_search.search.return_value["results"][0]
    first = {**hit, "text": "Engineering first section."}
    first.update(start=0, end=len(first["text"]))
    second = {**hit, "section": "dosage_and_administration", "text": "Other engineering section."}
    second.update(start=0, end=len(second["text"]))
    doc = answer_search.documents[hit["document_id"]]
    doc["sections"][first["section"]] = first["text"]
    doc["sections"][second["section"]] = second["text"]
    # Leave room for one source, but not the second source's metadata.
    second["label_id"] = "metadata " * 3000
    search = {**answer_search.search.return_value, "results": [first, second]}
    service = AnswerService(answer_search, settings={"max_context_tokens": 1500})
    sources, payload = service.context(search, "patient")
    assert len(sources) == 1 and sources[0]["text"] == first["text"]
    assert token_count(INSTRUCTIONS) + token_count(payload) <= 1500


def test_answer_groups_supported_details_and_warnings_and_reports_specific_gap(answer_search):
    writer = Mock()
    writer.write.return_value = {
        **good_answer(), "evidence_note": "The supplied excerpts do not report frequency information.",
        "statements": [
            {"section": "warnings", "text": "Keep the safety qualifier intact.", "source_ids": ["S1"]},
            good_answer()["statements"][0],
            {"section": "details", "text": "Engineering supporting detail.", "source_ids": ["S1"]},
        ],
    }
    result = AnswerService(answer_search, writer).answer("question")
    assert result["answer_status"] == "generated"
    assert [row["section"] for row in result["statements"]] == ["answer", "details", "warnings"]
    assert result["answer"].startswith("Answer\n") and "Important warnings\n" in result["answer"]
    assert result["evidence_note"] == writer.write.return_value["evidence_note"]
    assert result["citations"][0]["source_id"] == "S1"


def test_section_window_is_bounded_preserves_hit_and_offsets():
    prefix = "Earlier paragraph with exact units and conditions. " * 200
    hit = "Do NOT administer 2.5 mg/kg. "
    section = prefix + hit + "Later evidence with qualifications. " * 200
    start, end = section_window(section, len(prefix), len(prefix) + len(hit), 300)
    assert start <= len(prefix) and end >= len(prefix) + len(hit)
    assert hit in section[start:end] and token_count(section[start:end]) <= 300
    assert start > 0 and end < len(section)


@pytest.mark.parametrize("body", [
    {"status": "incomplete", "output": []},
    {"status": "failed", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "cut off {"},
    ]}]},
])
def test_incomplete_refused_or_malformed_provider_responses_are_not_displayed(monkeypatch, body):
    monkeypatch.setenv("OPENAI_API_KEY", "engineering-test-key")
    monkeypatch.setattr(requests, "post", Mock(return_value=SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: body,
    )))
    with pytest.raises(AnswerUnavailable):
        OpenAIAnswerWriter().write("fixture", AnswerSettings(), ["S1"])


def test_responses_request_has_exact_source_enum_and_no_stored_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "engineering-test-key")
    post = Mock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        "status": "completed", "model": "provider-reported-snapshot", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(good_answer())},
        ]}],
    }))
    monkeypatch.setattr(requests, "post", post)
    result = OpenAIAnswerWriter().write("fixture question and evidence", AnswerSettings(), ["S1"])
    assert result == {**good_answer(), "_model": "provider-reported-snapshot"}
    body = post.call_args.kwargs["json"]
    assert body["store"] is False and body["max_output_tokens"] == 1500
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["text"]["format"]["strict"] is True
    schema = body["text"]["format"]["schema"]
    assert set(schema["required"]) == {"insufficient_evidence", "statements", "evidence_note"}
    assert "section" in schema["$defs"]["AnswerStatement"]["required"]
    assert body["text"]["format"]["schema"]["$defs"]["AnswerStatement"]["properties"]["source_ids"]["items"]["enum"] == ["S1"]
    assert post.call_args.kwargs["allow_redirects"] is False


def test_authenticated_answer_endpoint_and_redacted_generation_failure(answer_search, auth_service, auth_headers):
    writer = Mock()
    writer.write.side_effect = RuntimeError("private-provider-key-and-question")
    def factory(service):
        return AnswerService(service, writer)
    with TestClient(create_app(lambda: answer_search, lambda: auth_service, factory)) as client:
        assert client.post("/answer", json={"query": "question"}).status_code == 401
        answer_search.search.assert_not_called()
        result = client.post("/answer", json={"query": "question"}, headers=auth_headers)
        assert result.status_code == 200 and result.json()["answer_status"] == "unavailable"
        assert result.json()["results"] and "private-provider" not in result.text
        assert result.headers["Cache-Control"] == "no-store"
        writer.write.side_effect = None
        writer.write.return_value = {**good_answer(), "_model": "provider-reported-snapshot"}
        response = client.post("/answer", json={"query": "question", "audience": "clinician"}, headers=auth_headers)
        assert response.status_code == 200 and response.json()["answer_status"] == "generated"
        assert response.json()["answer_model"] == "provider-reported-snapshot"
        assert response.json()["citations"][0]["source_id"] == "S1"
        for body in [{"query": " "}, {"query": "question", "audience": "doctor"},
                     {"query": "question", "k": 21}, {"query": "question", "k": True}]:
            assert client.post("/answer", json=body, headers=auth_headers).status_code == 422
