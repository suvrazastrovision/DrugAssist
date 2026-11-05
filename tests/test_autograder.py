"""Grade synthetic evidence and adversarial outputs without APIs or clinical claims."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from src.evaluation.dataset import EvalCase, EvalDataset, Reference, Requirement, verify_dataset
from src.evaluation.grading import apply_judgment, compare_reports, grade_response
from src.evaluation.judge import JudgeUnavailable, OpenAIJudge, validate_judgment
from src.evaluation.runner import evaluate, main
from src.services.answer_service import AnswerService
from src.utils.config import read_json, write_json

QUOTE = "Do not administer 2.5 mg/kg. Keep the safety qualifier intact."


@pytest.fixture
def evaluation_case(document):
    ref = Reference(
        document_id=document["document_id"], set_id=document["set_id"], section="warnings",
        start=0, end=len(QUOTE), quote=QUOTE, source_record_hash=document["source_record_hash"],
        source_body_sha256=document["provenance"]["body_sha256"],
    )
    return EvalCase(
        id="fixture", question="TEST COMPOUND warnings", split="development", category="warnings",
        expected_outcome="answer",
        requirements=[Requirement(id="r1", description="Preserve the negation, unit and qualifier.",
                                  references=[ref])],
    )


@pytest.fixture
def dataset(document, evaluation_case):
    return EvalDataset(snapshot_id=document["snapshot_id"], notes="Synthetic engineering fixture.",
                       cases=[evaluation_case])


@pytest.fixture
def app_answer(document, evaluation_case):
    hit = {
        "rank": 1, "score": 5.0, "drug_name": "TEST COMPOUND", "label_id": document["label_id"],
        "set_id": document["set_id"], "document_id": document["document_id"], "chunk_id": "fixture-chunk",
        "snapshot_id": document["snapshot_id"], "section": "warnings", "text": QUOTE,
        "start": 0, "end": len(QUOTE), "effective_time": document["effective_time"],
        "source_url": document["authority_url"], "source_record_hash": document["source_record_hash"],
        "source_body_sha256": document["provenance"]["body_sha256"],
    }
    search = {
        "query": evaluation_case.question,
        "parsed": {"drug": None, "condition": None, "information_type": None},
        "no_match": False, "no_match_reason": None, "method": "hybrid_rerank",
        "snapshot_id": document["snapshot_id"], "applied_filters": {}, "score_kind": "cross_encoder_score",
        "score_note": "Engineering fixture.", "threshold": {
            "method": "hybrid_rerank", "min_score": 0.5, "status": "experimental",
        }, "results": [hit], "disclaimer": "Engineering test only.",
    }
    writer = Mock()
    writer.write.return_value = {
        "insufficient_evidence": False, "evidence_note": None,
        "statements": [{"section": "answer", "text": QUOTE, "source_ids": ["S1"]}],
    }
    service = AnswerService(SimpleNamespace(search=Mock(return_value=search),
                                            documents={document["document_id"]: document}), writer)
    return service.answer(evaluation_case.question, 5, evaluation_case.audience)


def judgment():
    return {
        "statements": [{"index": 1, "supported": True, "reason": "Matches the cited fixture.",
                        "evidence": [{"source_id": "S1", "quote": QUOTE}]}],
        "requirements": [{"id": "r1", "satisfied": True, "reason": "The qualifier is intact."}],
        "relevant": True, "appropriate_abstention": True, "critical_errors": [], "reason": "Supported.",
    }


def test_dataset_rejects_changed_source_anchor(document, dataset):
    verify_dataset(dataset, [document])
    bad = dataset.model_copy(deep=True)
    bad.cases[0].requirements[0].references[0].quote = "Administer 2.5 mg/kg."
    with pytest.raises(ValueError, match="Source anchor mismatch"):
        verify_dataset(bad, [document])
    changed = deepcopy(document)
    changed["provenance"]["body_sha256"] = "different-version"
    with pytest.raises(ValueError, match="Source anchor mismatch"):
        verify_dataset(dataset, [changed])
    with pytest.raises(ValueError, match="unique"):
        EvalDataset(**{**dataset.model_dump(), "cases": [dataset.cases[0].model_dump()] * 2})


def test_valid_citations_do_not_prove_semantic_correctness(document, evaluation_case, app_answer):
    grade = grade_response(evaluation_case, app_answer, {document["document_id"]: document},
                           document["snapshot_id"], "answer", 5)
    assert all(item["passed"] for item in grade["checks"])
    assert grade["status"] == "review_required" and grade["retrieval"]["coverage"] == 1
    assert grade["semantic"]["status"] == "not_run"
    wrong = judgment()
    wrong["statements"][0].update(supported=False, evidence=[], reason="Negation was reversed.")
    wrong["critical_errors"] = ["Reversed warning."]
    apply_judgment(grade, {"result": wrong, "model": "fake-judge", "cache_hit": False})
    assert grade["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("text", "Administer 2.5 mg/kg."), ("start", 1), ("source_record_hash", "another-version"),
    ("effective_time", "20990101"), ("source_url", "https://unrelated.invalid"),
])
def test_tampered_citation_blocks_judging(document, dataset, app_answer, field, value):
    app_answer["citations"][0][field] = value
    judge = SimpleNamespace(configuration={"model": "fake"}, grade=Mock())
    report = evaluate(dataset, [document], lambda _: {"response": app_answer}, mode="answer", judge=judge)
    assert report["summary"]["failed"] == 1
    assert report["cases"][0]["grade"]["semantic"]["status"] == "blocked_invalid_evidence"
    judge.grade.assert_not_called()


def test_displayed_answer_is_also_checked(document, evaluation_case, app_answer):
    app_answer["answer"] = "Invented additional uncited advice."
    grade = grade_response(evaluation_case, app_answer, {document["document_id"]: document},
                           document["snapshot_id"], "answer", 5)
    assert grade["status"] == "failed"
    assert not next(item["passed"] for item in grade["checks"] if item["name"] == "rendered_answer")


def test_missing_evidence_fails_and_duplicates_do_not_help(document, evaluation_case, app_answer):
    hit = app_answer["results"][0]
    hit.update(end=10, text=document["sections"]["warnings"][:10])
    app_answer["results"] = [{**hit, "rank": i} for i in range(1, 6)]
    grade = grade_response(evaluation_case, app_answer, {document["document_id"]: document},
                           document["snapshot_id"], "answer", 5)
    assert grade["retrieval"]["coverage"] == 0 and grade["status"] == "failed"
    assert grade["cited_evidence"]["coverage"] == 1  # Expanded context and retrieved spans differ.


def test_explicit_abstention_is_distinct_from_generation_failure(document, evaluation_case, app_answer):
    negative = evaluation_case.model_copy(update={"expected_outcome": "abstain", "requirements": []})
    app_answer.update(answer=None, statements=[], citations=[], insufficient_evidence=True,
                      answer_status="insufficient_evidence", evidence_note="Not in the supplied evidence.")
    grade = grade_response(negative, app_answer, {document["document_id"]: document},
                           document["snapshot_id"], "answer", 5)
    assert grade["status"] == "passed" and grade["retrieval"]["coverage"] is None
    positive = grade_response(evaluation_case, app_answer, {document["document_id"]: document},
                              document["snapshot_id"], "answer", 5)
    assert positive["status"] == "failed"  # Refusing every question is not success.
    app_answer.update(answer_status="unavailable", insufficient_evidence=None)
    failed = grade_response(negative, app_answer, {document["document_id"]: document},
                            document["snapshot_id"], "answer", 5)
    assert failed["status"] == "error"


@pytest.mark.parametrize("mutation", [
    "missing_statement", "duplicate_requirement", "fake_quote", "wrong_source",
])
def test_judge_must_grade_every_item_and_quote_actual_citations(evaluation_case, app_answer, mutation):
    result = judgment()
    if mutation == "missing_statement":
        result["statements"] = []
    elif mutation == "duplicate_requirement":
        result["requirements"] *= 2
    elif mutation == "fake_quote":
        result["statements"][0]["evidence"][0]["quote"] = "Fabricated quote."
    else:
        result["statements"][0]["evidence"][0]["source_id"] = "S2"
    with pytest.raises(ValueError):
        validate_judgment(result, evaluation_case, app_answer)


def test_judge_cache_is_keyed_by_answer_model_and_rubric(evaluation_case, app_answer, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "engineering-test-secret")
    body = {"status": "completed", "model": "pinned-test-judge", "usage": {"input_tokens": 100},
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(judgment())},
            ]}]}
    post = Mock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: body))
    monkeypatch.setattr("src.evaluation.judge.requests.post", post)
    judge = OpenAIJudge({"model": "test-judge"}, tmp_path / "cache")
    first = judge.grade(evaluation_case, app_answer)
    second = judge.grade(evaluation_case, app_answer)
    assert first["model"] == "pinned-test-judge" and not first["cache_hit"] and second["cache_hit"]
    assert post.call_count == 1
    request = post.call_args.kwargs["json"]
    assert request["store"] is False and request["text"]["format"]["strict"] is True
    assert not post.call_args.kwargs["allow_redirects"]
    cached_text = "".join(path.read_text() for path in (tmp_path / "cache").glob("*"))
    assert "engineering-test-secret" not in cached_text
    changed = evaluation_case.model_copy(update={"question": "Another fixture question"})
    judge.grade(changed, app_answer)
    OpenAIJudge({"model": "different-judge"}, tmp_path / "cache").grade(evaluation_case, app_answer)
    monkeypatch.setattr("src.evaluation.judge.JUDGE_INSTRUCTIONS", "A changed grading rubric.")
    judge.grade(evaluation_case, app_answer)
    assert post.call_count == 4


@pytest.mark.parametrize("body", [
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "not JSON"},
    ]}]},
])
def test_invalid_judge_output_is_not_a_grade(body, evaluation_case, app_answer, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.evaluation.judge.requests.post", Mock(return_value=SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: body,
    )))
    with pytest.raises(JudgeUnavailable):
        OpenAIJudge({"model": "test-judge"}, tmp_path / "cache").grade(evaluation_case, app_answer)
    assert not (tmp_path / "cache").exists()


def test_judge_errors_are_redacted_and_separate_from_bad_answers(document, dataset, app_answer):
    judge = SimpleNamespace(configuration={"model": "fake"},
                            grade=Mock(side_effect=requests.RequestException("secret-provider-body")))
    report = evaluate(dataset, [document], lambda _: {"response": app_answer}, mode="answer", judge=judge)
    assert report["summary"]["error"] == 1 and report["summary"]["failed"] == 0
    assert report["summary"]["semantic_graded"] == 0
    assert "secret-provider-body" not in json.dumps(report)


def test_capture_errors_are_saved_and_progress_survives(document, dataset, app_answer, tmp_path):
    second = dataset.cases[0].model_copy(update={"id": "second"})
    dataset = dataset.model_copy(update={"cases": [dataset.cases[0], second]})
    capture = Mock(side_effect=[{"response": app_answer, "elapsed_ms": 10}, RuntimeError("private-key")])
    path = tmp_path / "run.json"
    report = evaluate(dataset, [document], capture, mode="answer", output=path)
    assert report["status"] == "complete" and report["summary"]["review_required"] == 1
    assert report["summary"]["error"] == 1 and read_json(path) == report
    assert "review_required" in path.with_suffix(".md").read_text(encoding="utf-8")
    assert "private-key" not in path.read_text()


def test_compare_rejects_confounded_runs_and_detects_regressions(document, dataset, app_answer):
    baseline = evaluate(dataset, [document], lambda _: {"response": app_answer})
    changed = deepcopy(baseline)
    changed["cases"][0]["grade"].update(status="failed", retrieval={"coverage": 0})
    assert compare_reports(baseline, changed)["regressions"] == ["fixture"]
    for field, value in (("k", 10), ("dataset_hash", "different"), ("grading_hash", "different")):
        changed = deepcopy(baseline)
        changed[field] = value
        with pytest.raises(ValueError, match=field):
            compare_reports(baseline, changed)
    changed = deepcopy(baseline)
    changed["cases"] *= 2
    with pytest.raises(ValueError, match="duplicated"):
        compare_reports(baseline, changed)


def test_live_collection_requires_explicit_opt_in(monkeypatch):
    capture = Mock(side_effect=AssertionError("The app must not be initialized"))
    monkeypatch.setattr("src.evaluation.runner._capture_service", capture)
    assert main(["run"]) == 2
    capture.assert_not_called()


def test_comparison_keeps_judge_errors_and_model_changes_distinct(document, dataset, app_answer):
    judge = SimpleNamespace(configuration={"model": "fake"}, grade=Mock(return_value={
        "result": judgment(), "model": "pinned-judge", "cache_hit": False,
    }))
    baseline = evaluate(dataset, [document], lambda _: {"response": app_answer}, mode="answer", judge=judge)
    failed = deepcopy(baseline)
    failed["cases"][0]["grade"].update(status="error", semantic={"status": "error"})
    comparison = compare_reports(baseline, failed)
    assert comparison["current"]["error"] == 1 and comparison["regressions"] == ["fixture"]
    changed_model = deepcopy(baseline)
    changed_model["cases"][0]["grade"]["semantic"]["model"] = "other-model"
    with pytest.raises(ValueError, match="resolved judge models"):
        compare_reports(baseline, changed_model)


def test_semantic_regression_is_visible_even_when_case_already_fails(document, dataset, app_answer):
    judge = SimpleNamespace(configuration={"model": "fake"}, grade=Mock(return_value={
        "result": judgment(), "model": "pinned-judge", "cache_hit": False,
    }))
    baseline = evaluate(dataset, [document], lambda _: {"response": app_answer}, mode="answer", judge=judge)
    baseline["cases"][0]["grade"]["status"] = "failed"
    changed = deepcopy(baseline)
    changed["cases"][0]["grade"]["semantic"]["result"]["statements"][0]["supported"] = False
    assert compare_reports(baseline, changed)["regressions"] == ["fixture"]


def test_reports_cannot_overwrite_their_own_json_with_markdown(tmp_path):
    from src.evaluation.runner import _new_path

    with pytest.raises(ValueError, match=".json extension"):
        _new_path(tmp_path / "report.md")


def test_cli_regrades_saved_responses_without_models(document, dataset, app_answer, tmp_path, monkeypatch):
    original = evaluate(dataset, [document], lambda _: {"response": app_answer}, mode="answer")
    write_json(tmp_path / "dataset.json", dataset.model_dump())
    write_json(tmp_path / "saved.json", original)
    monkeypatch.setattr("src.evaluation.runner.load_sources", lambda _: {"documents": [document]})
    output = tmp_path / "regraded.json"
    args = ["grade", "--dataset", str(tmp_path / "dataset.json"),
            "--responses", str(tmp_path / "saved.json"), "--output", str(output)]
    assert main(args) == 2  # Generated text still requires a semantic assessment.
    assert read_json(output)["summary"]["review_required"] == 1
    assert main(args) == 2  # Existing outputs must not be overwritten.


def test_evaluate_cli_dispatch(monkeypatch):
    from src import cli

    callback = Mock(return_value=2)
    monkeypatch.setattr("src.evaluation.runner.main", callback)
    monkeypatch.setattr("sys.argv", ["fda-rag", "evaluate", "grade", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    callback.assert_called_once_with(["grade", "--help"])
