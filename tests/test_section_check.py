from copy import deepcopy

import pytest

from src.evaluation.section_check import (
    check_results,
    covers_reference,
    make_hard_questions,
    make_questions,
    summarize,
)
from src.preprocessing.normalization import normalize_record


def test_references_use_identity_and_sections_not_interaction_mentions(record):
    record["openfda"]["generic_name"] = ["TEST COMPOUND AND SECOND COMPOUND"]
    record["indications_and_usage"] = ["Engineering fixture uses section."]
    record["drug_interactions"] = ["UNRELATED COMPOUND"]
    envelope = {"kind": "fda_full", "record": record, "provenance": {"body_sha256": "fixture-body"}}
    snapshot = {"manifest": {"snapshot_id": "fixture"}, "records": [envelope]}
    questions = make_questions(snapshot, ["test compound", "second compound"])
    assert len(questions) == 2
    assert [q["id"] for q in questions] == ["baseline_01", "baseline_02"]
    assert all(q["group"] == "baseline" for q in questions)
    assert questions[0]["topic"] == "Uses" and questions[1]["topic"] == "Dosage"
    doc = normalize_record(envelope, "fixture")
    assert questions[0]["references"][0]["unit_id"] == doc["document_id"] + ":indications_and_usage"
    assert questions[0]["references"][0]["source_record_hash"] == doc["source_record_hash"]
    with pytest.raises(ValueError, match="No source-backed"):
        make_questions(snapshot, ["unrelated compound"])


def test_correct_drug_with_wrong_section_does_not_pass():
    question = {"references": [{"unit_id": "doc:dosage_and_administration"}]}
    wrong = {"document_id": "doc", "section": "warnings"}
    right = {"document_id": "doc", "section": "dosage_and_administration"}
    assert not check_results([wrong] * 10, question)["found_in_10"]
    result = check_results([wrong, right], question)
    assert result == {"found_first": False, "found_in_10": True, "first_matching_rank": 2}
    assert not check_results([wrong] * 10 + [right], question)["found_in_10"]
    other_label = deepcopy(right)
    other_label["document_id"] = "another-doc"
    assert not check_results([other_label], question)["found_in_10"]


def test_hard_references_preserve_exact_source_offsets_and_all_occurrences(record, monkeypatch):
    from src.evaluation import section_check

    quote = "Do not stop without medical supervision."
    record["warnings"] = [quote + " Additional text. " + quote]
    envelope = {"kind": "fda_full", "record": record, "provenance": {"body_sha256": "fixture-body"}}
    snapshot = {"manifest": {"snapshot_id": "fixture"}, "records": [envelope]}
    task = {"drug": "test compound", "topic": "Vocabulary", "text": "Find the stopping rule.",
            "requirements": (("Stopping rule", "warnings", (quote, "Alternative not in this version")),)}
    monkeypatch.setattr(section_check, "HARD_TASKS", [task])
    question = make_hard_questions(snapshot)[0]
    text = normalize_record(envelope, "fixture")["sections"]["warnings"]
    assert question["id"] == "hard_01" and question["group"] == "hard"
    assert len(question["references"]) == 2
    assert all(text[r["start"]:r["end"]] == r["quote"] == quote for r in question["references"])
    task["requirements"] = (("Absent evidence", "warnings", "This claim is not present."),)
    with pytest.raises(ValueError, match="Missing saved evidence"):
        make_hard_questions(snapshot)


@pytest.mark.parametrize("intervals,expected", [
    ([(0, 12)], False), ([(0, 12), (10, 25)], True),
    ([(0, 12), (13, 25)], False), ([(0, 12), (0, 12)], False),
])
def test_evidence_coverage_requires_full_contiguous_text(intervals, expected):
    reference = {"document_id": "doc", "section": "warnings", "start": 5, "end": 20}
    rows = [{"document_id": "doc", "section": "warnings", "start": start, "end": end}
            for start, end in intervals]
    assert covers_reference(rows, reference) is expected
    assert not covers_reference([{**row, "document_id": "wrong"} for row in rows], reference)
    assert not covers_reference([{**row, "section": "wrong"} for row in rows], reference)


def hard_fixture():
    question = {"group": "hard", "requirements": [{"id": "r1"}, {"id": "r2"}], "references": [
        {"requirement_id": "r1", "document_id": "doc", "section": "warnings", "start": 0, "end": 10},
        {"requirement_id": "r2", "document_id": "doc", "section": "interactions", "start": 20, "end": 30},
    ]}
    first = {"document_id": "doc", "section": "warnings", "start": 0, "end": 10}
    second = {"document_id": "doc", "section": "interactions", "start": 20, "end": 30}
    return question, first, second


def test_hard_scoring_distinguishes_a_hit_from_all_required_evidence():
    question, first, second = hard_fixture()
    partial = check_results([first] * 10 + [second], question)
    assert partial["found_in_10"] and partial["evidence_coverage"] == 0.5
    assert partial["requirements_found"] == 1 and not partial["all_evidence_found"]
    complete = check_results([first, second, first], question)
    assert complete["all_evidence_found"] and complete["evidence_coverage"] == 1
    assert complete["requirements_found"] == 2
    assert complete["matched_requirements"] == ["r1", "r2"]


def test_partial_spans_can_complete_a_requirement_at_a_later_rank():
    question, first, _ = hard_fixture()
    scores = check_results([{**first, "end": 6}, {**first, "start": 5}], question)
    assert scores["first_matching_rank"] == 2 and not scores["found_first"]
    assert scores["evidence_coverage"] == 0.5


def test_hard_summary_uses_mean_question_coverage_and_separate_complete_count():
    question, first, second = hard_fixture()
    rows = [dict(check_results([first], question), group="hard", latency_ms=10),
            dict(check_results([first, second], question), group="hard", latency_ms=30)]
    summary = summarize(rows)
    assert summary["questions"] == 2 and summary["complete_in_10"] == 1
    assert summary["evidence_coverage"] == 0.75 and summary["average_ms"] == 20


def test_comparison_saves_groups_and_reuses_only_matching_questions(record, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.evaluation import section_check
    from src.utils.config import write_json

    record["indications_and_usage"] = ["Fixture use."]
    record["warnings"] = ["Do not stop suddenly."]
    envelope = {"kind": "fda_full", "record": record, "provenance": {"body_sha256": "fixture-body"}}
    snapshot = {"manifest": {"snapshot_id": "fixture"}, "records": [envelope]}
    document = normalize_record(envelope, "fixture")
    chunks = [{"document_id": document["document_id"], "set_id": document["set_id"], "section": section,
               "chunk_id": section, "start": 0, "end": len(text), "rank": 1, "score": 1}
              for section, text in document["sections"].items()]
    task = {"drug": "test compound", "topic": "Vocabulary", "text": "Stopping rule?",
            "requirements": (("Stopping", "warnings", "Do not stop suddenly."),)}
    write_json(tmp_path / "results/search_latest.json", {"run_path": "results/run/index.json"})
    write_json(tmp_path / "results/run/index.json", {
        "snapshot_id": "fixture", "run_id": "fixture-run", "selection": {},
        "configuration": {"ingestion": {"lookup_names": ["test compound"]}},
    })
    write_json(tmp_path / "results/run/chunks.json", chunks)
    source = tmp_path / "src/evaluation/section_check.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixture implementation")
    monkeypatch.setattr(section_check, "ROOT", tmp_path)
    monkeypatch.setattr(section_check, "HARD_TASKS", [task])
    monkeypatch.setattr(section_check, "METHODS", ("keyword", "dense"))
    monkeypatch.setattr(section_check, "load_snapshot", lambda _: snapshot)
    monkeypatch.setattr(section_check, "verify_snapshot", lambda _: {"snapshot_integrity": "passed"})
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.embedder = SimpleNamespace(encode=lambda texts: None)
            self.retriever = SimpleNamespace(store=SimpleNamespace(embedder=self.embedder))

        def search(self, text, k):
            calls.append(text)
            return {"results": chunks[:k], "latency_ms": 5}

    monkeypatch.setattr(section_check, "SearchSession", Session)
    report = section_check.compare_methods()
    assert report["corpus_drug_names"] == 1 and report["question_groups"] == {"baseline": 1, "hard": 1}
    assert all(m["questions"] == 2 and m["groups"]["hard"]["complete_in_10"] == 1 for m in report["methods"])
    previous_calls = len(calls)
    section_check.compare_methods()
    assert len(calls) == previous_calls
    task["text"] = "Changed stopping question?"
    assert section_check.compare_methods()["comparison_id"] != report["comparison_id"]
    assert len(calls) > previous_calls
