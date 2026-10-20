from copy import deepcopy

import pytest

from src.evaluation.section_check import check_results, make_questions
from src.preprocessing.normalization import normalize_record


def test_references_use_identity_and_sections_not_interaction_mentions(record):
    record["openfda"]["generic_name"] = ["TEST COMPOUND AND SECOND COMPOUND"]
    record["indications_and_usage"] = ["Engineering fixture uses section."]
    record["drug_interactions"] = ["UNRELATED COMPOUND"]
    envelope = {"kind": "fda_full", "record": record, "provenance": {"body_sha256": "fixture-body"}}
    snapshot = {"manifest": {"snapshot_id": "fixture"}, "records": [envelope]}
    questions = make_questions(snapshot, ["test compound", "second compound"])
    assert len(questions) == 2
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
