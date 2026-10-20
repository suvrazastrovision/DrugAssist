"""Expansion coverage and acquisition safety, using engineering fixtures only."""

from copy import deepcopy

import pytest

from src.ingestion.coverage import coverage_report, record_covers_name
from src.ingestion.data_loader import ingest
from src.utils.config import load_config, read_json, write_json


def test_default_sample_is_sixty_unique_names_and_retains_original_twelve():
    settings = load_config()["ingestion"]
    names = settings["lookup_names"]
    assert len(names) == len(set(names)) == 60
    assert {
        "aspirin",
        "metformin",
        "warfarin",
        "lisinopril",
        "atorvastatin",
        "amoxicillin",
        "ibuprofen",
        "sertraline",
        "omeprazole",
        "albuterol",
        "levothyroxine",
        "clopidogrel",
    } <= set(names)
    assert settings["fda_verify_per_name"] == 2


def test_coverage_requires_identity_not_interaction_mentions(record):
    record["drug_interactions"] = ["Contains a mention of aspirin for this engineering test."]
    assert not record_covers_name(record, "aspirin")
    record["active_ingredient"] = ["ASPIRIN 100 mg"]
    assert record_covers_name(record, "aspirin")
    assert not record_covers_name(record, "asp")


def test_combination_label_counts_as_one_set_for_two_names(record):
    record["openfda"]["generic_name"] = ["TEST COMPOUND AND SECOND COMPOUND"]
    snapshot = {"manifest": {"snapshot_id": "test"}, "records": [{"kind": "fda_full", "record": record}]}
    result = coverage_report(snapshot, ["test compound", "second compound"])
    assert result["covered_names"] == 2
    assert result["full_fda_label_sets"] == 1


def test_fda_fallback_rejects_wrong_identity_preserves_parent_and_does_not_activate(
    record, tmp_path, monkeypatch
):
    from src.ingestion import data_loader

    monkeypatch.setattr(data_loader, "ROOT", tmp_path)
    write_json(tmp_path / "data/processed/latest.json", {"snapshot_id": "parent"})
    meta = {"body_sha256": "fixture-hash", "retrieved_at": "2024-01-01"}
    parent = {
        "manifest": {"snapshot_id": "parent", "responses": [meta]},
        "records": [{"kind": "fda_full", "record": record, "provenance": meta}],
        "verification": [],
    }
    candidate = deepcopy(record)
    candidate["set_id"] = "22222222-2222-4222-8222-222222222222"
    candidate["openfda"] = {"generic_name": ["NEW COMPOUND"]}
    wrong = deepcopy(candidate)
    wrong["set_id"] = "33333333-3333-4333-8333-333333333333"
    wrong["openfda"] = {"generic_name": ["UNRELATED COMPOUND"]}
    wrong["drug_interactions"] = ["NEW COMPOUND"]

    class Client:
        def __init__(self):
            self.calls = []

        def discover(self):
            return {"scope": "engineering fixture"}

        def recent(self, limit):
            return {"labels": []}, meta

        def lookup(self, name):
            self.calls.append(name)
            return {"labels": []}, meta

        def fda_name(self, name):
            return {"results": [wrong, candidate]}, meta

    client = Client()
    config = load_config()
    config["ingestion"].update(lookup_names=["test compound", "new compound"], fda_verify_per_name=1)
    result = ingest(config, client, parent, set_latest=False)
    assert client.calls == ["new compound"]
    assert coverage_report(result, config["ingestion"]["lookup_names"])["covered_names"] == 2
    assert len(result["records"]) == 2
    assert result["manifest"]["parent_snapshot_ids"] == ["parent"]
    assert read_json(tmp_path / "data/processed/latest.json")["snapshot_id"] == "parent"
    assert any(
        row.get("acquisition", {}).get("method") == "direct_fda_name_fallback" for row in result["records"]
    )


@pytest.mark.parametrize("names", [[""], ["test", "TEST"], ["test", "   "]])
def test_invalid_names_fail_before_network(names):
    config = load_config()
    config["ingestion"]["lookup_names"] = names
    with pytest.raises(ValueError, match="unique"):
        ingest(config)
