from copy import deepcopy

from src.ingestion.incremental import merge_snapshots


def snapshot(envelopes, snapshot_id):
    return {
        "records": envelopes,
        "verification": [],
        "manifest": {"snapshot_id": snapshot_id, "config": {}, "responses": [], "errors": []},
    }


def test_incremental_union_deduplicates_exact_content_and_preserves_versions(record, tmp_path):
    first = {"kind": "fda_full", "record": record, "provenance": {"body_sha256": "fixture-original"}}
    changed = deepcopy(first)
    changed["record"]["version"] = "2"
    changed["record"]["effective_time"] = "20240201"
    base, update = snapshot([first], "original"), snapshot([first, changed], "update")
    merged = merge_snapshots(base, update, processed_dir=tmp_path, set_latest=True)
    assert len(merged["records"]) == 2
    assert {d["version"] for d in merged["documents"]} == {"1", "2"}
    assert merged["manifest"]["duplicate_envelopes"] == 1
    assert record["version"] == "1"
    again = merge_snapshots(base, update, processed_dir=tmp_path)
    assert again == merged
