"""Full-corpus matching and count-only sizing with isolated offline source fixtures."""

from copy import deepcopy
from io import StringIO
import json
import sqlite3
import zipfile

import pytest

from src.ingestion import bulk_analysis as bulk
from src.preprocessing.chunking import chunk_documents, token_count
from src.preprocessing.normalization import normalize_record
from src.utils.config import canonical, digest

CHUNKING = {"strategy": "recursive", "size": 256, "overlap": 32, "encoding": "cl100k_base"}


def source_zip(path, records, suffix=""):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source.json", json.dumps({"meta": {"unicode": "Ä"}, "results": records}) + suffix)
    return path


def guard(tmp_path):
    return bulk.ResourceGuard(tmp_path, min_free=0)


def silent(*args, **kwargs):
    pass


def approvals(db, tmp_path):
    path = source_zip(tmp_path / "approvals.zip", [
        {"application_number": "ANDA123", "submissions": [{"submission_status": "AP"}]},
        {"application_number": "NDA456", "submissions": [{"submission_status": "TA"}]},
        {"application_number": "BLA789", "submissions": []},
    ])
    bulk.build_approvals(db, path, 3, guard(tmp_path), silent)


def test_streaming_parser_handles_small_blocks_unicode_and_truncation(tmp_path):
    records = [{"warnings": ["Äα🙂" * 500], "set_id": "one"}, {"nested": {"x": [1, 2]}}]
    path = source_zip(tmp_path / "valid.zip", records)
    assert list(bulk.iter_results(path, block_size=7)) == records
    invalid = source_zip(tmp_path / "invalid.zip", records, suffix=" trailing")
    with pytest.raises(ValueError, match="trailing"):
        list(bulk.iter_results(invalid, block_size=11))
    with zipfile.ZipFile(tmp_path / "truncated.zip", "w") as archive:
        archive.writestr("source.json", '{"results":[{"unfinished":')
    with pytest.raises(ValueError):
        list(bulk.iter_results(tmp_path / "truncated.zip", block_size=3))


@pytest.mark.parametrize("value, expected", [
    (" anda-123 ", "ANDA000123"), ("NDA 000123", "NDA000123"),
    ("bla789", "BLA000789"), ("123", None), ("NDA1234567", None), (None, None),
])
def test_application_normalization_preserves_prefix_and_rejects_guessing(value, expected):
    assert bulk.normalize_application(value) == expected


def test_matching_multiple_applications_tentative_and_missing_fields(tmp_path):
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    approvals(db, tmp_path)
    for values, category in [
        (["NDA456", "ANDA000123"], "matched"), (["NDA456"], "tentative_without_approval"),
        (["BLA789"], "application_without_approval"), (["ANDA999"], "unmatched_application"),
        (["OTC monograph"], "invalid_application"), ([], "missing_application"),
    ]:
        actual, matches = bulk.match_applications(db, {"openfda": {"application_number": values}})
        assert actual == category
        assert matches == (["ANDA000123"] if category == "matched" else [])
    assert bulk.match_applications(db, {"openfda": []})[0] == "malformed_record"
    db.close()


def test_newest_matched_version_resume_and_duplicate_accounting(tmp_path, record):
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    approvals(db, tmp_path)
    older = deepcopy(record)
    older["openfda"]["application_number"] = ["ANDA123"]
    newer = deepcopy(older)
    newer["version"] = "10"
    unapproved = deepcopy(newer)
    unapproved["version"] = "11"
    unapproved["openfda"]["application_number"] = ["NDA456"]
    path = source_zip(tmp_path / "labels.zip", [older, newer, older, unapproved])
    with pytest.raises(bulk.Paused):
        bulk.match_partition(db, path, 4, guard(tmp_path), silent, stop_after=2)
    assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2
    bulk.match_partition(db, path, 4, guard(tmp_path), silent)
    assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 4
    assert db.execute("SELECT source_hash, version FROM selected").fetchone() == (digest(newer), 10)
    bulk.match_partition(db, path, 4, guard(tmp_path), silent)
    assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 4
    db.close()


def test_sizing_matches_existing_chunks_tokens_serialization_and_deduplicates(tmp_path, record):
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    approvals(db, tmp_path)
    record["openfda"]["application_number"] = ["ANDA123"]
    second = deepcopy(record)
    second["set_id"] = "another-set"
    path = source_zip(tmp_path / "labels.zip", [record, second])
    bulk.match_partition(db, path, 2, guard(tmp_path), silent)
    original = normalize_record({"record": record, "kind": "fda_full",
                                 "provenance": {"body_sha256": "zip-hash"}}, "bulk-snapshot")
    chunks = chunk_documents([original], **CHUNKING)
    expected_tokens = sum(token_count(chunk.embedding_text) for chunk in chunks)
    with pytest.raises(bulk.Paused):
        bulk.size_partition(db, path, 2, "zip-hash", "bulk-snapshot", CHUNKING,
                            guard(tmp_path), silent, stop_after=1)
    bulk.size_partition(db, path, 2, "zip-hash", "bulk-snapshot", CHUNKING, guard(tmp_path), silent)
    total = db.execute("SELECT SUM(passages), SUM(embedding_tokens) FROM sized").fetchone()
    assert total == (2 * len(chunks), 2 * expected_tokens)
    assert db.execute("SELECT COUNT(*), SUM(tokens) FROM inputs").fetchone() == (len(chunks), expected_tokens)
    actual_size = db.execute("SELECT passage_jsonl_bytes FROM sized WHERE set_id=?", (record["set_id"],))
    assert actual_size.fetchone()[0] == sum(
        len(canonical(c.to_dict()).encode("utf-8")) + 1 for c in chunks
    )
    before = list(db.execute("SELECT * FROM sized ORDER BY set_id"))
    bulk.size_partition(db, path, 2, "zip-hash", "bulk-snapshot", CHUNKING, guard(tmp_path), silent)
    assert list(db.execute("SELECT * FROM sized ORDER BY set_id")) == before
    assert not list(tmp_path.glob("*.jsonl"))
    db.close()


def test_changed_configuration_refuses_resume(tmp_path):
    path = tmp_path / "state.sqlite3"
    bulk.open_database(path, "original").close()
    with pytest.raises(ValueError, match="changed"):
        bulk.open_database(path, "different")


def test_resource_limits_and_source_count_mismatch(tmp_path, monkeypatch):
    limited = bulk.ResourceGuard(tmp_path, max_bytes=10, min_free=0)
    (tmp_path / "large.bin").write_bytes(b"x" * 11)
    with pytest.raises(bulk.Paused, match="storage"):
        limited.check()
    monkeypatch.setattr(bulk, "peak_rss_bytes", lambda: 5 * bulk.GIB)
    with pytest.raises(bulk.Paused, match="RAM"):
        guard(tmp_path).check()
    monkeypatch.setattr(bulk, "peak_rss_bytes", lambda: 0)
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    path = source_zip(tmp_path / "labels.zip", [])
    with pytest.raises(ValueError, match="count mismatch"):
        bulk.match_partition(db, path, 1, guard(tmp_path), silent)
    assert not bulk.completed(db, "label:labels.zip")[1]
    db.close()


def test_empty_and_scalar_streams_have_clear_validation(tmp_path):
    assert list(bulk.iter_results(source_zip(tmp_path / "empty.zip", []), block_size=1)) == []
    assert bulk.JsonStream(StringIO('1234'), block_size=1).value() == 1234
    db = sqlite3.connect(":memory:")
    db.close()


def test_report_separates_exact_counts_from_unmeasured_server_requirements(tmp_path, record):
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    approvals(db, tmp_path)
    record["openfda"]["application_number"] = ["ANDA123"]
    path = source_zip(tmp_path / "labels.zip", [record, record])
    bulk.match_partition(db, path, 2, guard(tmp_path), silent)
    bulk.size_partition(db, path, 2, "zip-hash", "bulk", CHUNKING, guard(tmp_path), silent)
    report = bulk.analysis_report(db, {"compressed_bytes": 123}, {
        "embedding": {"dimensions": 1536}, "chunking": CHUNKING,
    }, "fixture", guard(tmp_path), "complete")
    assert report["matching"]["scanned_records"] == 2
    assert report["matching"]["unique_source_records"] == 1
    assert report["matching"]["selected_records"] == 1
    assert report["matching"]["matched_duplicate_or_superseded_records"] == 1
    size = report["sizing"]
    assert size["raw_float32_vector_bytes"] == size["passages"] * 6144
    assert size["raw_float32_cache_payload_bytes"] == size["unique_embedding_inputs"] * 6144
    assert not size["index_built"] and not size["embeddings_requested"] and not size["passages_saved"]
    assert report["server_decision"]["recommended_ram_gib"] is None
    comparison = bulk.baseline_comparison(db, [{"kind": "fda_full", "record": record}])
    assert comparison["records"] == comparison["approval_matched_records"] == 1
    assert comparison["details"][0]["identical_to_selected_record"]
    db.close()


def test_uncommitted_sizing_batch_rolls_back_all_counters(tmp_path, record):
    db = bulk.open_database(tmp_path / "state.sqlite3", "fixture")
    bulk.size_record(db, record, "source", "snapshot", CHUNKING, guard(tmp_path), commit=False)
    assert db.execute("SELECT COUNT(*) FROM sized").fetchone()[0] == 1
    db.rollback()
    for table in ("sized", "inputs", "sections"):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    db.close()
