"""The shared archive must restore usable data without exporting user secrets/history."""

import json
import sqlite3
from contextlib import closing
from zipfile import ZipFile

import pytest

from scripts import share_data


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path):
    for name in share_data.COMPATIBILITY:
        write(tmp_path, name, {})
    body = write(tmp_path, "data/raw/http/public.body", {"drug": "example"})
    snapshot = "data/processed/snapshot"
    for name in share_data.SNAPSHOT_FILES:
        write(tmp_path, f"{snapshot}/{name}", [])
    write(tmp_path, f"{snapshot}/manifest.json", {"responses": [{
        "raw_file": "data\\raw\\http\\public.body", "body_sha256": share_data.sha256(body),
    }]})
    write(tmp_path, "results/search_latest.json", {"run_path": "results\\search_indexes\\active\\index.json"})
    write(tmp_path, "results/search_indexes/active/index.json", {
        "snapshot_id": "snapshot", "selection": {
            "embedding": {"collection_name": "active"},
            "reranker": {"model": "test/reranker", "revision": "pinned"},
        },
    })
    write(tmp_path, "results/search_indexes/active/chunks.json", [{"chunk_id": "passage"}])
    folder = "data/models/models--test--reranker/snapshots/pinned"
    write(tmp_path, f"{folder}/config.json", {})
    write(tmp_path, f"{folder}/model.safetensors", "public model weights")
    database = tmp_path / "data/chroma/chroma.sqlite3"
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript("""
            CREATE TABLE collections(id TEXT, name TEXT);
            CREATE TABLE segments(id TEXT, collection TEXT, scope TEXT);
            CREATE TABLE embeddings(id INTEGER, segment_id TEXT);
            INSERT INTO collections VALUES ('collection', 'active');
            INSERT INTO segments VALUES ('segment', 'collection', 'METADATA');
            INSERT INTO embeddings VALUES (1, 'segment');
        """)
    write(tmp_path, "data/chroma/vector/data_level0.bin", "public vector index")
    for name in [".env", "data/auth/users.sqlite3", "data/cache/embeddings.sqlite3",
                 "data/processed/old/documents.json", "reports/private.json",
                 "data/models/.locks/locked", "results/search_indexes/old/chunks.json"]:
        write(tmp_path, name, "PRIVATE DO NOT SHARE")
    return tmp_path


def extract(prepared, destination):
    archive = prepared / "drugassist-data.zip"
    share_data.export_bundle(prepared, archive)
    destination.mkdir()
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for info in zipped.infolist():
            assert "\\" not in info.filename
            assert b"PRIVATE DO NOT SHARE" not in zipped.read(info)
        zipped.extractall(destination)
    for name in share_data.COMPATIBILITY:
        write(destination, name, {})
    return destination


def test_public_bundle_restores_without_accounts_or_query_history(prepared, tmp_path):
    destination = extract(prepared, tmp_path / "recipient")
    share_data.check_bundle(destination)
    assert not (destination / "data/auth").exists()
    assert not (destination / "data/cache").exists()
    assert not (destination / ".env").exists()
    assert share_data.read_json(destination / "data/processed/latest.json")["snapshot_id"] == "snapshot"


def test_changed_configuration_is_rejected(prepared, tmp_path):
    destination = extract(prepared, tmp_path / "recipient")
    write(destination, "config/default.json", {"different": True})
    with pytest.raises(ValueError, match="Bundle and repository differ"):
        share_data.check_bundle(destination)


def test_corrupt_immutable_data_is_rejected_even_after_first_check(prepared, tmp_path):
    destination = extract(prepared, tmp_path / "recipient")
    share_data.check_bundle(destination)
    write(destination, "data/processed/snapshot/documents.json", "corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        share_data.check_bundle(destination)


def test_normal_chroma_bookkeeping_changes_do_not_block_later_launches(prepared, tmp_path):
    destination = extract(prepared, tmp_path / "recipient")
    share_data.check_bundle(destination)
    with closing(sqlite3.connect(destination / "data/chroma/chroma.sqlite3")) as db, db:
        db.execute("CREATE TABLE bookkeeping(value TEXT)")
    share_data.check_bundle(destination)
    with closing(sqlite3.connect(destination / "data/chroma/chroma.sqlite3")) as db, db:
        db.execute("DELETE FROM embeddings")
    with pytest.raises(ValueError, match="record count"):
        share_data.check_bundle(destination)


def test_archive_keeps_previous_output_if_sources_change(prepared, monkeypatch):
    archive = prepared / "drugassist-data.zip"
    archive.write_text("previous archive")
    original_write = ZipFile.write

    def change_database(*args, **kwargs):
        original_write(*args, **kwargs)
        with closing(sqlite3.connect(prepared / "data/chroma/chroma.sqlite3")) as db, db:
            db.execute("INSERT INTO embeddings VALUES (2, 'segment')")

    monkeypatch.setattr(ZipFile, "write", change_database)
    with pytest.raises(RuntimeError, match="Chroma changed"):
        share_data.export_bundle(prepared, archive)
    assert archive.read_text() == "previous archive"
    assert not archive.with_suffix(".zip.tmp").exists()


@pytest.mark.parametrize("path", ["../outside.json", "results/search_indexes/../../private/index.json"])
def test_descriptor_cannot_escape_the_public_index_directory(prepared, path):
    write(prepared, "results/search_latest.json", {"run_path": path})
    with pytest.raises(ValueError):
        share_data.active_files(prepared)


def test_modified_raw_source_is_rejected(prepared):
    write(prepared, "data/raw/http/public.body", "modified")
    with pytest.raises(ValueError, match="Raw response checksum"):
        share_data.export_bundle(prepared, prepared / "drugassist-data.zip")


def test_git_line_endings_do_not_break_bundle_compatibility(prepared, tmp_path):
    for name in share_data.COMPATIBILITY:
        (prepared / name).write_bytes(b"{}\n")
    destination = extract(prepared, tmp_path / "recipient")
    for name in share_data.COMPATIBILITY:
        (destination / name).write_bytes(b"{}\r\n")
    share_data.check_bundle(destination)
