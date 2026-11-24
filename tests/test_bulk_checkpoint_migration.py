"""The explicit metadata migration preserves counters and rejects changed inputs."""

import json
from pathlib import Path

import pytest

from scripts import migrate_bulk_checkpoint as migration
from src.ingestion.bulk_analysis import open_database
from src.utils.config import digest, write_json


def migration_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    folder = workspace / "analysis"
    folder.mkdir(parents=True)
    code_names = ("src/ingestion/bulk_analysis.py", "scripts/analyze_fda_bulk.py",
                  "src/preprocessing/normalization.py", "src/preprocessing/chunking.py")
    for name in code_names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + name)
    old_bulk, old_cli = "a" * 64, "b" * 64
    code = {str(Path(name)): migration.sha256(repo / name) for name in code_names}
    previous_code = code | {str(Path(code_names[0])): old_bulk, str(Path(code_names[1])): old_cli}
    source, config = {"manifest_sha256": "manifest"}, {"chunking": "fixture"}
    previous = digest({"sources": {}, "manifest": "manifest", "config": config, "code": previous_code})
    write_json(workspace / "download_state.json", {})
    write_json(folder / "baseline_before.json", {})
    db = open_database(folder / "analysis.sqlite3", previous)
    db.execute("INSERT INTO sized VALUES ('set', 42, 1234, 1000, 500, 400, 300, 200, 0)")
    db.commit()
    db.close()
    monkeypatch.setattr(migration, "REPO", repo)
    monkeypatch.setattr(migration, "verify_sources", lambda path: (None, source, None))
    monkeypatch.setattr(migration, "load_config", lambda: config)
    monkeypatch.setattr(migration, "verify_baseline", lambda before: {"unchanged": True})
    return workspace, folder, previous, old_bulk, old_cli


def test_migration_backs_up_and_changes_only_identity(tmp_path, monkeypatch):
    workspace, folder, previous, old_bulk, old_cli = migration_fixture(tmp_path, monkeypatch)
    migration.migrate(workspace, old_bulk, old_cli)
    audit = json.loads((folder / "checkpoint_migration.json").read_text())
    assert audit["previous_fingerprint"] == previous
    assert audit["sized_labels_passages_tokens_preserved"] == [1, 42, 1234]
    db = open_database(folder / "analysis.sqlite3", audit["new_fingerprint"])
    assert db.execute("SELECT passages FROM sized").fetchone()[0] == 42
    db.close()
    backup = open_database(next((folder / "checkpoint-backups").glob("*.sqlite3")), previous)
    assert backup.execute("SELECT passages FROM sized").fetchone()[0] == 42
    backup.close()


def test_migration_refuses_changed_normalizer(tmp_path, monkeypatch):
    workspace, folder, previous, old_bulk, old_cli = migration_fixture(tmp_path, monkeypatch)
    (migration.REPO / "src/preprocessing/normalization.py").write_text("different semantics")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        migration.migrate(workspace, old_bulk, old_cli)
    db = open_database(folder / "analysis.sqlite3", previous)
    assert db.execute("SELECT passages FROM sized").fetchone()[0] == 42
    db.close()
