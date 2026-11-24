"""Explicit schema-1 checkpoint migration for the commit-batching optimization."""

import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.analyze_fda_bulk import verify_sources, workspace_lock  # noqa: E402
from scripts.download_fda_bulk import read_json, sha256, verify_baseline  # noqa: E402
from src.ingestion.bulk_analysis import ResourceGuard, SCHEMA_VERSION  # noqa: E402
from src.utils.config import canonical, digest, load_config, utcnow, write_json  # noqa: E402


def migrate(workspace, old_bulk_sha, old_cli_sha):
    """Back up counters, prove unchanged inputs, and update only the code fingerprint."""
    workspace = workspace.resolve()
    if workspace.is_relative_to(REPO) or "onedrive" in str(workspace).lower():
        raise ValueError("Use the separate bulk workspace")
    if SCHEMA_VERSION != 1 or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in (old_bulk_sha, old_cli_sha)):
        raise ValueError("Require schema 1 and explicit previous SHA-256 hashes")
    folder = workspace / "analysis"
    guard = ResourceGuard(folder)
    guard.check()
    with workspace_lock(workspace / "download.lock"):
        _, source, _ = verify_sources(workspace)
        config = load_config()
        code = {str(Path(name)): sha256(REPO / name) for name in (
            "src/ingestion/bulk_analysis.py", "scripts/analyze_fda_bulk.py",
            "src/preprocessing/normalization.py", "src/preprocessing/chunking.py",
        )}
        previous_code = code | {
            str(Path("src/ingestion/bulk_analysis.py")): old_bulk_sha,
            str(Path("scripts/analyze_fda_bulk.py")): old_cli_sha,
        }
        common = {"sources": read_json(workspace / "download_state.json"),
                  "manifest": source["manifest_sha256"], "config": config}
        previous = digest(common | {"code": previous_code})
        current = digest(common | {"code": code})
        if not verify_baseline(read_json(folder / "baseline_before.json"))["unchanged"]:
            raise ValueError("Baseline changed; investigate before migrating counters")
        db = sqlite3.connect(folder / "analysis.sqlite3")
        try:
            identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
            if identity != {"schema": 1, "fingerprint": previous}:
                raise ValueError("Previous fingerprint mismatch; sources/configuration/code are incompatible")
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Checkpoint integrity check failed")
            backup_folder = folder / "checkpoint-backups"
            backup_folder.mkdir(exist_ok=True)
            backup = backup_folder / (previous[:16] + ".sqlite3")
            if backup.exists():
                raise ValueError("Refusing to overwrite an existing migration backup")
            required = (folder / "analysis.sqlite3").stat().st_size + 16 * 1024**2
            if guard.usage() + required >= guard.max_bytes:
                raise ValueError("Backup would exceed the analysis storage allowance")
            with sqlite3.connect(backup) as target:
                db.backup(target)
            count_sql = "SELECT COUNT(*), SUM(passages), SUM(embedding_tokens) FROM sized"
            before = db.execute(count_sql).fetchone()
            db.execute("UPDATE meta SET value=? WHERE key='identity'", (
                canonical({"schema": 1, "fingerprint": current}),
            ))
            db.commit()
            if db.execute(count_sql).fetchone() != before:
                raise RuntimeError("Unexpected counter change during metadata migration")
            audit = {"updated_at": utcnow(), "previous_fingerprint": previous,
                     "new_fingerprint": current, "backup": str(backup),
                     "previous_code": previous_code, "current_code": code,
                     "sized_labels_passages_tokens_preserved": before,
                     "change": "Batch sizing commits; bounded 64 MiB SQLite cache"}
            write_json(folder / "checkpoint_migration.json", audit)
            guard.check()
            print(json.dumps(audit, indent=2))
        finally:
            db.close()


def main():
    """Require explicit prior implementation hashes for a deliberate compatibility migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--previous-bulk-sha", required=True)
    parser.add_argument("--previous-cli-sha", required=True)
    args = parser.parse_args()
    migrate(args.workspace, args.previous_bulk_sha, args.previous_cli_sha)


if __name__ == "__main__":
    main()
