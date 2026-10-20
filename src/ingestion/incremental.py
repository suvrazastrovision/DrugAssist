"""Append bounded acquisitions to a new immutable snapshot, preserving versions."""

from pathlib import Path

from src.preprocessing.normalization import normalize_record
from src.quality.validators import validate_record
from src.utils.config import ROOT, digest, read_json, utcnow, write_json


def merge_snapshots(base, update, processed_dir=None, set_latest=False):
    """Deduplicate exact source-kind/record content; never overwrite an earlier version.

    This is append-only sample maintenance, not undocumented MORVS pagination.
    Existing versions remain in the new snapshot and must be handled explicitly
    when defining a future retrieval corpus.
    """
    processed = Path(processed_dir or ROOT / "data/processed")
    envelopes = base["records"] + update["records"]
    unique = {digest([envelope["kind"], envelope["record"]]): envelope for envelope in envelopes}
    snapshot_id = digest(sorted(unique))[:20]
    folder = processed / snapshot_id
    if not (folder / "manifest.json").exists():
        documents, quarantine = [], []
        for record_hash, envelope in sorted(unique.items()):
            issues = validate_record(envelope["record"], envelope["kind"])
            if any(issue["severity"] == "error" for issue in issues):
                quarantine.append({"record_hash": record_hash, "issues": issues})
            else:
                documents.append(normalize_record(envelope, snapshot_id))
        comparisons = {digest(row): row for row in base["verification"] + update["verification"]}
        manifest = {
            "snapshot_id": snapshot_id,
            "created_at": utcnow(),
            "config": update["manifest"]["config"],
            "parent_snapshot_ids": [base["manifest"]["snapshot_id"], update["manifest"]["snapshot_id"]],
            "scope": "append-only union of bounded samples; historical label versions retained",
            "responses": base["manifest"]["responses"] + update["manifest"]["responses"],
            "source_envelopes": len(envelopes),
            "unique_records": len(unique),
            "duplicate_envelopes": len(envelopes) - len(unique),
            "normalized_records": len(documents),
            "quarantined_records": len(quarantine),
            "errors": base["manifest"]["errors"] + update["manifest"]["errors"],
            "content_hash": digest(sorted(unique)),
            "record_ids": sorted(d["document_id"] for d in documents),
        }
        write_json(folder / "records.json", list(unique.values()))
        write_json(folder / "documents.json", documents)
        write_json(folder / "quarantine.json", quarantine)
        write_json(folder / "verification.json", list(comparisons.values()))
        write_json(folder / "manifest.json", manifest)
    if set_latest:
        write_json(processed / "latest.json", {"snapshot_id": snapshot_id})
    return {
        "manifest": read_json(folder / "manifest.json"),
        "records": read_json(folder / "records.json"),
        "documents": read_json(folder / "documents.json"),
        "verification": read_json(folder / "verification.json"),
    }
