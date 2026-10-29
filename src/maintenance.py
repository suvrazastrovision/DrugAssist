"""Acquire, index and check the configured drug collection."""

from src.indexing.build_index import build_search_index
from src.ingestion.coverage import coverage_report
from src.ingestion.data_loader import ingest, load_snapshot
from src.ingestion.integrity import verify_snapshot
from src.quality.validators import quality_report
from src.retrieval.search import SearchSession
from src.utils.config import ROOT, read_json, write_json


def available_snapshot(snapshot_id=None):
    """Load a local snapshot, returning None when its referenced files are absent."""
    try:
        return load_snapshot(snapshot_id)
    except FileNotFoundError:
        return None


def acquire(config):
    """Expand local source data and persist coverage, quality, and integrity checks."""
    checkpoint = ROOT / "reports/data_checks.json"
    previous = read_json(checkpoint) if checkpoint.exists() else {}
    baseline = available_snapshot(previous.get("snapshot_id"))
    if baseline is None:
        baseline = available_snapshot()
    snapshot = ingest(
        config, base_snapshot=baseline, report_dir=checkpoint.parent, set_latest=False
    )
    coverage = coverage_report(snapshot, config["ingestion"]["lookup_names"])
    checks = {
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "coverage": coverage,
        "quality": quality_report(snapshot),
        "integrity": verify_snapshot(snapshot),
    }
    if previous.get("snapshot_id") == checks["snapshot_id"] and "name_search_checks" in previous:
        checks["name_search_checks"] = previous["name_search_checks"]
    write_json(checkpoint, checks)
    print("Saved dataset:", checks["snapshot_id"])
    print("Covered drug names:", coverage["covered_names"], "of", coverage["requested_names"])
    print("Full FDA label sets:", coverage["full_fda_label_sets"])
    if coverage["covered_names"] != coverage["requested_names"]:
        raise RuntimeError("Drug coverage is incomplete; see reports/data_checks.json")


def index(config):
    """Build and activate an index from the most recently acquired local snapshot."""
    checkpoint = ROOT / "reports/data_checks.json"
    snapshot_id = read_json(checkpoint)["snapshot_id"] if checkpoint.exists() else None
    snapshot = available_snapshot(snapshot_id) or available_snapshot()
    if snapshot is None:
        raise FileNotFoundError("No local source data. Run fda-rag setup --from-source first.")
    descriptor = build_search_index(snapshot, config)
    print("Activated searchable corpus:", descriptor["coverage"]["covered_names"], "drug names")


def verify(config):
    """Offline name/source checks; these do not measure medical answer quality."""
    session = SearchSession(method="bm25")
    coverage = session.run["coverage"]
    expected = len(config["ingestion"]["lookup_names"])
    if coverage["covered_names"] != expected or coverage["requested_names"] != expected:
        raise ValueError("Active index does not cover the configured drug names")
    snapshot = load_snapshot(session.snapshot_id)
    source_sets = {
        row["record"]["set_id"] for row in snapshot["records"] if row["kind"] == "fda_full"
    }
    if source_sets != {chunk.set_id for chunk in session.chunks}:
        raise ValueError("Active index label sets do not match its saved source dataset")
    rows = []
    for drug in coverage["names"]:
        response = session.search(drug["name"], k=10)
        matching = sorted({row["set_id"] for row in response["results"]} & set(drug["set_ids"]))
        rows.append(
            {
                "name": drug["name"],
                "identity_matched_label_in_top_10": bool(matching),
                "matching_set_ids": matching,
            }
        )
    if not all(row["identity_matched_label_in_top_10"] for row in rows):
        raise ValueError("Some drug names did not retrieve a matching source label")
    checkpoint = ROOT / "reports/data_checks.json"
    checks = read_json(checkpoint) if checkpoint.exists() else {"snapshot_id": session.snapshot_id}
    if checks["snapshot_id"] != session.snapshot_id:
        raise ValueError("Data checks and active index differ; index the acquired dataset first")
    checks["name_search_checks"] = {
        "snapshot_id": session.snapshot_id,
        "method": "bm25",
        "checked_names": len(rows),
        "passed_names": len(rows),
        "indexed_fda_label_sets": len(source_sets),
        "limitation": (
            "Exact-name lookup and source identity smoke checks, not an independently "
            "judged relevance benchmark."
        ),
        "names": rows,
    }
    write_json(checkpoint, checks)
    print(
        f"All {len(rows)} drug names retrieved a matching source; "
        f"all {len(source_sets)} source label sets indexed."
    )


def maintain(config, step="verify"):
    """Run a selected maintenance step or acquire, index, and verify in sequence."""
    actions = {"acquire": acquire, "index": index, "verify": verify}
    for name in actions if step == "all" else [step]:
        actions[name](config)
