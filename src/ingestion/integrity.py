"""Verify frozen lineage against actual raw response bytes, without HTTP calls."""

import hashlib

from src.utils.config import ROOT, digest


def verify_snapshot(snapshot):
    bodies = {}
    for envelope in snapshot["records"]:
        provenance = envelope["provenance"]
        path = ROOT / provenance["raw_file"]
        expected = provenance["body_sha256"]
        if expected not in bodies:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Raw response integrity failure: {path.name}")
            bodies[expected] = True
    record_hashes = {digest(e["record"]) for e in snapshot["records"]}
    if any(d["source_record_hash"] not in record_hashes for d in snapshot["documents"]):
        raise ValueError("Document lineage does not match source records")
    if (
        digest(sorted(digest([e["kind"], e["record"]]) for e in snapshot["records"]))
        != snapshot["manifest"]["content_hash"]
    ):
        raise ValueError("Snapshot record manifest hash mismatch")
    return {
        "verified_raw_bodies": len(bodies),
        "verified_record_hashes": len(record_hashes),
        "documents": len(snapshot["documents"]),
        "snapshot_integrity": "passed",
    }
