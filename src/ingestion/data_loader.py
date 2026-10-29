"""Immutable snapshots and incremental cross-run record identity."""

from pathlib import Path

from src.ingestion.api_client import APIClient, APIError
from src.ingestion.coverage import record_covers_name
from src.preprocessing.normalization import normalize_record
from src.quality.validators import validate_record
from src.utils.config import ROOT, digest, read_json, utcnow, write_json


def ingest(config, client=None, base_snapshot=None, report_dir=None, set_latest=True):
    """Acquire and validate label records, persist a snapshot, and optionally activate it."""
    settings = config["ingestion"]
    names = settings["lookup_names"]
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len({name.strip().casefold() for name in names}) != len(names)
    ):
        raise ValueError("Configure nonempty, unique drug search names")
    client = client or APIClient(timeout=settings["timeout"], retries=settings["retries"])
    errors = []
    morvs_available = True
    try:
        discovery = client.discover()
    except APIError as exc:
        morvs_available = False
        discovery = {"scope": "bounded direct openFDA name lookups; MORVS discovery unavailable"}
        errors.append({"stage": "morvs_discovery", "error": str(exc)})
    write_json(Path(report_dir or ROOT / "reports") / "api_contract.json", discovery)
    responses = list(base_snapshot["manifest"]["responses"]) if base_snapshot else []
    source_records = [r for r in base_snapshot["records"] if r["kind"] != "fda_full"] if base_snapshot else []
    fda_records = [r for r in base_snapshot["records"] if r["kind"] == "fda_full"] if base_snapshot else []
    comparisons = list(base_snapshot["verification"]) if base_snapshot else []
    recent = {"labels": []}
    if morvs_available:
        try:
            recent, meta = client.recent(settings["recent_limit"])
            if not isinstance(recent, dict) or not isinstance(recent.get("labels"), list):
                raise APIError("MORVS recent response schema changed")
            responses.append(meta)
        except APIError as exc:
            morvs_available = False
            errors.append({"stage": "morvs_recent", "error": str(exc)})
            recent = {"labels": []}
    for record in recent["labels"]:
        source_records.append({"record": record, "provenance": meta, "kind": "recent_summary"})
    verified = {r["record"]["set_id"] for r in fda_records}
    for name in names:
        covered = {r["record"]["set_id"] for r in fda_records if record_covers_name(r["record"], name)}
        target = settings["fda_verify_per_name"]
        if len(covered) >= target:
            print(f"Reused {name}: {len(covered)} matching full FDA labels", flush=True)
            continue
        try:
            payload = {"labels": []}
            if morvs_available:
                payload, meta = client.lookup(name)
                if not isinstance(payload, dict) or not isinstance(payload.get("labels"), list):
                    raise APIError("MORVS lookup response schema changed")
                responses.append(meta)
        except APIError as exc:
            errors.append({"stage": "morvs_lookup", "name": name, "error": str(exc)})
            payload = {"labels": []}
        for record in payload["labels"]:
            source_records.append({"record": record, "provenance": meta, "kind": "lookup"})
        selected = [r for r in payload["labels"] if r.get("set_id") not in verified]
        for record in selected:
            if len(covered) >= target:
                break
            set_id = record.get("set_id")
            if not set_id:
                continue
            try:
                official, fda_meta = client.fda_label(set_id)
                results = official.get("results", [])
                if not results:
                    raise APIError("FDA: no matching record")
                fda = results[0]
                if fda.get("set_id") != set_id:
                    raise APIError("FDA returned a different label-set identifier")
                if not record_covers_name(fda, name):
                    errors.append(
                        {
                            "stage": "identity_mismatch",
                            "name": name,
                            "set_id": set_id,
                            "error": "Requested name is absent from FDA identity/ingredient fields",
                        }
                    )
                    continue
                issues = validate_record(fda, "fda_full")
                if any(i["severity"] == "error" for i in issues):
                    raise APIError("FDA record failed schema validation")
                verified.add(set_id)
                covered.add(set_id)
                responses.append(fda_meta)
                fda_records.append({"record": fda, "provenance": fda_meta, "kind": "fda_full"})
                same_version = fda.get("effective_time") == record.get("effective_time")
                for section in (
                    "indications_and_usage",
                    "dosage_and_administration",
                    "warnings",
                    "contraindications",
                    "adverse_reactions",
                    "drug_interactions",
                ):
                    third = record.get(section)
                    authority = "\n\n".join(fda.get(section, []))
                    comparisons.append(
                        {
                            "set_id": set_id,
                            "section": section,
                            "same_effective_time": same_version,
                            "morvs_length": len(third or ""),
                            "fda_length": len(authority),
                            "exact_agreement": bool(third) and third == authority,
                            "prefix_agreement": bool(third) and authority.startswith(third),
                            "observed_truncation": bool(third)
                            and authority.startswith(third)
                            and len(authority) > len(third),
                        }
                    )
            except APIError as exc:
                errors.append({"stage": "fda_verification", "set_id": set_id, "error": str(exc)})
        if len(covered) < target:
            try:
                direct, direct_meta = client.fda_name(name)
                responses.append(direct_meta)
                for fda in direct.get("results", []):
                    if len(covered) >= target:
                        break
                    set_id = fda.get("set_id")
                    if not set_id or set_id in verified or not record_covers_name(fda, name):
                        continue
                    if any(i["severity"] == "error" for i in validate_record(fda, "fda_full")):
                        continue
                    verified.add(set_id)
                    covered.add(set_id)
                    fda_records.append(
                        {
                            "record": fda,
                            "provenance": direct_meta,
                            "kind": "fda_full",
                            "acquisition": {"method": "direct_fda_name_fallback", "query_name": name},
                        }
                    )
            except APIError as exc:
                errors.append({"stage": "fda_name_fallback", "name": name, "error": str(exc)})
        print(
            f"Acquired {name}: {len(payload['labels'])} MORVS labels; "
            f"{len(covered)} matching full FDA labels",
            flush=True,
        )
    # FDA full records remain a distinct, explicitly identified supplementary corpus.
    all_records = source_records + fda_records
    unique = {digest([r["kind"], r["record"]]): r for r in all_records}
    snapshot_id = digest(sorted(unique))[:20]
    folder = ROOT / "data/processed" / snapshot_id
    normalized, invalid = [], []
    for key, envelope in sorted(unique.items()):
        issues = validate_record(envelope["record"], envelope["kind"])
        if any(i["severity"] == "error" for i in issues):
            invalid.append({"record_hash": key, "issues": issues})
            continue
        normalized.append(normalize_record(envelope, snapshot_id))
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": utcnow(),
        "config": config,
        "scope": discovery["scope"],
        "responses": responses,
        "source_envelopes": len(all_records),
        "unique_records": len(unique),
        "duplicate_envelopes": len(all_records) - len(unique),
        "normalized_records": len(normalized),
        "quarantined_records": len(invalid),
        "errors": errors,
        "content_hash": digest(sorted(unique)),
        "record_ids": sorted(d["document_id"] for d in normalized),
    }
    if base_snapshot:
        manifest["parent_snapshot_ids"] = [base_snapshot["manifest"]["snapshot_id"]]
        manifest["scope"] += "; expanded from preserved parent snapshot"
    if not (folder / "manifest.json").exists():
        write_json(folder / "records.json", list(unique.values()))
        write_json(folder / "documents.json", normalized)
        write_json(folder / "quarantine.json", invalid)
        write_json(folder / "verification.json", comparisons)
        write_json(folder / "manifest.json", manifest)
    if set_latest:
        write_json(ROOT / "data/processed/latest.json", {"snapshot_id": snapshot_id})
    return load_snapshot(snapshot_id)


def load_snapshot(snapshot_id=None):
    """Load a named snapshot's manifest and data, defaulting to the latest pointer."""
    snapshot_id = snapshot_id or read_json(ROOT / "data/processed/latest.json")["snapshot_id"]
    folder = ROOT / "data/processed" / snapshot_id
    return {
        "manifest": read_json(folder / "manifest.json"),
        "documents": read_json(folder / "documents.json"),
        "records": read_json(folder / "records.json"),
        "verification": read_json(folder / "verification.json"),
    }
