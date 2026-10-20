"""Observable quality indicators; agreement does not establish clinical accuracy."""

import re
from collections import Counter
from datetime import UTC, datetime
from uuid import UUID

from src.preprocessing.normalization import ATTRIBUTES, SECTIONS, strings
from src.utils.config import digest


def validate_record(record, kind="lookup"):
    issues = []
    if not isinstance(record, dict):
        return [{"field": "$", "severity": "error", "issue": "record must be an object"}]

    def add(field, issue, severity="error"):
        issues.append({"field": field, "severity": severity, "issue": issue})

    try:
        UUID(record.get("set_id", ""))
    except (ValueError, TypeError, AttributeError):
        add("set_id", "missing or invalid UUID")
    date = record.get("effective_time")
    if date:
        try:
            parsed = datetime.strptime(str(date), "%Y%m%d").date()
            if parsed > datetime.now(UTC).date():
                add("effective_time", "future effective date", "warning")
        except ValueError:
            add("effective_time", "invalid YYYYMMDD")
    else:
        add("effective_time", "missing label date", "warning")
    if kind == "fda_full" and not isinstance(record.get("openfda", {}), dict):
        add("openfda", "must be an object")
    for key in SECTIONS:
        value = record.get(key)
        if value is None:
            continue
        if kind == "fda_full" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            add(key, "FDA section must be a list of strings")
        elif kind != "fda_full" and not isinstance(value, (str, list)):
            add(key, "section must contain text")
        for text in strings(value):
            if "\ufffd" in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
                add(key, "encoding or control-character anomaly", "warning")
            if kind == "lookup" and len(text) in (200, 1000, 2000, 4000):
                add(key, "possible fixed-width truncation", "warning")
    attrs = record.get("openfda", {}) if kind == "fda_full" else record
    if isinstance(attrs, dict):
        for key in ATTRIBUTES:
            value = attrs.get(key)
            if value is not None and not (
                isinstance(value, str) or isinstance(value, list) and all(isinstance(v, str) for v in value)
            ):
                add(key, "attribute must be string or string list")
        for ndc in strings(attrs.get("product_ndc")):
            if not re.fullmatch(r"\d{4,5}-\d{3,4}", ndc):
                add("product_ndc", "unexpected NDC format", "warning")
    return issues


def quality_report(snapshot):
    records, documents = snapshot["records"], snapshot["documents"]
    field_rows, violations = [], []
    for kind in sorted({r["kind"] for r in records}):
        group = [r for r in records if r["kind"] == kind]
        for field in (*ATTRIBUTES, *SECTIONS, "effective_time", "set_id"):
            values = [
                (
                    r["record"].get("openfda", {})
                    if kind == "fda_full" and field in ATTRIBUTES
                    else r["record"]
                ).get(field)
                for r in group
            ]
            present = sum(bool(strings(v)) for v in values)
            valid = sum(
                isinstance(v, str) or isinstance(v, list) and all(isinstance(x, str) for x in v)
                for v in values
                if v is not None
            )
            denominator = sum(v is not None for v in values)
            field_rows.append(
                {
                    "kind": kind,
                    "field": field,
                    "records": len(group),
                    "present": present,
                    "coverage_pct": 100 * present / len(group),
                    "valid_type_pct_nonnull": 100 * valid / denominator if denominator else None,
                }
            )
        for r in group:
            violations.extend(
                {"kind": kind, "record_hash": digest(r["record"]), **i}
                for i in validate_record(r["record"], kind)
            )
    labels = Counter((d["kind"], d["set_id"], d["effective_time"]) for d in documents)
    by_kind = Counter(d["kind"] for d in documents)
    text_counts = Counter(digest(t) for d in documents for t in d["sections"].values())
    summary = {
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "records": len(records),
        "by_kind": dict(by_kind),
        "distinct_label_sets": len({d["set_id"] for d in documents}),
        "repeated_versions_within_source": sum(c - 1 for c in labels.values()),
        "duplicate_envelope_rate": snapshot["manifest"]["duplicate_envelopes"]
        / max(1, snapshot["manifest"]["source_envelopes"]),
        "repeated_section_texts": sum(c - 1 for c in text_counts.values()),
        "schema_errors": sum(i["severity"] == "error" for i in violations),
        "warnings": sum(i["severity"] == "warning" for i in violations),
        "quarantined": snapshot["manifest"]["quarantined_records"],
        "observed_prefix_truncations": sum(
            v["observed_truncation"] and v["same_effective_time"] for v in snapshot["verification"]
        ),
        "acceptance": {
            "raw_preserved": all(d["provenance"].get("body_sha256") for d in documents),
            "no_unquarantined_schema_errors": all(
                not any(i["severity"] == "error" for i in validate_record(r["record"], r["kind"]))
                for r in records
                if digest(r["record"]) in {d["source_record_hash"] for d in documents}
            ),
            "authoritative_corpus_available": by_kind["fda_full"] > 0,
        },
    }
    return {
        "summary": summary,
        "fields": field_rows,
        "violations": violations,
        "limitations": [
            "Coverage is a bounded purposive sample, not a population estimate.",
            "Missing harmonized fields do not imply absent regulatory information.",
            "openFDA presence does not establish FDA approval or clinical validity.",
            "Source agreement is observed agreement, not independent clinical accuracy.",
        ],
    }
