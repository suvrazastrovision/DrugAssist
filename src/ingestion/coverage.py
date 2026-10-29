"""Measure named-drug coverage using source identity fields, not retrieval predictions."""

import re

from src.preprocessing.normalization import strings


def record_covers_name(record, name):
    """Literal identity evidence only; a mention in interactions is insufficient.

    This establishes named sample coverage, not ingredient equivalence or a
    clinical/ontology assertion. Combination products are reported explicitly.
    """
    attributes = record.get("openfda", {})
    identity = [
        value
        for key in ("brand_name", "generic_name", "substance_name")
        for value in strings(attributes.get(key))
    ]
    identity += strings(record.get("active_ingredient"))
    identity += strings(record.get("spl_product_data_elements"))
    pattern = r"(?<!\w)" + re.escape(name.strip()) + r"(?!\w)"
    return any(re.search(pattern, value, re.IGNORECASE) for value in identity)


def coverage_report(snapshot, names):
    """Summarize named-drug coverage using identity evidence from full FDA records."""
    records = [envelope["record"] for envelope in snapshot["records"] if envelope["kind"] == "fda_full"]
    rows = []
    for name in names:
        selected = [record for record in records if record_covers_name(record, name)]
        rows.append(
            {
                "name": name,
                "full_fda_label_sets": len({record["set_id"] for record in selected}),
                "set_ids": sorted({record["set_id"] for record in selected}),
                "source_generic_names": sorted(
                    {
                        value
                        for record in selected
                        for value in strings(record.get("openfda", {}).get("generic_name"))
                    }
                ),
            }
        )
    return {
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "requested_names": len(names),
        "covered_names": sum(row["full_fda_label_sets"] > 0 for row in rows),
        "names_with_two_labels": sum(row["full_fda_label_sets"] >= 2 for row in rows),
        "full_fda_records": len(records),
        "full_fda_label_sets": len({r["set_id"] for r in records}),
        "names": rows,
        "selection": "Manually selected named-drug sample; not a representative FDA census.",
        "coverage_definition": (
            "Name occurs in source identity/ingredient/product fields; combinations may "
            "cover multiple names."
        ),
    }
