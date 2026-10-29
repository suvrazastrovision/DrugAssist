"""Derived views preserve section text, qualifiers, units and negation verbatim."""

from src.utils.config import digest

SECTIONS = (
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "boxed_warning",
    "adverse_reactions",
    "drug_interactions",
    "active_ingredient",
    "inactive_ingredient",
    "description",
    "clinical_pharmacology",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
    "how_supplied",
    "information_for_patients",
    "spl_product_data_elements",
)
ATTRIBUTES = (
    "brand_name",
    "generic_name",
    "substance_name",
    "manufacturer_name",
    "product_type",
    "route",
    "product_ndc",
    "package_ndc",
    "rxcui",
    "pharm_class_epc",
    "pharm_class_cs",
    "application_number",
)


def strings(value):
    """Return nonblank strings from a string or list, ignoring other value types."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


def normalize_record(envelope, snapshot_id):
    """Build a normalized document while preserving source text and provenance."""
    record, kind, provenance = envelope["record"], envelope["kind"], envelope["provenance"]
    attributes = record.get("openfda", {}) if kind == "fda_full" else record
    metadata = {key: strings(attributes.get(key)) for key in ATTRIBUTES}
    sections = {key: "\n\n".join(strings(record.get(key))) for key in SECTIONS if strings(record.get(key))}
    if kind == "recent_summary" and record.get("indications_snippet"):
        sections = {"indications_snippet": record["indications_snippet"]}
    set_id = record.get("set_id")
    return {
        "document_id": digest([kind, record])[:32],
        "snapshot_id": snapshot_id,
        "set_id": set_id,
        "label_id": record.get("id"),
        "version": record.get("version"),
        "effective_time": record.get("effective_time"),
        "kind": kind,
        "metadata": metadata,
        "sections": sections,
        "source_record_hash": digest(record),
        "provenance": provenance,
        "authority_url": f"https://api.fda.gov/drug/label.json?search=set_id:%22{set_id}%22",
        "content_status": "authoritative_full" if kind == "fda_full" else "third_party_transformation",
        "truncation_possible": kind != "fda_full",
    }
