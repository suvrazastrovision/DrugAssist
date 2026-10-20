"""Conservative literal parsing; no benchmark questions or clinical inference."""

import re
from dataclasses import dataclass
from typing import Literal

from src.ingestion.coverage import record_covers_name
from src.preprocessing.normalization import strings

InformationType = Literal[
    "indications", "contraindications", "dosage", "adverse_reactions", "warnings", "interactions"
]
INFORMATION_PATTERNS = {
    "indications": r"\b(?:indications?|used for|uses|treats?)\b",
    "contraindications": r"\b(?:contraindications?|contraindicated)\b",
    "dosage": r"\b(?:dosage|doses?|dosing)\b",
    "adverse_reactions": r"\b(?:adverse reactions?|side effects?)\b",
    "warnings": r"\b(?:warnings?|precautions?)\b",
    "interactions": r"\b(?:interactions?|interacts?|interact)\b",
}
# Everyday wording identifies a label topic; it never rewrites the search question.
DRUG_REFERENCE = r"(?:__drug__|it|this (?:drug|medicine|medication))"
EVERYDAY_PATTERNS = {
    "indications": (
        rf"\bwhat (?:is|are) {DRUG_REFERENCE} (?:(?:prescribed|given|taken|good) )?for\b"
        rf"|\bwhat (?:does|can) {DRUG_REFERENCE} (?:treat|help with)\b"
    ),
    "dosage": (
        rf"\bhow (?:should {DRUG_REFERENCE} be (?:taken|used|administered)"
        rf"|(?:do|should) (?:i|we) (?:take|use) {DRUG_REFERENCE}"
        rf"|to (?:take|use) {DRUG_REFERENCE}"
        rf"|(?:much|often) (?:(?:should|can) (?:i|we) (?:take|use)(?: of)? )?{DRUG_REFERENCE})\b"
    ),
    "adverse_reactions": (
        r"\b(?:unwanted|undesirable) (?:effects?|reactions?)\b"
        rf"|\bwhat problems (?:can|could|does) {DRUG_REFERENCE} cause\b"
    ),
}
# Literal terminology, independently defined here; never used as database filters.
CONDITIONS = (
    "type 2 diabetes",
    "diabetes",
    "hypertension",
    "asthma",
    "depression",
    "heart failure",
    "renal impairment",
    "pain",
)
SECTION_FILTERS = {
    "indications": ["indications_and_usage"],
    "contraindications": ["contraindications"],
    "dosage": ["dosage_and_administration"],
    "adverse_reactions": ["adverse_reactions"],
    "warnings": ["warnings", "warnings_and_cautions", "boxed_warning"],
    "interactions": ["drug_interactions"],
}


@dataclass(frozen=True)
class ParsedQuery:
    drug: str | None
    condition: str | None
    information_type: InformationType | None


def _literal_matches(text, terms):
    matches = [
        (term, match.span())
        for term in terms
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE)
    ]
    # A longer exact alias takes precedence over a name inside that alias.
    return {
        term
        for term, (start, end) in matches
        if not any(a <= start and end <= b and (a, b) != (start, end) for _, (a, b) in matches)
    }


def _single(values):
    return next(iter(values)) if len(values) == 1 else None


class QueryParser:
    def __init__(self, documents, known_names=()):
        self.drug_documents = {}
        for doc in documents:
            if doc["kind"] != "fda_full":
                continue
            metadata = doc["metadata"]
            aliases = {
                value.strip().casefold()
                for field in ("brand_name", "generic_name", "substance_name")
                for value in strings(metadata.get(field))
                if len(value.strip()) >= 3
            }
            # Short configured names must also occur in FDA identity fields.
            identity = {
                "openfda": metadata,
                **{
                    field: doc["sections"].get(field)
                    for field in ("active_ingredient", "spl_product_data_elements")
                },
            }
            aliases.update(name.casefold() for name in known_names if record_covers_name(identity, name))
            for alias in aliases:
                self.drug_documents.setdefault(alias, set()).add(doc["document_id"])

    def parse(self, query):
        drugs = _literal_matches(query, self.drug_documents)
        intent_text = query
        for alias in sorted(drugs, key=len, reverse=True):
            intent_text = re.sub(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", "__drug__", intent_text, flags=re.IGNORECASE
            )
        intent_text = re.sub(r"\s+", " ", intent_text)
        types = {
            kind for kind, pattern in INFORMATION_PATTERNS.items() if re.search(pattern, query, re.IGNORECASE)
        }
        types.update(
            kind for kind, pattern in EVERYDAY_PATTERNS.items()
            if re.search(pattern, intent_text, re.IGNORECASE)
        )
        return ParsedQuery(
            drug=_single(drugs),
            condition=_single(_literal_matches(query, CONDITIONS)),
            information_type=_single(types),
        )

    def recognizes_drug(self, query):
        return bool(_literal_matches(query, self.drug_documents))

    def filters(self, parsed):
        filters = {}
        if parsed.drug:
            filters["document_id"] = sorted(self.drug_documents[parsed.drug])
        if parsed.information_type:
            filters["section"] = SECTION_FILTERS[parsed.information_type]
        return filters
