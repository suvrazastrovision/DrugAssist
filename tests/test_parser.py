from copy import deepcopy

import pytest

from src.query_parser import QueryParser


@pytest.fixture
def parser(document):
    metformin = deepcopy(document)
    metformin["metadata"]["generic_name"] = ["METFORMIN HYDROCHLORIDE"]
    metformin["metadata"]["brand_name"] = ["Fixture brand"]
    metformin["metadata"]["substance_name"] = []
    return QueryParser([metformin], ["metformin"])


def test_literal_drug_and_contraindication(parser):
    parsed = parser.parse("What are the contraindications of metformin?")
    assert parsed.drug == "metformin" and parsed.condition is None
    assert parsed.information_type == "contraindications"
    assert parser.filters(parsed)["section"] == ["contraindications"]


def test_unknown_drug_and_missing_metadata_do_not_create_filters(document, parser):
    assert parser.parse("dose of unlistedcompound").drug is None
    assert "document_id" not in parser.filters(parser.parse("dose of unlistedcompound"))
    document["metadata"] = {}
    document["sections"] = {"warnings": "An incidental mention of metformin."}
    empty = QueryParser([document], ["metformin"])
    assert empty.parse("metformin").drug is None


def test_condition_is_literal_and_never_a_filter(parser):
    parsed = parser.parse("metformin for type 2 diabetes")
    assert parsed.condition == "type 2 diabetes"
    assert "condition" not in parser.filters(parsed)
    assert parser.parse("metformin for an unspecified disease").condition is None


def test_ambiguous_attributes_are_null(parser):
    assert parser.parse("metformin dosage and contraindications").information_type is None
    assert parser.parse("metformin for asthma and depression").condition is None


def test_longest_alias_and_word_boundaries(parser):
    assert parser.parse("METFORMIN HYDROCHLORIDE warnings").drug == "metformin hydrochloride"
    assert parser.parse("metforminfoobar warnings").drug is None
    assert parser.parse("Fixture brand side effects").information_type == "adverse_reactions"


def test_multiple_drugs_do_not_silently_filter_to_one(document):
    other = deepcopy(document)
    other["document_id"] = "other"
    other["metadata"] = {"generic_name": ["Second compound"]}
    parsed = QueryParser([document, other]).parse("test compound and second compound interactions")
    assert parsed.drug is None and parsed.information_type == "interactions"


@pytest.mark.parametrize(
    "query,canonical",
    [
        ("What is metformin for?", "metformin indications"),
        ("What is METFORMIN HYDROCHLORIDE prescribed for?", "METFORMIN HYDROCHLORIDE indications"),
        ("What can Fixture brand help with?", "Fixture brand indications"),
        ("How should metformin be taken?", "metformin dosage"),
        ("How do I take Fixture brand?", "Fixture brand dosage"),
        ("How much metformin should I take?", "metformin dosage"),
        ("How often should I take metformin?", "metformin dosage"),
        ("metformin: how should this medicine be used?", "metformin dosage"),
        ("How to take metformin?", "metformin dosage"),
        ("What unwanted effects can metformin cause?", "metformin side effects"),
        ("metformin: what undesirable reactions can it cause?", "metformin side effects"),
        ("What problems could metformin cause?", "metformin side effects"),
    ],
)
def test_everyday_and_formal_wording_select_same_label_scope(parser, query, canonical):
    natural, formal = parser.parse(query), parser.parse(canonical)
    assert natural.drug == formal.drug
    assert natural.information_type == formal.information_type
    assert parser.filters(natural) == parser.filters(formal)


@pytest.mark.parametrize("query", [
    "What is the price of metformin for a month?",
    "How much does metformin cost?",
    "metformin dosage and unwanted effects",
    "How should metformin be taken and what are its interactions?",
])
def test_everyday_patterns_do_not_force_unrelated_or_multiple_topics(parser, query):
    assert parser.parse(query).information_type is None


def test_everyday_wording_does_not_invent_a_drug_or_correct_a_typo(parser):
    for query in ["What is unlistedcompound for?", "How should metforminn be taken?"]:
        assert parser.parse(query).drug is None
        assert not parser.recognizes_drug(query)
        assert "document_id" not in parser.filters(parser.parse(query))
