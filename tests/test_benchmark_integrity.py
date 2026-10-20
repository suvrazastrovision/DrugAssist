"""Verify the real curated benchmark where the local frozen corpus is present."""

from copy import deepcopy

import pytest

from src.evaluation.benchmark_data import validate_benchmark
from src.ingestion.data_loader import load_snapshot
from src.utils.config import ROOT, read_json


@pytest.fixture
def benchmark_corpus():
    if not (ROOT / "data/processed/latest.json").exists():
        pytest.skip("Real corpus not acquired; use ingestion first for source-integration checks")
    benchmark = read_json(ROOT / "data/benchmarks/benchmark.json")
    return benchmark, [
        d for d in load_snapshot(benchmark["snapshot_id"])["documents"] if d["kind"] == "fda_full"
    ]


def test_80_queries_offsets_and_partitions(benchmark_corpus):
    benchmark, corpus = benchmark_corpus
    assert validate_benchmark(benchmark, corpus, require_review=False)
    assert sum(not q["answerable"] for q in benchmark["queries"]) == 8


def test_draft_is_never_silently_accepted_as_gold(benchmark_corpus):
    benchmark, corpus = benchmark_corpus
    with pytest.raises(ValueError, match="human relevance review"):
        validate_benchmark(benchmark, corpus)


def test_altered_evidence_is_rejected(benchmark_corpus):
    benchmark, corpus = benchmark_corpus
    bad = deepcopy(benchmark)
    bad["queries"][0]["evidence"][0]["text"] = "modified text"
    with pytest.raises(ValueError, match="span no longer matches"):
        validate_benchmark(bad, corpus, require_review=False)


def test_unknown_source_unit_is_rejected(benchmark_corpus):
    benchmark, corpus = benchmark_corpus
    bad = deepcopy(benchmark)
    bad["queries"][0]["qrels"]["missing:section"] = 2
    with pytest.raises(ValueError, match="absent source"):
        validate_benchmark(bad, corpus, require_review=False)


def test_partial_and_irrelevant_judgments_are_supported(benchmark_corpus):
    benchmark, corpus = benchmark_corpus
    updated = deepcopy(benchmark)
    first = updated["queries"][0]
    key = next(iter(first["qrels"]))
    first["qrels"][key] = 1
    negative = next(q for q in updated["queries"] if not q["answerable"])
    negative["qrels"][key] = 0
    assert validate_benchmark(updated, corpus, require_review=False)
