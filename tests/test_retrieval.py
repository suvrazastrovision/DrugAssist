from dataclasses import replace

import numpy as np
import pytest

from src.indexing.chroma_store import ChromaStore
from src.retrieval.base import SearchResult, tokenize
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever, fuse
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever


def test_biomedical_tokenization():
    tokens = tokenize("Do NOT use 2.5 mg/kg with β-blockers")
    assert "not" in tokens and "2.5" in tokens and "mg" in tokens and "β-blockers" in tokens


@pytest.mark.parametrize("kind", [KeywordRetriever, BM25Retriever])
def test_lexical_search_and_filters(chunks, kind):
    retriever = kind(chunks)
    rows = retriever.search("safety qualifier", 3, {"section": "warnings"})
    assert rows and all(r.chunk.section == "warnings" for r in rows)
    assert retriever.search("ZZZZUnseenToken", 3) == []
    assert retriever.search("safety", 3, {"set_id": "absent"}) == []
    with pytest.raises(ValueError):
        retriever.search("safety", 0)


def test_bm25_prefers_rare_discriminating_term(chunks):
    a = replace(chunks[0], chunk_id="a", text="alpha alpha alpha", identity="fixture")
    b = replace(chunks[0], chunk_id="b", text="alpha rareterm", identity="fixture")
    result = BM25Retriever([a, b]).search("rareterm", 2)
    assert result[0].chunk.chunk_id == "b"


def test_rrf_hand_computed_and_duplicates(chunks):
    a, b = chunks[:2]
    ranked = fuse([[SearchResult(a, 1000), SearchResult(b, 3)], [SearchResult(b, 0.1)]], rrf_k=10)
    assert ranked[0].chunk == b
    assert ranked[0].score == pytest.approx(0.5 / 12 + 0.5 / 11)
    single = fuse([[SearchResult(a, 10), SearchResult(a, 9)], []])
    assert single[0].score == pytest.approx(0.5 / 61)


def test_normalized_fusion_invariant_to_linear_score_scale(chunks):
    a, b = chunks[:2]
    baseline = fuse(
        [[SearchResult(a, 10), SearchResult(b, 2)], [SearchResult(b, 0.9), SearchResult(a, 0.3)]],
        "normalized",
    )
    rescaled = fuse(
        [[SearchResult(a, 1000), SearchResult(b, 200)], [SearchResult(b, 0.9), SearchResult(a, 0.3)]],
        "normalized",
    )
    assert [(r.chunk.chunk_id, r.score) for r in baseline] == [(r.chunk.chunk_id, r.score) for r in rescaled]


def test_hybrid_filters_apply_to_both_stages(chunks):
    hybrid = HybridRetriever(BM25Retriever(chunks), KeywordRetriever(chunks))
    assert all(r.chunk.section == "warnings" for r in hybrid.search("test", filters={"section": "warnings"}))


class DeterministicTestEmbedder:
    """Engineering-only fixture; never used by scientific pipeline."""

    fingerprint = "test-embedding-fixture-v1"
    dimensions = 3

    def encode(self, texts):
        return np.array([[float("safety" in t.lower()), float("dosage" in t.lower()), 1.0] for t in texts])


def test_real_chroma_persistence_idempotence_and_filters(chunks, tmp_path):
    store = ChromaStore(chunks, DeterministicTestEmbedder(), path=tmp_path)
    store.index()
    store.index()
    assert store.collection.count() == len(chunks)
    rows = SemanticRetriever(store).search("safety", 3, {"section": "warnings"})
    assert rows and all(r.chunk.section == "warnings" for r in rows)
    reopened = ChromaStore(chunks, DeterministicTestEmbedder(), path=tmp_path)
    assert reopened.collection.count() == len(chunks)


def test_chroma_collection_isolates_changed_corpus(chunks, tmp_path):
    left = ChromaStore(chunks, DeterministicTestEmbedder(), path=tmp_path)
    right = ChromaStore(chunks[:1], DeterministicTestEmbedder(), path=tmp_path)
    assert left.fingerprint != right.fingerprint


def test_reranker_uses_one_pinned_revision_and_separate_scores(chunks, monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.retrieval.reranker import CrossEncoderReranker

    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=float)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    lexical = BM25Retriever(chunks)
    reranker = CrossEncoderReranker(lexical, "test-model", "test-immutable-commit", candidates=3)
    rows = reranker.search("safety", 2)
    assert captured["revision"] == "test-immutable-commit"
    assert "revision" not in captured["automodel_args"]
    assert len(rows) == 2 and rows[0].score > rows[1].score


def test_reranker_requests_extra_candidates_even_when_k_exceeds_default(chunks, monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.retrieval.reranker import CrossEncoderReranker

    requested = []

    class Retriever:
        def search(self, query, k, filters):
            requested.append(k)
            return [SearchResult(chunk, 1.0) for chunk in chunks[:k]]

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=float)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    reranker = CrossEncoderReranker(Retriever(), "fixture", "pinned-fixture", candidates=2)
    assert len(reranker.search("fixture", k=4)) == 4
    assert requested == [5]
