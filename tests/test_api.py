"""Exercise the API and shared retrieval against an isolated real Chroma collection."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.indexing.chroma_store import ChromaStore
from src.preprocessing.chunking import chunk_documents
from src.preprocessing.normalization import normalize_record
from src.retrieval.search import SearchSession
from src.services.search_service import SearchService
from src.utils.config import write_json


@pytest.fixture
def api_service(record, fake_embedder, tmp_path, monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model, **kwargs):
            self.pairs = []

        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            return np.arange(1, len(pairs) + 1, dtype=float)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    documents = []
    for i, name in enumerate(("metformin", "warfarin"), 1):
        source = deepcopy(record)
        source["set_id"] = f"{i:08d}-1111-4111-8111-111111111111"
        source["openfda"] = {"generic_name": [name], "substance_name": [name]}
        source["contraindications"] = [f"{name} engineering contraindications fixture only. " * 30]
        documents.append(
            normalize_record(
                {"kind": "fda_full", "record": source, "provenance": {"body_sha256": "fixture-body-hash"}},
                "api-fixture",
            )
        )
    chunking = {"strategy": "recursive", "size": 32, "overlap": 4, "encoding": "cl100k_base"}
    chunks = chunk_documents(documents, **chunking)
    store = ChromaStore(chunks, fake_embedder, path=tmp_path / "chroma")
    store.index()
    fusion = {"fusion": "normalized", "rrf_k": 60, "alpha": 0.25}
    run = {
        "run_id": "offline-api-index",
        "snapshot_id": "api-fixture",
        "provisional": True,
        "default_method": "hybrid_normalized",
        "selection": {
            "chunking": chunking,
            "bm25": {"k1": 2.0, "b": 0.75},
            "hybrid": fusion,
            "reranker": {
                "model": "offline-test-reranker",
                "revision": "fixture-pinned-revision",
                "batch_size": 4,
                "max_length": 128,
            },
            "embedding": {
                "embedding": {"provider": "fake"},
                "corpus_hash": store.corpus_hash,
                "embedding_fingerprint": fake_embedder.fingerprint,
                "collection_name": store.collection.name,
            },
        },
        "configuration": {
            "chroma": store.settings,
            "retrieval": {"candidates": 30},
            "ingestion": {"lookup_names": ["metformin", "warfarin"]},
        },
        "fusion_settings": {"hybrid_normalized": fusion},
    }
    write_json(tmp_path / "index.json", run)
    write_json(tmp_path / "chunks.json", [chunk.to_dict() for chunk in chunks])
    session = SearchSession(
        tmp_path / "index.json",
        method="hybrid_rerank",
        embedder=fake_embedder,
        chroma_path=tmp_path / "chroma",
        query_cache=tmp_path / "questions.sqlite3",
    )
    service = SearchService(session, documents, min_score=0.0, threshold_method="hybrid_rerank")
    yield service
    service.close()


@pytest.fixture
def client(api_service, auth_service, auth_headers):
    with TestClient(create_app(lambda: api_service, lambda: auth_service), headers=auth_headers) as client:
        yield client


def test_successful_source_traceable_response(client, api_service):
    response = client.post("/search", json={"query": "What are the contraindications of metformin?", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert not body["no_match"] and len(body["results"]) == 3
    assert body["method"] == "hybrid_rerank" and body["score_kind"] == "cross_encoder_score"
    assert body["threshold"]["method"] == "hybrid_rerank"
    assert body["results"][0]["score"] > 1  # Raw reranker scores are not constrained to probabilities.
    assert body["parsed"] == {"drug": "metformin", "condition": None, "information_type": "contraindications"}
    assert (
        body["threshold"]["status"] == "experimental"
        and "not correctness probabilities" in body["score_note"]
    )
    assert "not medical advice" in body["disclaimer"]
    assert body["applied_filters"]["section"] == ["contraindications"]
    for row in body["results"]:
        assert row["section"] == "contraindications" and "metformin" in row["drug_name"]
        assert row["label_id"] and row["set_id"] and row["document_id"] and row["chunk_id"]
        assert row["source_record_hash"] and row["source_body_sha256"] and row["text"]
        assert row["source_url"].startswith("https://api.fda.gov/")
        assert row["snapshot_id"] == api_service.session.snapshot_id


@pytest.mark.parametrize("k", [1, 2, 4])
def test_requested_k_is_enforced(client, k):
    body = client.post("/search", json={"query": "metformin contraindications", "k": k}).json()
    assert len(body["results"]) == k
    assert [row["rank"] for row in body["results"]] == list(range(1, k + 1))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"k": 1},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": None},
        {"query": 4},
        {"query": "metformin", "k": 0},
        {"query": "metformin", "k": 21},
        {"query": "metformin", "k": 1.5},
        {"query": "metformin", "k": True},
        {"query": "metformin", "k": "2"},
        {"query": "metformin", "unexpected": "parameter"},
    ],
)
def test_invalid_request_is_422_before_database_initialization(payload, auth_service, auth_headers):
    def unavailable():
        raise RuntimeError("Database should not be opened for invalid requests")

    with TestClient(create_app(unavailable, lambda: auth_service), headers=auth_headers) as client:
        assert client.post("/search", json=payload).status_code == 422


def test_provider_failure_is_503_and_redacted(client, fake_embedder):
    fake_embedder.fail = True
    response = client.post("/search", json={"query": "metformin contraindications"})
    assert response.status_code == 503
    assert "secret" not in response.text and "Traceback" not in response.text


def test_cached_request_still_works_if_provider_goes_down(client, fake_embedder):
    payload = {"query": "metformin contraindications", "k": 1}
    assert client.post("/search", json=payload).status_code == 200
    calls = fake_embedder.calls
    fake_embedder.fail = True
    assert client.post("/search", json=payload).status_code == 200
    assert fake_embedder.calls == calls


@pytest.mark.parametrize(
    "query",
    [
        "What is the weather in Berlin?",
        "metformin stock price",
        "unlistedcompound",
        "contraindications of unlistedcompound",
    ],
)
def test_off_topic_or_unsupported_query_is_no_match(client, fake_embedder, query):
    calls = fake_embedder.calls
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200 and response.json()["no_match"]
    assert response.json()["results"] == []
    assert fake_embedder.calls == calls


def test_below_threshold_is_no_match(client, api_service):
    api_service.min_score = 1e9
    body = client.post("/search", json={"query": "metformin contraindications"}).json()
    assert body["no_match"] and body["no_match_reason"] == "below_experimental_threshold"


def test_empty_filters_are_not_relaxed(client):
    body = client.post("/search", json={"query": "metformin side effects"}).json()
    assert body["no_match"] and body["no_match_reason"] == "no_candidates"
    assert body["applied_filters"]["section"] == ["adverse_reactions"]


def test_health_checks_live_database_and_counts(client, api_service, monkeypatch):
    body = client.get("/health").json()
    assert body == {
        "status": "ready",
        "chromadb": "connected",
        "indexed_documents": 2,
        "indexed_chunks": len(api_service.session.chunks),
    }

    def fail():
        raise RuntimeError("internal-database-secret")

    monkeypatch.setattr(api_service.session.store.collection, "count", fail)
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["chromadb"] == "unavailable"
    assert "secret" not in response.text
    assert client.post("/search", json={"query": "metformin contraindications"}).status_code == 503


def test_missing_database_is_503_without_leaking_paths(auth_service, auth_headers):
    def unavailable():
        raise FileNotFoundError("private/database/path")

    with TestClient(create_app(unavailable, lambda: auth_service), headers=auth_headers) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/search", json={"query": "metformin"})
        assert response.status_code == 503 and "private" not in response.text


def test_search_does_not_modify_label_vectors(client, api_service):
    before = api_service.session.store.embedding_digest()
    client.post("/search", json={"query": "metformin contraindications", "k": 4})
    assert api_service.session.store.embedding_digest() == before


@pytest.mark.parametrize("query,topic,sections", [
    ("What is metformin for?", "indications", ["indications_and_usage"]),
    ("How should metformin be taken?", "dosage", ["dosage_and_administration"]),
    ("What unwanted effects can metformin cause?", "adverse_reactions", ["adverse_reactions"]),
])
def test_everyday_questions_apply_topic_filter_and_reach_retrieval_unchanged(
    client, api_service, monkeypatch, query, topic, sections,
):
    original = api_service.session.search
    received = []

    def capture(question, **kwargs):
        received.append(question)
        return original(question, **kwargs)

    monkeypatch.setattr(api_service.session, "search", capture)
    body = client.post("/search", json={"query": query, "k": 2}).json()
    assert body["query"] == query and received == [query]
    assert body["parsed"]["drug"] == "metformin"
    assert body["parsed"]["information_type"] == topic
    assert body["applied_filters"]["section"] == sections
    assert all(hit["section"] in sections for hit in body["results"])
    if topic == "dosage":
        assert body["results"]


def test_negation_and_quantities_reach_reranker_unchanged(client, api_service):
    query = "How often should I take metformin? Do not omit the 2.5 mg/kg qualifier."
    body = client.post("/search", json={"query": query, "k": 2}).json()
    assert body["parsed"]["information_type"] == "dosage" and body["results"]
    assert all(pair[0] == query for pair in api_service.session.retriever.model.pairs)


def test_threshold_cannot_be_applied_to_another_scoring_method(api_service):
    with pytest.raises(ValueError, match="active scoring method"):
        SearchService(api_service.session, [], min_score=0.5, threshold_method="dense")


def test_reranker_reorders_more_candidates_than_requested(client, api_service):
    query = "metformin contraindications"
    filters = api_service.parser.filters(api_service.parser.parse(query))
    candidates = api_service.session.retriever.retriever.search(query, 30, filters)
    body = client.post("/search", json={"query": query, "k": 3}).json()
    assert len(api_service.session.retriever.model.pairs) > 3
    assert [row["chunk_id"] for row in body["results"]] == [
        row.chunk.chunk_id for row in reversed(candidates[-3:])
    ]
    assert [row["score"] for row in body["results"]] == sorted(
        [row["score"] for row in body["results"]], reverse=True
    )


def test_service_factory_uses_configured_method(api_service, monkeypatch):
    import src.services.search_service as module

    captured = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return api_service.session

    monkeypatch.setattr(module, "SearchSession", session_factory)
    monkeypatch.setattr(
        module, "load_snapshot", lambda snapshot_id: {"documents": list(api_service.documents.values())}
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda: {"api": {"method": "hybrid_rerank", "threshold_method": "hybrid_rerank", "min_score": 0.5}},
    )
    service = module.build_search_service()
    assert captured["method"] == "hybrid_rerank" and service.min_score == 0.5


def test_incompatible_embedder_is_rejected_before_querying(api_service, fake_embedder, tmp_path):
    fake_embedder.fingerprint = "incompatible-space"
    with pytest.raises(ValueError, match="Embedding model differs"):
        SearchSession(tmp_path / "index.json", embedder=fake_embedder, chroma_path=tmp_path / "chroma")


def test_missing_database_file_is_not_created(chunks, fake_embedder, tmp_path):
    folder = tmp_path / "empty-database-folder"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        ChromaStore(chunks, fake_embedder, path=folder, create=False)
    assert list(folder.iterdir()) == []
