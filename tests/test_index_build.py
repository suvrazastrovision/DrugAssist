"""Index rebuilding works offline with explicit settings and no old experiment files."""

import pytest

from src.indexing import build_index
from src.indexing.chroma_store import ChromaStore
from src.retrieval.search import SearchSession
from src.utils.config import load_config, read_json


@pytest.mark.parametrize("complete", [True, False])
def test_build_without_legacy_artifacts(
    record, document, fake_embedder, tmp_path, monkeypatch, complete
):
    config = load_config()
    config["ingestion"]["lookup_names"] = ["test compound"]
    snapshot = {
        "manifest": {"snapshot_id": "test-snapshot"},
        "records": [{"kind": "fda_full", "record": record}],
        "documents": [document],
    }
    monkeypatch.setattr(build_index, "ROOT", tmp_path)
    monkeypatch.setattr(build_index, "verify_snapshot", lambda snapshot: {"verified": True})
    monkeypatch.setattr(
        build_index, "quality_report", lambda snapshot: {"summary": {"acceptance": {"valid": True}}}
    )
    fake_embedder.usage_tokens = 0
    monkeypatch.setattr(build_index, "OpenAIEmbedder", lambda **settings: fake_embedder)

    def temporary_store(chunks, embedder, settings):
        store = ChromaStore(chunks, embedder, settings=settings, path=tmp_path / "chroma")
        if not complete:
            monkeypatch.setattr(store, "index", lambda **kwargs: None)
        return store

    monkeypatch.setattr(build_index, "ChromaStore", temporary_store)
    if not complete:
        with pytest.raises(RuntimeError, match="incomplete"):
            build_index.build_search_index(snapshot, config)
        assert not (tmp_path / "results/search_latest.json").exists()
        assert not (tmp_path / "data/processed/latest.json").exists()
        return

    descriptor = build_index.build_search_index(snapshot, config)
    path = tmp_path / read_json(tmp_path / "results/search_latest.json")["run_path"]
    assert descriptor["selection"]["chunking"] == config["chunking"]
    assert descriptor["selection"]["bm25"] == {"k1": 2.0, "b": 0.75}
    assert descriptor["selection"]["hybrid"] == {
        key: config["retrieval"][key] for key in ("fusion", "rrf_k", "alpha")
    }
    assert descriptor["default_method"] == "hybrid_rerank"
    assert descriptor["fusion_settings"] == config["fusion_settings"]
    assert descriptor["evaluation_status"] == "not_independently_reviewed"
    assert read_json(tmp_path / "data/processed/latest.json")["snapshot_id"] == "test-snapshot"
    assert SearchSession(path, method="bm25").search("safety")["results"]
