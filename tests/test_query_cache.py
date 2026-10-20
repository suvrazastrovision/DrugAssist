from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from src.embeddings.cache import QueryEmbeddingCache
from src.embeddings.embedder import EmbeddingUnavailable, OpenAIEmbedder
from src.utils.config import digest


def test_cache_survives_restarts_and_provider_failure(fake_embedder, tmp_path):
    path = tmp_path / "questions.sqlite3"
    cache = QueryEmbeddingCache(fake_embedder, path)
    vector = cache.encode(["metformin", "metformin"])
    assert fake_embedder.calls == 1
    cache.close()
    fake_embedder.fail = True
    reopened = QueryEmbeddingCache(fake_embedder, path)
    assert np.array_equal(reopened.encode(["metformin"])[0], vector[0])
    assert fake_embedder.calls == 1
    with pytest.raises(EmbeddingUnavailable, match="provider is unavailable"):
        reopened.encode(["unseen question"])
    reopened.close()


def test_cache_keys_include_embedding_space(fake_embedder, tmp_path):
    path = tmp_path / "questions.sqlite3"
    first = QueryEmbeddingCache(fake_embedder, path)
    first.encode(["metformin"])
    first.close()
    fake_embedder.fingerprint = "different-embedding-space"
    changed = QueryEmbeddingCache(fake_embedder, path)
    changed.encode(["metformin"])
    assert fake_embedder.calls == 2
    changed.close()


def test_cache_is_safe_across_worker_threads(fake_embedder, tmp_path):
    cache = QueryEmbeddingCache(fake_embedder, tmp_path / "questions.sqlite3")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.encode(["metformin"]), range(8)))
    assert all(np.array_equal(results[0], row) for row in results)
    assert fake_embedder.calls == 1
    cache.close()


def test_invalid_provider_vectors_are_not_cached(fake_embedder, tmp_path, monkeypatch):
    cache = QueryEmbeddingCache(fake_embedder, tmp_path / "questions.sqlite3")
    monkeypatch.setattr(fake_embedder, "encode", lambda texts: [[float("nan")] * 4])
    with pytest.raises(EmbeddingUnavailable):
        cache.encode(["metformin"])
    assert cache.db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
    cache.close()


def test_corrupt_cached_dimensions_are_rejected(fake_embedder, tmp_path):
    cache = QueryEmbeddingCache(fake_embedder, tmp_path / "questions.sqlite3")
    key = digest([fake_embedder.fingerprint, "metformin"])
    cache.db.execute("INSERT INTO vectors VALUES (?,?)", (key, np.ones(8, dtype=np.float32).tobytes()))
    cache.db.commit()
    with pytest.raises(EmbeddingUnavailable, match="Cached embedding vectors are invalid"):
        cache.encode(["metformin"])
    assert fake_embedder.calls == 0
    cache.close()


def test_openai_cache_reuses_document_vectors_for_queries_offline(tmp_path, monkeypatch):
    """Document indexing and later search share persisted vectors, without two caches."""
    path = tmp_path / "embeddings.sqlite3"
    provider = OpenAIEmbedder(dimensions=4, batch_size=1, cache=path)
    provider.key = "engineering-fixture-key"
    requests = []

    def respond(url, **kwargs):
        requests.append(kwargs["json"]["input"])
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {
                "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}],
                "usage": {"total_tokens": 1},
            },
        )

    monkeypatch.setattr(provider.session, "post", respond)
    expected = provider.encode(["metformin", "warfarin", "metformin"])
    assert requests == [["metformin"], ["warfarin"]]
    assert provider.usage_tokens == 2
    provider.close()

    # The offline fixture blocks real HTTP and removes the API key on reopen.
    reopened = OpenAIEmbedder(dimensions=4, cache=path)
    assert np.array_equal(reopened.encode(["metformin"])[0], expected[0])
    assert reopened.usage_tokens == 0
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
        reopened.encode(["unseen question"])
    reopened.close()
