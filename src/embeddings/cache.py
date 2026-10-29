"""One fingerprinted disk cache for document and query embeddings."""

import sqlite3
from pathlib import Path
from threading import RLock

import numpy as np

from src.utils.config import digest


class EmbeddingUnavailable(RuntimeError):
    """Embedding provider or cached vectors are unavailable."""


class EmbeddingCache:
    """A thread-locked SQLite cache keyed by model fingerprint and input text."""

    def __init__(self, fingerprint, dimensions, path):
        """Open the vector cache for a specific embedding fingerprint and dimension."""
        self.fingerprint, self.dimensions = fingerprint, dimensions
        self.cache_path = Path(path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)")
        self.lock = RLock()

    def encode(self, texts, encode_missing):
        """Return vectors in input order, validating and caching provider misses.

        Call encode_missing once with the unique uncached texts. Both cached and
        fresh vectors must be finite and have the configured dimension; otherwise
        raise EmbeddingUnavailable without returning partial results.
        """
        with self.lock:
            keys = [digest([self.fingerprint, text]) for text in texts]
            vectors, missing = {}, {}
            for key, text in zip(keys, texts, strict=True):
                row = self.db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    vector = np.frombuffer(row[0], dtype=np.float32).copy()
                    if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
                        raise EmbeddingUnavailable("Cached embedding vectors are invalid")
                    vectors[key] = vector
                else:
                    missing[key] = text
            if missing:
                fresh = np.asarray(encode_missing(list(missing.values())), dtype=np.float32)
                if fresh.shape != (len(missing), self.dimensions) or not np.isfinite(fresh).all():
                    raise EmbeddingUnavailable("Provider embedding vectors are invalid")
                for key, vector in zip(missing, fresh, strict=True):
                    vectors[key] = vector
                    self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, vector.tobytes()))
                self.db.commit()
            result = np.asarray([vectors[key] for key in keys], dtype=np.float32).reshape(-1, self.dimensions)
            if not np.isfinite(result).all():
                raise EmbeddingUnavailable("Cached embedding vectors are invalid")
            return result

    def close(self):
        """Close the SQLite cache connection."""
        self.db.close()


class QueryEmbeddingCache(EmbeddingCache):
    """Use the same cache for providers without their own disk cache."""

    def __init__(self, embedder, path):
        """Wrap an embedder with a disk cache matching its fingerprint and dimension."""
        self.embedder = embedder
        super().__init__(embedder.fingerprint, embedder.dimensions, path)

    def encode(self, texts):
        """Return cached query vectors and request missing vectors from the provider."""
        return super().encode(texts, self._encode_missing)

    def _encode_missing(self, texts):
        try:
            return self.embedder.encode(texts)
        except Exception:  # noqa: BLE001 -- Redact provider-specific errors.
            raise EmbeddingUnavailable("Embedding provider is unavailable") from None

    def close(self):
        """Close the cache and release the wrapped provider when supported."""
        super().close()
        close = getattr(self.embedder, "close", None)
        if close is not None:
            close()
