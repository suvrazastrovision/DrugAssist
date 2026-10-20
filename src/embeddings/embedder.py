"""Cached, fingerprinted embeddings; no credentials written to artifacts."""

import os
import time

import numpy as np
import requests
from dotenv import load_dotenv

from src.embeddings.cache import EmbeddingCache, EmbeddingUnavailable
from src.preprocessing.chunking import token_count
from src.utils.config import ROOT, digest


class OpenAIEmbedder(EmbeddingCache):
    def __init__(self, model="text-embedding-3-small", dimensions=1536, batch_size=32, cache=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENAI_API_KEY")
        self.model, self.dimensions, self.batch_size = model, dimensions, batch_size
        fingerprint = digest({"provider": "openai", "model": model, "dimensions": dimensions})
        super().__init__(fingerprint, dimensions, cache or ROOT / "data/cache/embeddings.sqlite3")
        self.usage_tokens = 0
        self.session = requests.Session()

    def encode(self, texts):
        return super().encode(texts, self._encode_missing)

    def _encode_missing(self, texts):
        if any(not text.strip() or token_count(text) > 8191 for text in texts):
            raise ValueError("Embedding input must contain 1..8191 cl100k_base tokens")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            if not self.key:
                raise RuntimeError("OPENAI_API_KEY is not configured; no synthetic fallback is used")
            batch = texts[start : start + self.batch_size]
            for attempt in range(5):
                try:
                    r = self.session.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": "Bearer " + self.key},
                        timeout=(10, 90),
                        json={
                            "model": self.model,
                            "dimensions": self.dimensions,
                            "input": batch,
                            "encoding_format": "float",
                        },
                    )
                except requests.RequestException:
                    if attempt == 4:
                        raise RuntimeError("OpenAI embeddings network failure") from None
                    time.sleep(2**attempt)
                    continue
                if (r.status_code == 429 or r.status_code >= 500) and attempt < 4:
                    delay = r.headers.get("Retry-After", "")
                    time.sleep(min(float(delay) if delay.isdigit() else 2**attempt, 60))
                    continue
                if not r.ok:
                    if r.status_code in (403, 404):
                        raise EmbeddingUnavailable(f"{self.model}: provider returned HTTP {r.status_code}")
                    raise RuntimeError(
                        f"OpenAI embeddings HTTP {r.status_code}; credentials and body redacted"
                    )
                payload = r.json()
                values = sorted(payload["data"], key=lambda v: v["index"])
                if len(values) != len(batch):
                    raise RuntimeError("Embedding response count mismatch")
                self.usage_tokens += payload.get("usage", {}).get("total_tokens", 0)
                for item in values:
                    vector = np.asarray(item["embedding"], dtype=np.float32)
                    if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
                        raise RuntimeError("Invalid embedding dimensions or values")
                    vectors.append(vector)
                break
        return np.asarray(vectors, dtype=np.float32).reshape(-1, self.dimensions)

    def close(self):
        super().close()
        self.session.close()


class LocalEmbedder:
    """Optional local candidate; callers must pin an immutable HF revision."""

    def __init__(self, model, revision, max_length=256, batch_size=32, cache=None):
        from sentence_transformers import SentenceTransformer

        if revision == "main" or not revision:
            raise ValueError("Pin an immutable model commit for local embeddings")
        self.model = SentenceTransformer(
            model, revision=revision, cache_folder=str(cache or ROOT / "data/models"), device="cpu"
        )
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.fingerprint = digest([model, revision, max_length, "normalized", "cpu"])
        self.dimensions = self.model.get_sentence_embedding_dimension()
        self.usage_tokens = 0
        self._cache = {}

    def encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = self.model.encode(
                missing,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            self._cache.update(zip(missing, vectors, strict=True))
        return np.asarray([self._cache[text] for text in texts], dtype=np.float32).reshape(
            -1, self.dimensions
        )
