"""Persistent, collection-isolated cosine HNSW storage with explicit lineage."""

import hashlib
import os
from pathlib import Path

import numpy as np

from src.utils.config import ROOT, digest


class ChromaStore:
    def __init__(self, chunks, embedder, path=None, settings=None, create=True):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb
        from chromadb.config import Settings

        self.chunks = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.settings = settings or {
            "space": "cosine",
            "ef_construction": 100,
            "ef_search": 100,
            "max_neighbors": 16,
        }
        self.corpus_hash = digest(sorted(self.chunks))
        self.fingerprint = digest([self.corpus_hash, embedder.fingerprint, self.settings])
        storage_path = Path(path or ROOT / "data/chroma")
        if not create and not (storage_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Saved vector index is missing; restore it or run the benchmark first")
        self.client = chromadb.PersistentClient(
            str(storage_path), settings=Settings(anonymized_telemetry=False)
        )
        name = "fda_" + self.fingerprint[:32]
        if create:
            self.collection = self.client.get_or_create_collection(
                name,
                embedding_function=None,
                configuration={"hnsw": self.settings},
                metadata={"corpus_hash": self.corpus_hash, "embedding_fingerprint": embedder.fingerprint},
            )
        else:
            try:
                self.collection = self.client.get_collection(name, embedding_function=None)
            except chromadb.errors.NotFoundError:
                raise FileNotFoundError(
                    "Saved vector collection is missing; restore it or run the benchmark first"
                ) from None

    def index(self, progress=None):
        existing = set(self.collection.get(include=[])["ids"])
        pending = [c for key, c in self.chunks.items() if key not in existing]
        for offset in range(0, len(pending), 64):
            batch = pending[offset : offset + 64]
            vectors = self.embedder.encode([c.embedding_text for c in batch])
            metadata = [
                {k: v for k, v in c.to_dict().items() if k not in ("text", "identity")} for c in batch
            ]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors.tolist(),
                documents=[c.embedding_text for c in batch],
                metadatas=metadata,
            )
            if progress:
                progress(offset + len(batch), len(pending))
        if self.collection.count() != len(self.chunks):
            raise RuntimeError("Index/corpus cardinality mismatch")

    def query(self, query, k, filters=None):
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks:
            return []
        where = None
        if filters:
            clauses = [
                {key: {"$in": value} if isinstance(value, (list, tuple)) else value}
                for key, value in filters.items()
            ]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        result = self.collection.query(
            query_embeddings=self.embedder.encode([query]).tolist(),
            n_results=min(k, len(self.chunks)),
            where=where,
            include=["distances"],
        )
        return [
            (self.chunks[key], 1 - float(distance))
            for key, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def embedding_digest(self):
        """Hash actual stored vectors, not just provider/model configuration."""
        rows = self.collection.get(include=["embeddings"])
        hasher = hashlib.sha256()
        for key, vector in sorted(zip(rows["ids"], rows["embeddings"], strict=True), key=lambda row: row[0]):
            hasher.update(key.encode("utf-8"))
            hasher.update(np.asarray(vector, dtype="<f4").tobytes())
        return hasher.hexdigest()
