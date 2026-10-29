"""Adapt vector-store similarity results to the shared retrieval contract."""

from src.retrieval.base import SearchResult


class SemanticRetriever:
    """A retriever backed by vector-store similarity search."""

    def __init__(self, store):
        """Store the vector index used to retrieve matching source chunks."""
        self.store = store

    def search(self, query, k=10, filters=None):
        """Return up to k vector matches with their similarity scores."""
        return [SearchResult(c, score) for c, score in self.store.query(query, k, filters)]
