from src.retrieval.base import SearchResult


class SemanticRetriever:
    def __init__(self, store):
        self.store = store

    def search(self, query, k=10, filters=None):
        return [SearchResult(c, score) for c, score in self.store.query(query, k, filters)]
