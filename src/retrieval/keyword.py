"""Rank passages by weighted coverage of unique query terms."""

from src.retrieval.base import ranked, tokenize


class KeywordRetriever:
    """Weighted unique term coverage: identity terms weight 3, section/body weight 1."""

    def __init__(self, chunks):
        """Precompute unique identity and body terms for each source chunk."""
        self.chunks = chunks
        self.identity = [set(tokenize(c.identity)) for c in chunks]
        self.body = [set(tokenize(c.section.replace("_", " ") + " " + c.text)) for c in chunks]

    def search(self, query, k=10, filters=None):
        """Rank filtered chunks by weighted coverage of the query's terms."""
        terms = set(tokenize(query))
        scores = [3 * len(terms & a) + len(terms & b) for a, b in zip(self.identity, self.body, strict=True)]
        return ranked(self.chunks, scores, k, filters)
