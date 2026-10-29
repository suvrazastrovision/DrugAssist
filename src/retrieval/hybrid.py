"""Combine lexical and semantic rankings with explicit fusion weights."""

from src.retrieval.base import SearchResult


def fuse(rankings, method="rrf", rrf_k=60, alpha=0.5):
    """Merge rankings by weighted reciprocal rank or min-max normalized scores."""
    if method not in {"rrf", "normalized"} or rrf_k <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid fusion parameters")
    scores, chunks = {}, {}
    for index, ranking in enumerate(rankings):
        weight = alpha if index == 0 else 1 - alpha
        values = [r.score for r in ranking]
        low, high = (min(values), max(values)) if values else (0, 0)
        seen = set()
        for position, result in enumerate(ranking, 1):
            key = result.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = result.chunk
            contribution = (
                1 / (rrf_k + position)
                if method == "rrf"
                else ((result.score - low) / (high - low) if high > low else 1.0)
            )
            scores[key] = scores.get(key, 0) + weight * contribution
    return [
        SearchResult(chunks[key], score)
        for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class HybridRetriever:
    """A fusion of lexical and semantic candidate rankings."""

    def __init__(self, lexical, semantic, candidates=30, fusion="rrf", rrf_k=60, alpha=0.5):
        """Store the retrievers, candidate count, and ranking fusion settings."""
        self.lexical, self.semantic = lexical, semantic
        self.candidates = candidates
        self.fusion, self.rrf_k, self.alpha = fusion, rrf_k, alpha

    def search(self, query, k=10, filters=None):
        """Retrieve candidates from both methods and return the top fused results."""
        if k < 1:
            raise ValueError("k must be positive")
        count = max(k, self.candidates)
        return fuse(
            [self.lexical.search(query, count, filters), self.semantic.search(query, count, filters)],
            self.fusion,
            self.rrf_k,
            self.alpha,
        )[:k]
