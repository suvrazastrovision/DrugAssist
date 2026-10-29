"""Shared ranking contract and biomedical-safe lexical tokenization."""

import re
from dataclasses import dataclass
from typing import Protocol

from src.preprocessing.chunking import Chunk


def tokenize(text):
    """Split text into case-folded terms while preserving numbers and negation."""
    # Preserve numbers/decimals; do not stem drug names or discard negation.
    return re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+(?:[-'][^\W\d_]+)*", text.casefold(), re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    """A source chunk paired with its retrieval method's relevance score."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """The common query and metadata-filter contract for passage retrievers."""

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[SearchResult]:
        """Return up to k ranked passages satisfying the supplied metadata filters."""
        ...


def matches(chunk, filters):
    """Return whether a chunk satisfies every scalar or list-valued filter."""
    return all(
        getattr(chunk, field, None) in value
        if isinstance(value, (list, tuple))
        else getattr(chunk, field, None) == value
        for field, value in (filters or {}).items()
    )


def ranked(chunks, scores, k, filters=None):
    """Return up to k positive-scoring, filtered chunks in deterministic order."""
    if k < 1:
        raise ValueError("k must be positive")
    rows = [
        SearchResult(c, float(s))
        for c, s in zip(chunks, scores, strict=True)
        if s > 0 and matches(c, filters)
    ]
    return sorted(rows, key=lambda r: (-r.score, r.chunk.chunk_id))[:k]
