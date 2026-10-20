"""Shared ranking contract and biomedical-safe lexical tokenization."""

import re
from dataclasses import dataclass
from typing import Protocol

from src.preprocessing.chunking import Chunk


def tokenize(text):
    # Preserve numbers/decimals; do not stem drug names or discard negation.
    return re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+(?:[-'][^\W\d_]+)*", text.casefold(), re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[SearchResult]: ...


def matches(chunk, filters):
    return all(
        getattr(chunk, field, None) in value
        if isinstance(value, (list, tuple))
        else getattr(chunk, field, None) == value
        for field, value in (filters or {}).items()
    )


def ranked(chunks, scores, k, filters=None):
    if k < 1:
        raise ValueError("k must be positive")
    rows = [
        SearchResult(c, float(s))
        for c, s in zip(chunks, scores, strict=True)
        if s > 0 and matches(c, filters)
    ]
    return sorted(rows, key=lambda r: (-r.score, r.chunk.chunk_id))[:k]
