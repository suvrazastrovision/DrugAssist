"""Rank passages using Okapi BM25 term statistics."""

import math
from collections import Counter

import numpy as np

from src.retrieval.base import ranked, tokenize


class BM25Retriever:
    """Okapi BM25 with positive Robertson IDF, k1=1.5 and b=0.75 by default."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        """Compute corpus term frequencies and validate BM25 hyperparameters."""
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 hyperparameters")
        self.chunks, self.k1, self.b = chunks, k1, b
        self.counts = [Counter(tokenize(c.embedding_text)) for c in chunks]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avgdl = float(self.lengths.mean()) if len(chunks) else 1.0
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(chunks) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query, k=10, filters=None):
        """Rank filtered passages by BM25 score for the query's unique terms."""
        scores = np.zeros(len(self.chunks))
        normalizer = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1))
        for term in set(tokenize(query)):
            tf = np.array([c[term] for c in self.counts])
            scores += self.idf.get(term, 0) * tf * (self.k1 + 1) / (tf + normalizer + 1e-12)
        return ranked(self.chunks, scores, k, filters)
