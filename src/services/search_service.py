"""Read-only application service shared by API callers and engineering tests."""

import math
import re
from threading import RLock

from src.ingestion.data_loader import load_snapshot
from src.query_parser import QueryParser
from src.retrieval.search import SearchSession
from src.utils.config import ROOT, load_config

DISCLAIMER = "FDA label excerpts are for information only, not medical advice. Consult a qualified healthcare professional."
SCORE_KINDS = {
    "keyword": "weighted_keyword_count",
    "bm25": "bm25_score",
    "dense": "cosine_similarity",
    "hybrid_rrf": "weighted_reciprocal_rank_fusion",
    "hybrid_normalized": "weighted_minmax_fusion",
    "hybrid_rerank": "cross_encoder_score",
}
# General non-label topics; these rules do not come from evaluation questions.
OUTSIDE_LABEL_SCOPE = re.compile(
    r"\b(?:stock price|share price|weather|lottery|bitcoin|football score)\b", re.IGNORECASE
)


class ServiceUnavailable(RuntimeError):
    pass


class SearchService:
    def __init__(self, session, documents, *, min_score, threshold_method, known_names=()):
        if threshold_method != session.method:
            raise ValueError("The threshold must name the active scoring method")
        if isinstance(min_score, bool) or not math.isfinite(min_score):
            raise ValueError("The experimental threshold must be finite")
        self.session, self.min_score = session, min_score
        indexed_documents = {chunk.document_id for chunk in session.chunks}
        self.documents = {
            doc["document_id"]: doc for doc in documents if doc["document_id"] in indexed_documents
        }
        self.parser = QueryParser(self.documents.values(), known_names)
        self.lock = RLock()

    def health(self):
        try:
            with self.lock:
                count = self.session.store.collection.count()
                if count != len(self.session.chunks):
                    raise ValueError("The index is incomplete")
            return {
                "status": "ready",
                "chromadb": "connected",
                "indexed_chunks": count,
                "indexed_documents": len({chunk.document_id for chunk in self.session.chunks}),
            }
        except Exception:  # noqa: BLE001 -- Public service boundary; redact database details.
            raise ServiceUnavailable() from None

    def search(self, query, k=5):
        parsed = self.parser.parse(query)
        filters = self.parser.filters(parsed)
        reason, hits = None, []
        if OUTSIDE_LABEL_SCOPE.search(query) or not (self.parser.recognizes_drug(query) or parsed.condition):
            reason = "outside_supported_label_scope"
        else:
            try:
                with self.lock:
                    # Confirm connectivity even when filters have no eligible passages.
                    self.health()
                    response = self.session.search(query.strip(), k=k, filters=filters)
                eligible = [row for row in response["results"] if row["score"] >= self.min_score]
                for rank, row in enumerate(eligible, 1):
                    doc = self.documents.get(row["document_id"], {})
                    hits.append(
                        {
                            **row,
                            "rank": rank,
                            "drug_name": row["identity"],
                            "label_id": doc.get("label_id"),
                            "source_url": row["authority_url"],
                        }
                    )
                if not hits:
                    reason = "below_experimental_threshold" if response["results"] else "no_candidates"
            except Exception:  # noqa: BLE001 -- Public service boundary; redact provider errors.
                raise ServiceUnavailable() from None
        return {
            "query": query,
            "parsed": parsed,
            "no_match": not hits,
            "no_match_reason": reason,
            "method": self.session.method,
            "snapshot_id": self.session.snapshot_id,
            "applied_filters": filters,
            "score_kind": SCORE_KINDS[self.session.method],
            "score_note": (
                "Raw cross-encoder relevance scores; higher ranks first and values can be negative or exceed 1. "
                "Scores are not correctness probabilities and are different from the hybrid candidate scores."
                if self.session.method == "hybrid_rerank"
                else "Higher scores rank first within this method. Scores are not correctness probabilities; normalized fusion is candidate-relative."
            ),
            "threshold": {
                "method": self.session.method,
                "min_score": self.min_score,
                "status": "experimental",
            },
            "results": hits,
            "disclaimer": DISCLAIMER,
        }

    def close(self):
        close = getattr(self.session.embedder, "close", None)
        if close is not None:
            close()


def build_search_service():
    settings = dict(load_config()["api"])
    method = settings.pop("method", "selected")
    session = SearchSession(method=method, query_cache=ROOT / "data/cache/embeddings.sqlite3")
    snapshot = load_snapshot(session.snapshot_id)
    return SearchService(
        session,
        snapshot["documents"],
        **settings,
        known_names=session.run["configuration"]["ingestion"]["lookup_names"],
    )
