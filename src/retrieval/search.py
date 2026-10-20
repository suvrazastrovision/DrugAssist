"""Search the saved FDA label index without rebuilding it."""

import json
import textwrap
import time
from pathlib import Path

from src.embeddings.cache import QueryEmbeddingCache
from src.embeddings.embedder import LocalEmbedder, OpenAIEmbedder
from src.indexing.chroma_store import ChromaStore
from src.preprocessing.chunking import Chunk
from src.retrieval.base import matches
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticRetriever
from src.utils.config import ROOT, digest, read_json

METHODS = ("selected", "keyword", "bm25", "dense", "hybrid_rrf", "hybrid_normalized", "hybrid_rerank")
FILTER_FIELDS = {"section", "set_id", "document_id", "effective_time"}


class SearchSession:
    def __init__(
        self,
        run_path=None,
        method="selected",
        snapshot_id=None,
        embedder=None,
        chroma_path=None,
        query_cache=None,
    ):
        self.embedder = embedder
        self.chroma_path = chroma_path
        self.query_cache = query_cache
        self.store = None
        pointer = ROOT / "results/search_latest.json"
        if not run_path and not pointer.exists():
            raise FileNotFoundError(
                "No active search index; restore it or run fda-rag maintain --step index"
            )
        path = Path(run_path) if run_path else ROOT / read_json(pointer)["run_path"]
        self.run = read_json(path)
        self.snapshot_id = self.run["snapshot_id"]
        if snapshot_id and snapshot_id != self.snapshot_id:
            raise ValueError(
                "Requested snapshot does not match the saved index; select its descriptor with --run"
            )
        if method not in METHODS:
            raise ValueError("Unknown retrieval method")
        self.method = self.run["default_method"] if method == "selected" else method
        self.chunks = [Chunk(**row) for row in read_json(path.parent / "chunks.json")]
        expected = self.run["selection"]["embedding"]["corpus_hash"]
        if digest(sorted(c.chunk_id for c in self.chunks)) != expected:
            raise ValueError("Saved chunk inventory differs from the indexed corpus")
        encoding = self.run["selection"]["chunking"]["encoding"]
        for c in self.chunks:
            actual = digest(
                [c.document_id, c.section, c.start, c.end, c.text, c.strategy, c.size, c.overlap, encoding]
            )[:40]
            if c.snapshot_id != self.snapshot_id or c.chunk_id != actual:
                raise ValueError("Saved chunk text or provenance is inconsistent with its identifier")
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        if self.method == "keyword":
            return KeywordRetriever(self.chunks)
        settings = self.run["selection"]["bm25"]
        bm25 = BM25Retriever(self.chunks, k1=settings["k1"], b=settings["b"])
        if self.method == "bm25":
            return bm25
        selected = self.run["selection"]["embedding"]
        embedding = selected["embedding"]
        embedder = self.embedder
        if embedder is None:
            cls = {"local": LocalEmbedder, "openai": OpenAIEmbedder}[embedding["provider"]]
            options = {k: v for k, v in embedding.items() if k != "provider"}
            if cls is OpenAIEmbedder and self.query_cache is not None:
                options["cache"] = self.query_cache
            embedder = cls(**options)
        if embedder.fingerprint != selected["embedding_fingerprint"]:
            raise ValueError("Embedding model differs from the saved index")
        if self.query_cache is not None and not isinstance(embedder, OpenAIEmbedder):
            embedder = QueryEmbeddingCache(embedder, self.query_cache)
        self.embedder = embedder
        store = ChromaStore(
            self.chunks,
            embedder,
            path=self.chroma_path,
            settings=self.run["configuration"]["chroma"],
            create=False,
        )
        self.store = store
        if store.collection.name != selected["collection_name"] or store.collection.count() != len(
            self.chunks
        ):
            raise ValueError("Saved vector collection is incomplete or does not match the index")
        dense = SemanticRetriever(store)
        if self.method == "dense":
            return dense
        candidates = self.run["configuration"]["retrieval"]["candidates"]
        if self.method == "hybrid_rerank":
            fusion = self.run["selection"]["hybrid"]
        else:
            fusion = self.run["fusion_settings"][self.method]
        hybrid = HybridRetriever(
            bm25, dense, candidates=candidates, **{k: fusion[k] for k in ("fusion", "rrf_k", "alpha")}
        )
        if self.method != "hybrid_rerank":
            return hybrid
        import torch

        torch.set_num_threads(4)
        return CrossEncoderReranker(hybrid, candidates=candidates, **self.run["selection"]["reranker"])

    def search(self, question, k=5, filters=None):
        if not question or not question.strip():
            raise ValueError("Enter a nonempty question")
        if not 1 <= k <= 100:
            raise ValueError("Choose --k between 1 and 100")
        filters = filters or {}
        if set(filters) - FILTER_FIELDS:
            raise ValueError("Unsupported metadata filter")
        begin = time.perf_counter()
        eligible = any(matches(c, filters) for c in self.chunks)
        rows = self.retriever.search(question, k, filters) if eligible else []
        return {
            "question": question,
            "method": self.method,
            "run_id": self.run["run_id"],
            "snapshot_id": self.snapshot_id,
            "evaluation_status": self.run.get("evaluation_status", "not_independently_reviewed"),
            "corpus_drug_names": self.run.get("coverage", {}).get("covered_names"),
            "corpus_labels": len({c.set_id for c in self.chunks}),
            "corpus_chunks": len(self.chunks),
            "filters": filters,
            "latency_ms": (time.perf_counter() - begin) * 1000,
            "score_note": "Scores rank candidates within this method; they are not correctness probabilities.",
            "result_note": "Retrieved source passages; no generated answer or automatic clinical validation.",
            "results": [
                {"rank": i, "score": row.score, **row.chunk.to_dict()} for i, row in enumerate(rows, 1)
            ],
        }


def format_results(response):
    lines = [
        f"Question: {response['question']}",
        f"Method: {response['method']}",
        f"Corpus: {response['corpus_labels']} FDA label sets / {response['corpus_chunks']} chunks",
        f"Snapshot: {response['snapshot_id']}",
        response["score_note"],
        response["result_note"],
        "Source-section checks do not establish clinical relevance.",
    ]
    if response.get("corpus_drug_names") is not None:
        lines.insert(3, f"Verified named-drug coverage: {response['corpus_drug_names']}")
    if not response["results"]:
        lines.append("No matching passages found. No filters were relaxed.")
    for row in response["results"]:
        lines.extend(
            [
                "",
                f"{row['rank']}. {row['identity']}",
                f"   Section: {row['section']} | Score: {row['score']:.6f}",
                f"   Label effective date: {row['effective_time']}",
                f"   Label set: {row['set_id']}",
                f"   Document: {row['document_id']}",
                f"   Source: {row['authority_url']}",
                "   Passage:",
                textwrap.indent(row["text"], "    "),
            ]
        )
    return "\n".join(lines)


def run_search(
    question=None, method="selected", k=5, filters=None, run_path=None, snapshot_id=None, as_json=False
):
    session = SearchSession(run_path, method, snapshot_id)
    if question is not None:
        response = session.search(question, k, filters)
        print(json.dumps(response, ensure_ascii=True, indent=2) if as_json else format_results(response))
        return
    print(f"Ready: {session.method}. Enter a question; use /quit to exit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSearch session ended.")
            break
        if question.casefold() in {"/quit", "/exit"}:
            break
        if question:
            print(format_results(session.search(question, k, filters)))
