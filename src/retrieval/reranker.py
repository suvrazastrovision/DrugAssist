import numpy as np

from src.retrieval.base import SearchResult
from src.utils.config import ROOT


class CrossEncoderReranker:
    """Separate relevance model. Inputs are truncated only at model context length."""

    def __init__(self, retriever, model, revision, candidates=30, batch_size=16, max_length=512):
        from sentence_transformers import CrossEncoder

        if not revision or revision == "main":
            raise ValueError("Reranker requires an immutable Hugging Face commit")
        self.retriever, self.candidates = retriever, candidates
        self.model_id, self.revision, self.batch_size = model, revision, batch_size
        # A local snapshot also keeps CrossEncoder's separate AutoConfig lookup offline.
        from src.setup import model_folder

        prepared = model_folder(ROOT, {"model": model, "revision": revision})
        model_path = str(prepared) if (prepared / "config.json").is_file() else model
        self.model = CrossEncoder(
            model_path,
            revision=revision,
            max_length=max_length,
            device="cpu",
            automodel_args={"cache_dir": str(ROOT / "data/models")},
            tokenizer_args={"cache_dir": str(ROOT / "data/models")},
        )

    def search(self, query, k=10, filters=None):
        rows = self.retriever.search(query, max(k + 1, self.candidates), filters)
        if not rows:
            return []
        scores = self.model.predict(
            [(query, row.chunk.embedding_text) for row in rows],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return sorted(
            [
                SearchResult(row.chunk, float(score))
                for row, score in zip(rows, np.asarray(scores).reshape(-1), strict=True)
            ],
            key=lambda row: (-row.score, row.chunk.chunk_id),
        )[:k]
