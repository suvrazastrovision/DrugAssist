"""Build a searchable corpus from explicit settings and verified source files."""

import time
from copy import deepcopy

from src.embeddings.embedder import LocalEmbedder, OpenAIEmbedder
from src.indexing.chroma_store import ChromaStore
from src.ingestion.coverage import coverage_report
from src.ingestion.integrity import verify_snapshot
from src.preprocessing.chunking import chunk_documents
from src.quality.validators import quality_report
from src.utils.config import ROOT, digest, utcnow, write_json


def build_search_index(snapshot, config, activate=True):
    coverage = coverage_report(snapshot, config["ingestion"]["lookup_names"])
    if coverage["covered_names"] != len(config["ingestion"]["lookup_names"]):
        raise ValueError(
            "Every configured drug name must have matching full FDA identity evidence before activation"
        )
    integrity = verify_snapshot(snapshot)
    quality = quality_report(snapshot)
    if not all(quality["summary"]["acceptance"].values()):
        raise ValueError("Data quality acceptance failed; the active index was not changed")
    chunking = config["chunking"]
    embedding = config["embedding"]
    retrieval = config["retrieval"]
    documents = [document for document in snapshot["documents"] if document["kind"] == "fda_full"]
    chunks = chunk_documents(documents, **chunking)
    identifier = digest([snapshot["manifest"]["snapshot_id"], config])[:20]
    folder = ROOT / "results/search_indexes" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "chunks.json", [chunk.to_dict() for chunk in chunks])
    print(
        f"Indexing {coverage['covered_names']} drug names, {len(documents)} FDA records, {len(chunks)} chunks",
        flush=True,
    )
    cls = LocalEmbedder if embedding["provider"] == "local" else OpenAIEmbedder
    embedder = cls(**{key: value for key, value in embedding.items() if key != "provider"})
    store = ChromaStore(chunks, embedder, settings=config["chroma"])
    begin = time.perf_counter()

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(f"Indexed {done}/{total} pending chunks", flush=True)

    store.index(progress=progress)
    if store.collection.count() != len(chunks):
        raise RuntimeError("Index is incomplete; the active index was not changed")
    selection = {
        "chunking": chunking,
        "bm25": {key: retrieval[key] for key in ("k1", "b")},
        "hybrid": {key: retrieval[key] for key in ("fusion", "rrf_k", "alpha")},
        "reranker": config["reranker"],
        "embedding": {
            "embedding": embedding,
            "embedding_fingerprint": embedder.fingerprint,
            "collection_fingerprint": store.fingerprint,
            "collection_name": store.collection.name,
            "corpus_hash": store.corpus_hash,
            "chunks": len(chunks),
            "dimensions": embedder.dimensions,
            "stored_vector_sha256": store.embedding_digest(),
            "new_api_tokens": embedder.usage_tokens,
            "indexing_seconds": time.perf_counter() - begin,
        },
    }
    descriptor = {
        "artifact_type": "search_index",
        "run_id": "index_" + identifier,
        "snapshot_id": snapshot["manifest"]["snapshot_id"],
        "created_at": utcnow(),
        "evaluation_status": "not_independently_reviewed",
        "default_method": config["api"]["method"],
        "selection": selection,
        "fusion_settings": deepcopy(config["fusion_settings"]),
        "coverage": coverage,
        "configuration": deepcopy(config),
        "snapshot_integrity": integrity,
        "quality_summary": quality["summary"],
        "limitations": [
            "Building an index does not evaluate retrieval relevance or clinical answer quality.",
            coverage["selection"],
            coverage["coverage_definition"],
        ],
    }
    write_json(folder / "index.json", descriptor)
    if activate:
        write_json(
            ROOT / "results/search_latest.json", {"run_path": str((folder / "index.json").relative_to(ROOT))}
        )
        write_json(ROOT / "data/processed/latest.json", {"snapshot_id": descriptor["snapshot_id"]})
    return descriptor
