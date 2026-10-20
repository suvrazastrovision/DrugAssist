"""Build a searchable expanded corpus using frozen prior experiment settings.

This creates an index descriptor, never fabricated evaluation scores. The
original benchmark, query judgments and results pointer remain unchanged.
"""

import time
from copy import deepcopy

from src.embeddings.embedder import LocalEmbedder, OpenAIEmbedder
from src.indexing.chroma_store import ChromaStore
from src.ingestion.coverage import coverage_report
from src.ingestion.integrity import verify_snapshot
from src.preprocessing.chunking import chunk_documents
from src.quality.validators import quality_report
from src.utils.config import ROOT, digest, read_json, utcnow, write_json


def build_search_index(snapshot, config, baseline_path=None, activate=True):
    baseline_path = baseline_path or ROOT / read_json(ROOT / "results/latest.json")["run_path"]
    baseline = read_json(baseline_path)
    coverage = coverage_report(snapshot, config["ingestion"]["lookup_names"])
    if coverage["covered_names"] != len(config["ingestion"]["lookup_names"]):
        raise ValueError(
            "Every configured drug name must have matching full FDA identity evidence before activation"
        )
    integrity = verify_snapshot(snapshot)
    quality = quality_report(snapshot)
    if not all(quality["summary"]["acceptance"].values()):
        raise ValueError("Data quality acceptance failed; the active index was not changed")
    selected = baseline["selection"]
    chunking = selected["chunking"]
    embedding = selected["embedding"]["embedding"]
    documents = [document for document in snapshot["documents"] if document["kind"] == "fda_full"]
    chunks = chunk_documents(documents, **chunking)
    identifier = digest([snapshot["manifest"]["snapshot_id"], baseline["run_id"], chunking, embedding])[:20]
    folder = ROOT / "results/search_indexes" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "chunks.json", [chunk.to_dict() for chunk in chunks])
    print(
        f"Indexing {coverage['covered_names']} drug names, {len(documents)} FDA records, {len(chunks)} chunks",
        flush=True,
    )
    cls = LocalEmbedder if embedding["provider"] == "local" else OpenAIEmbedder
    embedder = cls(**{key: value for key, value in embedding.items() if key != "provider"})
    store = ChromaStore(chunks, embedder, settings=baseline["configuration"]["chroma"])
    begin = time.perf_counter()

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(f"Indexed {done}/{total} pending chunks", flush=True)

    store.index(progress=progress)
    if store.collection.count() != len(chunks):
        raise RuntimeError("Index is incomplete; the active index was not changed")
    fusion_settings = {}
    for kind in ("rrf", "normalized"):
        prior = max(
            (row for row in baseline["fusion_trials"] if row["fusion"] == kind),
            key=lambda row: row["dev_ndcg@10"],
        )
        fusion_settings["hybrid_" + kind] = {key: prior[key] for key in ("fusion", "rrf_k", "alpha")}
    selection = {
        "chunking": chunking,
        "bm25": {key: selected["bm25"][key] for key in ("k1", "b")},
        "hybrid": {key: selected["hybrid"][key] for key in ("fusion", "rrf_k", "alpha")},
        "reranker": selected["reranker"],
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
        "provisional": True,
        "evaluation_status": "expanded_corpus_not_benchmarked",
        "configuration_source_run_id": baseline["run_id"],
        "default_method": selected["preferred_on_dev"],
        "selection": selection,
        "fusion_settings": fusion_settings,
        "coverage": coverage,
        "configuration": deepcopy(baseline["configuration"]),
        "snapshot_integrity": integrity,
        "quality_summary": quality["summary"],
        "limitations": [
            "Prior 12-drug benchmark scores do not apply to the expanded corpus.",
            "Retrieval parameters are transferred without new tuning or evaluation.",
            coverage["selection"],
            coverage["coverage_definition"],
        ],
    }
    descriptor["configuration"]["ingestion"] = deepcopy(config["ingestion"])
    write_json(folder / "index.json", descriptor)
    if activate:
        write_json(
            ROOT / "results/search_latest.json", {"run_path": str((folder / "index.json").relative_to(ROOT))}
        )
        write_json(ROOT / "data/processed/latest.json", {"snapshot_id": descriptor["snapshot_id"]})
    return descriptor
