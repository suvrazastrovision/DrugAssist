"""Development-only selection, frozen test evaluation and transparent artifacts."""

import itertools
import platform
import time

import numpy as np
import pandas as pd

from src.embeddings.embedder import EmbeddingUnavailable, LocalEmbedder, OpenAIEmbedder
from src.evaluation.benchmark_data import evidence_coverage, validate_benchmark
from src.evaluation.metrics import retrieval_metrics
from src.evaluation.statistics import holm_adjust, paired_comparison
from src.indexing.chroma_store import ChromaStore
from src.preprocessing.chunking import chunk_documents
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticRetriever
from src.utils.config import ROOT, digest, read_json, utcnow, write_json


def score_query(retriever, query, ks=(1, 3, 5, 10)):
    begin = time.perf_counter()
    rows = retriever.search(query["text"], max(ks))
    elapsed = (time.perf_counter() - begin) * 1000
    units = [r.chunk.document_id + ":" + r.chunk.section for r in rows]
    metrics = retrieval_metrics(units, query["qrels"], ks)
    return {
        "query_id": query["query_id"],
        "partition": query["partition"],
        "drug": query["drug"],
        "difficulty": query["difficulty"],
        "answerable": query["answerable"],
        "latency_ms": elapsed,
        "top_score": rows[0].score if rows else None,
        "empty_result": not rows,
        **metrics,
        **{f"evidence_coverage@{k}": evidence_coverage(rows, query, k) for k in ks},
        "redundancy@10": 1 - len(set(units[:10])) / max(1, len(units[:10])),
        "ranking": [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.document_id,
                "set_id": r.chunk.set_id,
                "section": r.chunk.section,
                "start": r.chunk.start,
                "end": r.chunk.end,
                "score": r.score,
                "identity": r.chunk.identity,
                "source_url": r.chunk.authority_url,
                "text_excerpt": r.chunk.text[:240],
                "grade": query["qrels"].get(r.chunk.document_id + ":" + r.chunk.section),
            }
            for r in rows
        ],
    }


def evaluate(retriever, queries, ks=(1, 3, 5, 10)):
    return [score_query(retriever, q, ks) for q in queries]


def effectiveness(rows, metric="ndcg@10"):
    values = [r[metric] for r in rows if r[metric] is not None]
    return float(np.mean(values)) if values else 0.0


def tune_abstention(dev_rows):
    """Method-specific threshold chosen using development labels only."""
    observed = sorted({r["top_score"] for r in dev_rows if r["top_score"] is not None})
    thresholds = [observed[0] - 1e-9] + observed + [observed[-1] + 1e-9] if observed else [0.0]
    best = None
    for threshold in thresholds:
        positives = [r for r in dev_rows if r["answerable"]]
        negatives = [r for r in dev_rows if not r["answerable"]]
        if not positives or not negatives:
            raise ValueError("Both answerable and unsupported development queries are required")
        tpr = np.mean([r["top_score"] is not None and r["top_score"] >= threshold for r in positives])
        tnr = np.mean([r["top_score"] is None or r["top_score"] < threshold for r in negatives])
        candidate = {
            "threshold": threshold,
            "balanced_accuracy": float((tpr + tnr) / 2),
            "dev_sensitivity": float(tpr),
            "dev_specificity": float(tnr),
        }
        if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
            best = candidate
    return best


def _build_dense(chunks, embedding, config):
    begin = time.perf_counter()
    cls = LocalEmbedder if embedding["provider"] == "local" else OpenAIEmbedder
    embedder = cls(**{k: v for k, v in embedding.items() if k != "provider"})
    store = ChromaStore(chunks, embedder, settings=config["chroma"])
    store.index()
    return SemanticRetriever(store), {
        "embedding": embedding,
        "embedding_fingerprint": embedder.fingerprint,
        "collection_fingerprint": store.fingerprint,
        "collection_name": store.collection.name,
        "corpus_hash": store.corpus_hash,
        "chunks": len(chunks),
        "dimensions": embedder.dimensions,
        "indexing_seconds": time.perf_counter() - begin,
        "new_api_tokens": embedder.usage_tokens,
    }


def run_benchmark(snapshot, config, provisional=False, compare_embeddings=False, tune_chunks=True):
    import psutil
    import torch

    torch.manual_seed(config["seed"])
    torch.set_num_threads(min(4, max(1, psutil.cpu_count(logical=False) or 1)))
    np.random.seed(config["seed"])
    from src.ingestion.integrity import verify_snapshot
    from src.quality.validators import quality_report

    integrity = verify_snapshot(snapshot)
    if not all(quality_report(snapshot)["summary"]["acceptance"].values()):
        raise ValueError("Resolve data quality acceptance failures before benchmarking")
    benchmark = read_json(ROOT / "data/benchmarks/benchmark.json")
    documents = [d for d in snapshot["documents"] if d["kind"] == "fda_full"]
    if benchmark["snapshot_id"] != snapshot["manifest"]["snapshot_id"]:
        raise ValueError("Snapshot mismatch; rebind benchmark before evaluation")
    validate_benchmark(benchmark, documents, require_review=not provisional)
    dev = [q for q in benchmark["queries"] if q["partition"] == "dev"]
    test = [q for q in benchmark["queries"] if q["partition"] == "test"]
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    code_hash = digest(source_hashes)
    output = (
        ROOT
        / "results"
        / ("provisional_" if provisional else "reviewed_")
        / digest([benchmark, config, compare_embeddings, tune_chunks, code_hash])[:16]
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise ValueError(
            f"Completed immutable run already exists: {output}. Read it instead of retuning on test."
        )
    chunk_trials = []
    candidate_settings = [
        {"strategy": s, "size": n, "overlap": n // 8, "encoding": "cl100k_base"}
        for s in ("fixed", "recursive", "section", "semantic")
        for n in (128, 256, 512)
    ]
    if not tune_chunks:
        candidate_settings = [config["chunking"]]
    for settings in candidate_settings:
        chunks = chunk_documents(documents, **settings)
        trial = evaluate(BM25Retriever(chunks), dev)
        chunk_trials.append(
            {
                **settings,
                "chunks": len(chunks),
                "dev_ndcg@10": effectiveness(trial),
                "dev_evidence_coverage@10": effectiveness(trial, "evidence_coverage@10"),
            }
        )
    # Primary development objective predeclared; tie-breaker favors less index complexity.
    winner = max(chunk_trials, key=lambda t: (t["dev_ndcg@10"], -t["chunks"]))
    selected_chunking = {k: winner[k] for k in ("strategy", "size", "overlap", "encoding")}
    chunks = chunk_documents(documents, **selected_chunking)
    write_json(output / "chunks.json", [c.to_dict() for c in chunks])
    write_json(output / "chunk_trials.json", chunk_trials)
    print(f"Development chunk choice: {selected_chunking}; {len(chunks)} chunks", flush=True)
    lexical_trials = []
    for k1, b in itertools.product((0.8, 1.5, 2.0), (0.25, 0.75)):
        lexical_trials.append(
            {"k1": k1, "b": b, "dev_ndcg@10": effectiveness(evaluate(BM25Retriever(chunks, k1, b), dev))}
        )
    best_lexical = max(lexical_trials, key=lambda t: t["dev_ndcg@10"])
    bm25 = BM25Retriever(chunks, best_lexical["k1"], best_lexical["b"])
    embedding_settings = [config["embedding"]]
    if compare_embeddings:
        embedding_settings.extend(config.get("embedding_candidates", []))
    dense_candidates, embedding_trials = [], []
    for setting in embedding_settings:
        try:
            dense, info = _build_dense(chunks, setting, config)
        except EmbeddingUnavailable as exc:
            dense_candidates.append(None)
            embedding_trials.append({"embedding": setting, "available": False, "error": str(exc)})
            print(f"Embedding candidate unavailable: {exc}", flush=True)
            continue
        dev_rows = evaluate(dense, dev)
        info["available"] = True
        info["new_api_tokens"] = dense.store.embedder.usage_tokens
        info["stored_vector_sha256"] = dense.store.embedding_digest()
        info["dev_ndcg@10"] = effectiveness(dev_rows)
        dense_candidates.append(dense)
        embedding_trials.append(info)
        print(f"Indexed {setting['model']}: dev nDCG@10={info['dev_ndcg@10']:.4f}", flush=True)
    available = [i for i, trial in enumerate(embedding_trials) if trial["available"]]
    if not available:
        raise RuntimeError(
            "No embedding candidates are available; credentials and account access need resolution"
        )
    best_embedding = max(available, key=lambda i: embedding_trials[i]["dev_ndcg@10"])
    dense = dense_candidates[best_embedding]
    fusion_trials, fusion_candidates = [], []
    for fusion, constant, alpha in [
        ("rrf", 20, 0.5),
        ("rrf", 60, 0.5),
        ("normalized", 60, 0.25),
        ("normalized", 60, 0.5),
        ("normalized", 60, 0.75),
    ]:
        method = HybridRetriever(bm25, dense, config["retrieval"]["candidates"], fusion, constant, alpha)
        score = effectiveness(evaluate(method, dev))
        fusion_candidates.append(method)
        fusion_trials.append({"fusion": fusion, "rrf_k": constant, "alpha": alpha, "dev_ndcg@10": score})
    best_rrf = max(
        (i for i, f in enumerate(fusion_trials) if f["fusion"] == "rrf"),
        key=lambda i: fusion_trials[i]["dev_ndcg@10"],
    )
    best_normalized = max(
        (i for i, f in enumerate(fusion_trials) if f["fusion"] == "normalized"),
        key=lambda i: fusion_trials[i]["dev_ndcg@10"],
    )
    best_fusion = max(range(len(fusion_trials)), key=lambda i: fusion_trials[i]["dev_ndcg@10"])
    hybrid = fusion_candidates[best_fusion]
    rerank = CrossEncoderReranker(hybrid, **config["reranker"], candidates=config["retrieval"]["candidates"])
    methods = {
        "keyword": KeywordRetriever(chunks),
        "bm25": bm25,
        "dense": dense,
        "hybrid_rrf": fusion_candidates[best_rrf],
        "hybrid_normalized": fusion_candidates[best_normalized],
        "hybrid_rerank": rerank,
    }
    # Prewarm query embeddings before timing. Latency is explicitly warm-cache online retrieval.
    dense.store.embedder.encode([q["text"] for q in benchmark["queries"]])
    dev_results, thresholds = {}, {}
    for name, method in methods.items():
        print(f"Evaluating development: {name}", flush=True)
        dev_results[name] = evaluate(method, dev, config["evaluation"]["ks"])
        thresholds[name] = tune_abstention(dev_results[name])
    preferred = max(methods, key=lambda name: effectiveness(dev_results[name]))
    lock = {
        "locked_at": utcnow(),
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_content_hash": digest(benchmark),
        "snapshot_id": benchmark["snapshot_id"],
        "chunking": selected_chunking,
        "bm25": best_lexical,
        "embedding": embedding_trials[best_embedding],
        "hybrid": fusion_trials[best_fusion],
        "reranker": config["reranker"],
        "thresholds": thresholds,
        "preferred_on_dev": preferred,
        "primary_objective": "document-section ndcg@10",
        "test_queries": [q["query_id"] for q in test],
    }
    write_json(output / "selection_lock.json", lock)
    test_results = {}
    for name, method in methods.items():
        print(f"Evaluating held-out test: {name}", flush=True)
        test_results[name] = evaluate(method, test, config["evaluation"]["ks"])
    per_query = []
    for name in methods:
        for row in dev_results[name] + test_results[name]:
            accepted = row["top_score"] is not None and row["top_score"] >= thresholds[name]["threshold"]
            per_query.append(
                {
                    "method": name,
                    **row,
                    "accepted_evidence": accepted,
                    "rejection_correct": accepted == row["answerable"],
                }
            )
    flat = pd.DataFrame([{k: v for k, v in row.items() if k != "ranking"} for row in per_query])
    aggregate = []
    metric_columns = [c for c in flat if "@" in c] + ["latency_ms", "rejection_correct"]
    for (method, partition), group in flat.groupby(["method", "partition"]):
        row = {"method": method, "partition": partition, "queries": len(group)}
        row.update({metric: float(group[metric].mean()) for metric in metric_columns})
        row["p95_latency_ms"] = float(group.latency_ms.quantile(0.95))
        row["warm_queries_per_second"] = 1000 / group.latency_ms.mean()
        negative = group[~group.answerable]
        positive = group[group.answerable]
        row["unsupported_false_acceptance_rate"] = float(negative.accepted_evidence.mean())
        row["answerable_false_rejection_rate"] = float((~positive.accepted_evidence).mean())
        aggregate.append(row)
    stratified = flat.groupby(["method", "partition", "difficulty"], as_index=False)[metric_columns].mean()
    comparisons = []
    planned = [(name, "bm25") for name in methods if name != "bm25"]
    planned.append(
        (
            "hybrid_rerank",
            "hybrid_" + fusion_trials[best_fusion]["fusion"].replace("normalized", "normalized"),
        )
    )
    for left, right in planned:
        a = {r["query_id"]: r for r in test_results[left] if r["answerable"]}
        b = {r["query_id"]: r for r in test_results[right] if r["answerable"]}
        ids = sorted(a)
        stats = paired_comparison(
            [a[q]["ndcg@10"] for q in ids],
            [b[q]["ndcg@10"] for q in ids],
            [a[q]["drug"] for q in ids],
            config["seed"],
            config["evaluation"]["bootstrap_samples"],
            config["evaluation"]["permutation_samples"],
        )
        comparisons.append({"left": left, "right": right, "metric": "ndcg@10", **stats})
    for row, adjusted in zip(comparisons, holm_adjust([r["p_value"] for r in comparisons]), strict=True):
        row["holm_adjusted_p"] = adjusted
    chroma_bytes = sum(p.stat().st_size for p in (ROOT / "data/chroma").rglob("*") if p.is_file())
    run = {
        "run_id": output.name,
        "created_at": utcnow(),
        "provisional": provisional,
        "benchmark_status": benchmark["status"],
        "benchmark_id": benchmark["benchmark_id"],
        "snapshot_id": benchmark["snapshot_id"],
        "configuration": config,
        "source_code_hash": code_hash,
        "source_files": source_hashes,
        "snapshot_integrity": integrity,
        "selection": lock,
        "chunk_trials": chunk_trials,
        "bm25_trials": lexical_trials,
        "embedding_trials": embedding_trials,
        "fusion_trials": fusion_trials,
        "summary": aggregate,
        "stratified": stratified.where(pd.notna(stratified), None).to_dict(orient="records"),
        "comparisons": comparisons,
        "per_query": per_query,
        "resources": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "chroma_directory_bytes_all_experiments": chroma_bytes,
        },
        "limitations": benchmark["limitations"]
        + [
            "Development set is small and repeatedly used for configuration selection; risk of overfitting remains.",
            "Confidence intervals and p-values are exploratory with only eight test drug clusters.",
            "Warm-cache latency excludes network embedding calls, model loading and index construction.",
            "Hosted embedding aliases cannot pin provider weights; persist vectors and usage for this run.",
            "No monetary cost inferred: API tokens, dimensions, elapsed time and disk bytes are recorded.",
            "General-domain embeddings and MS MARCO reranker are not validated biomedical models.",
            "Automated thresholds detect some unsupported queries; score confidence does not prove evidence sufficiency.",
            "No final scientific superiority claim is valid until independent annotation review is complete.",
        ],
    }
    # Convert any numpy scalar/NaN artifacts to ordinary JSON null, not nonstandard NaN.
    run = read_json_compatible(run)
    write_json(output / "run.json", run)
    flat.to_csv(output / "per_query.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "summary.csv", index=False)
    stratified.to_csv(output / "stratified.csv", index=False)
    write_json(ROOT / "results/latest.json", {"run_path": str((output / "run.json").relative_to(ROOT))})
    return run


def read_json_compatible(value):
    if isinstance(value, dict):
        return {str(k): read_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [read_json_compatible(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_run():
    return read_json(ROOT / read_json(ROOT / "results/latest.json")["run_path"])
