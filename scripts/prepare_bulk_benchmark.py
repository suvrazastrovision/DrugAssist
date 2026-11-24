"""Prepare deterministic whole-label benchmark samples; never request embeddings."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ingestion.bulk_analysis import ResourceGuard  # noqa: E402
from src.utils.config import canonical, utcnow, write_json  # noqa: E402


def select_samples(rows, targets=(2000, 10000)):
    """Round-robin deterministic length quartiles, retaining whole sets and nested samples."""
    groups = [[] for _ in range(4)]
    for row in rows:
        if row["passages"] > 0:
            groups[row["stratum"] - 1].append(row)
    for group in groups:
        group.sort(key=lambda row: hashlib.sha256(row["set_id"].encode()).digest())
    sequence = []
    for position in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if position < len(group):
                sequence.append(group[position])
    samples, selected, passages, tokens, cursor = [], [], 0, 0, 0
    for target in sorted(set(targets)):
        while passages < target and cursor < len(sequence):
            row = sequence[cursor]
            selected.append(row)
            passages += row["passages"]
            tokens += row["embedding_tokens"]
            cursor += 1
        samples.append({
            "target_passages": target, "actual_passages": passages,
            "label_sets": len(selected), "embedding_input_tokens_upper_bound": tokens,
            "strata_represented": sorted({row["stratum"] for row in selected}),
            "records": list(selected),
        })
    return samples


def main():
    """Read a complete sizing database and save a reviewable, unpaid benchmark manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--usd-per-million", type=float, help="Verified token price; no API calls")
    parser.add_argument("--pricing-source", help="Documentation URL supporting the supplied price")
    args = parser.parse_args()
    if args.usd_per_million is not None and (
        not math.isfinite(args.usd_per_million) or args.usd_per_million < 0 or not args.pricing_source
    ):
        parser.error("A finite nonnegative token price requires --pricing-source")
    workspace = args.workspace.resolve()
    if workspace.is_relative_to(REPO) or "onedrive" in str(workspace).lower():
        parser.error("Use the separate bulk workspace outside OneDrive")
    folder = workspace / "analysis"
    report = json.loads((folder / "sizing_report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        parser.error("Full-corpus sizing must complete before preparing benchmark samples")
    guard = ResourceGuard(folder)
    guard.check()
    db = sqlite3.connect((folder / "analysis.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        if identity["fingerprint"] != report["fingerprint"]:
            raise ValueError("Sizing database and report fingerprints differ")
        rows = [dict(zip(("set_id", "source", "ordinal", "source_hash", "passages",
                          "embedding_tokens", "stratum"), row, strict=True)) for row in db.execute("""
            SELECT s.set_id, s.source, s.ordinal, s.source_hash, z.passages, z.embedding_tokens,
                   NTILE(4) OVER (ORDER BY z.passages, s.set_id)
            FROM selected s JOIN sized z ON s.set_id=z.set_id WHERE z.passages > 0
        """)]
    finally:
        db.close()
    plan = {
        "generated_at": utcnow(), "corpus_fingerprint": report["fingerprint"],
        "selection": "Whole label sets; passage-count quartiles; deterministic SHA-256 ordering",
        "samples": select_samples(rows),
        "tokens_note": "Upper bound before duplicate-input or existing-cache savings; no new requests made",
        "embedding_configuration": report["embedding_configuration"],
        "paid_execution_authorized": False,
        "benchmark_protocol": {
            "indexing": "Fresh process: elapsed time, peak process-tree RSS, cache and index disk growth",
            "search": "Separate process: cold startup; 100 fixed queries; warm p50/p95 and per-stage latency",
            "pipeline": "Existing Chroma, BM25, fusion and pinned reranker settings",
            "server_acceptance": "Full collection; RAM headroom; active and replacement index disk space",
            "hosting_budget_eur_per_month": 50,
        },
    }
    if args.usd_per_million is not None:
        unit = args.usd_per_million / 1_000_000
        plan["cost_estimates"] = {
            "currency": "USD", "input_price_per_million": args.usd_per_million,
            "pricing_source": args.pricing_source, "recorded_at": utcnow(),
            "full_corpus_before_deduplication": report["sizing"]["embedding_input_tokens"] * unit,
            "full_corpus_unique_inputs": report["sizing"]["unique_embedding_input_tokens"] * unit,
            "exclusions": "Tax, currency conversion, retries, query embeddings and generated answers",
        }
        for sample in plan["samples"]:
            sample["estimated_embedding_cost_usd_before_cache_savings"] = (
                sample["embedding_input_tokens_upper_bound"] * unit
            )
    if guard.usage() + len(canonical(plan).encode("utf-8")) * 2 >= guard.max_bytes:
        raise RuntimeError("Benchmark manifest would exceed the analysis storage allowance")
    write_json(folder / "benchmark_plan.json", plan)
    guard.check()
    for sample in plan["samples"]:
        print(f"Target {sample['target_passages']:,}: {sample['actual_passages']:,} passages, "
              f"{sample['label_sets']:,} label sets, "
              f"<= {sample['embedding_input_tokens_upper_bound']:,} tokens")
    print("No embeddings requested. Price the sample and set a spending limit before indexing.")


if __name__ == "__main__":
    main()
