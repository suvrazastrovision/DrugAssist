"""Command-line tools for FDA label acquisition, retrieval and evaluation."""

import argparse

from src.utils.config import ROOT, load_config, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "ingest",
            "prepare-benchmark",
            "benchmark",
            "verify-benchmark",
            "search",
            "maintain",
        ],
    )
    parser.add_argument("question", nargs="?", help="Question for search; omit to enter interactive mode")
    parser.add_argument("--config")
    parser.add_argument("--snapshot")
    parser.add_argument(
        "--step", choices=["acquire", "index", "verify", "all"], default="verify",
        help="Maintenance step; defaults to offline verification. 'all' acquires, indexes and verifies.",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Explicitly allow draft qrels; results are not scientific validation",
    )
    parser.add_argument("--compare-embeddings", action="store_true")
    parser.add_argument("--no-tune-chunks", action="store_true")
    parser.add_argument(
        "--method",
        default=None,
        choices=["selected", "keyword", "bm25", "dense", "hybrid_rrf", "hybrid_normalized", "hybrid_rerank"],
    )
    parser.add_argument("--k", type=int, default=5, help="Number of search passages to return")
    parser.add_argument("--run", help="Saved run/index descriptor; defaults to the active search index")
    parser.add_argument("--section", help="Exact FDA section name")
    parser.add_argument("--set-id", help="Restrict search to a label set")
    parser.add_argument("--document-id", help="Restrict search to one document/version")
    parser.add_argument("--effective-time", help="Restrict search to a label effective date (YYYYMMDD)")
    parser.add_argument("--json", action="store_true", help="Print structured results for a single question")
    args = parser.parse_args()
    if args.command == "search":
        import sys

        from src.retrieval.search import run_search

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if args.json and args.question is None:
            parser.error("Provide a quoted question when using --json")
        filters = {
            key: value
            for key, value in {
                "section": args.section,
                "set_id": args.set_id,
                "document_id": args.document_id,
                "effective_time": args.effective_time,
            }.items()
            if value
        }
        try:
            method = args.method
            if method is None:
                method = (
                    "selected"
                    if args.run
                    else load_config(args.config).get("api", {}).get("method", "selected")
                )
            run_search(args.question, method, args.k, filters, args.run, args.snapshot, args.json)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            parser.exit(1, f"Search failed: {exc}\n")
        return
    if args.question is not None:
        parser.error("A question argument is only supported by the search command")
    config = load_config(args.config)
    if args.command == "maintain":
        from src.maintenance import maintain

        maintain(config, args.step)
        return
    from src.ingestion.data_loader import ingest, load_snapshot

    if args.command == "ingest":
        snapshot = ingest(config)
        print(snapshot["manifest"]["snapshot_id"])
        return
    snapshot_id = args.snapshot
    if snapshot_id is None and args.command == "verify-benchmark":
        snapshot_id = read_json(ROOT / "data/benchmarks/benchmark.json")["snapshot_id"]
    snapshot = load_snapshot(snapshot_id)
    if args.command == "prepare-benchmark":
        from src.evaluation.benchmark_data import make_benchmark

        benchmark = make_benchmark(snapshot)
        print(benchmark["benchmark_id"], benchmark["status"])
    elif args.command == "verify-benchmark":
        from src.evaluation.benchmark_data import validate_benchmark

        validate_benchmark(
            read_json(ROOT / "data/benchmarks/benchmark.json"),
            [d for d in snapshot["documents"] if d["kind"] == "fda_full"],
        )
        print("Reviewed benchmark validated against frozen FDA corpus")
    elif args.command == "benchmark":
        from src.evaluation.benchmarking import run_benchmark

        run = run_benchmark(
            snapshot, config, args.provisional, args.compare_embeddings, not args.no_tune_chunks
        )
        print("Completed", run["run_id"], "provisional:", run["provisional"])


if __name__ == "__main__":
    main()
