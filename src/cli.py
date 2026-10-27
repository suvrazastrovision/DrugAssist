"""Command-line tools for FDA label acquisition and retrieval."""

import argparse

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["setup", "ingest", "search", "maintain"],
    )
    parser.add_argument("question", nargs="?", help="Question for search; omit to enter interactive mode")
    parser.add_argument("--config")
    parser.add_argument("--snapshot")
    parser.add_argument("--from-source", action="store_true", help="Build fresh data; embedding charges apply")
    parser.add_argument(
        "--step", choices=["acquire", "index", "verify", "all"], default="verify",
        help="Maintenance step; defaults to offline verification. 'all' acquires, indexes and verifies.",
    )
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
    if args.from_source and args.command != "setup":
        parser.error("--from-source is only supported by setup")
    if args.command == "setup":
        if args.question is not None:
            parser.error("setup does not accept a question")
        from src.setup import setup

        try:
            setup(from_source=args.from_source, config_path=args.config)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Setup failed: {exc}\n")
        return
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
    from src.ingestion.data_loader import ingest

    if args.command == "ingest":
        snapshot = ingest(config)
        print(snapshot["manifest"]["snapshot_id"])
        return


if __name__ == "__main__":
    main()
