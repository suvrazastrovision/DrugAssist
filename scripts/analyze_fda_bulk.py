"""Count the full approved FDA corpus locally without embeddings or index creation."""

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.download_fda_bulk import (  # noqa: E402
    baseline_inventory, read_json, sha256, validate_partition, verify_baseline,
)
from src.ingestion.bulk_analysis import (  # noqa: E402
    Paused, ResourceGuard, analysis_report, baseline_comparison, build_approvals,
    completed, match_partition, open_database, size_partition, write_readable_report,
)
from src.utils.config import digest, load_config, utcnow, write_json  # noqa: E402


@contextmanager
def workspace_lock(path):
    """Use the downloader's OS-level lock to prevent concurrent source/state mutation."""
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def offline_tokenizer(folder):
    """Seed the existing checked-in tokenizer asset; never request vocabulary over HTTP."""
    url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    asset = REPO / "tests/assets/cl100k_base.tiktoken"
    expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    if sha256(asset) != expected:
        raise ValueError("Checked-in tokenizer asset checksum mismatch")
    cache = folder / "tokenizer"
    cache.mkdir(exist_ok=True)
    target = cache / hashlib.sha1(url.encode()).hexdigest()
    if not target.exists() or sha256(target) != expected:
        target.write_bytes(asset.read_bytes())
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)


def verify_sources(workspace):
    """Verify pinned manifest and every downloaded byte before trusting saved checkpoints."""
    manifest = read_json(workspace / "fda_manifest.json")
    state = read_json(workspace / "download_state.json")
    report = read_json(workspace / "download_report.json")
    if not report["complete"] or sha256(workspace / "fda_manifest.json") != report["manifest_sha256"]:
        raise ValueError("Download manifest is incomplete or changed")
    partitions = {}
    for endpoint in ("drugsfda", "label"):
        partitions[endpoint] = []
        for part in manifest["results"]["drug"][endpoint]["partitions"]:
            name = validate_partition(part, endpoint)
            path = workspace / "downloads" / name
            saved = state[name]
            if saved["url"] != part["file"] or path.stat().st_size != saved["bytes"]:
                raise ValueError(f"Source size/URL changed: {name}")
            if sha256(path) != saved["sha256"]:
                raise ValueError(f"Source checksum changed: {name}")
            partitions[endpoint].append((path, part["records"], saved["sha256"]))
            print(f"Verified source {name}", flush=True)
    return manifest, report, partitions


def run(args):
    """Run a trial or resumable full analysis and always report baseline preservation."""
    workspace = args.workspace.resolve()
    if workspace.is_relative_to(REPO) or "onedrive" in str(workspace).lower():
        raise ValueError("Use the downloaded bulk workspace outside the repository and OneDrive")
    folder = workspace / "analysis"
    if args.stage == "status":
        path = folder / "sizing_report.json"
        if path.exists():
            print(path.read_text(encoding="utf-8"))
        else:
            print("No sizing report yet; inspect analysis/progress.json if a run is active.")
        return
    folder.mkdir(exist_ok=True)
    guard = ResourceGuard(folder)
    guard.check()
    with workspace_lock(workspace / "download.lock"):
        manifest, source_report, partitions = verify_sources(workspace)
        config = load_config()
        code = [REPO / name for name in (
            "src/ingestion/bulk_analysis.py", "scripts/analyze_fda_bulk.py",
            "src/preprocessing/normalization.py", "src/preprocessing/chunking.py",
        )]
        fingerprint = digest({
            "sources": read_json(workspace / "download_state.json"),
            "manifest": source_report["manifest_sha256"],
            "config": config, "code": {str(p.relative_to(REPO)): sha256(p) for p in code},
        })
        backups = folder / "checkpoint-backups"
        backup_bytes = sum(p.stat().st_size for p in backups.rglob("*") if p.is_file())
        db = open_database(folder / "analysis.sqlite3", fingerprint,
                           max_bytes=guard.max_bytes - backup_bytes)
        baseline_path = folder / "baseline_before.json"
        if not baseline_path.exists():
            write_json(baseline_path, baseline_inventory())
        baseline = read_json(baseline_path)
        offline_tokenizer(folder)
        snapshot_id = digest([source_report["manifest_sha256"], config["chunking"], "approval-matched"])[:20]
        started, timings = time.perf_counter(), {}
        status, error = "in_progress", None

        def progress(message, flush=True):
            print(message, flush=flush)
            write_json(folder / "progress.json", {
                "updated_at": utcnow(), "message": message,
                "elapsed_seconds_this_invocation": round(time.perf_counter() - started, 2),
            })

        def report_now():
            report = analysis_report(db, source_report, config, fingerprint, guard, status, error)
            report["chunking"] = config["chunking"]
            report["embedding_configuration"] = config["embedding"]
            report["timings_this_invocation_seconds"] = timings
            report["snapshot_id"] = snapshot_id
            report["exports"] = source_report["exports"]
            write_json(folder / "sizing_report.json", report)
            write_readable_report(folder / "sizing_report.txt", report)
            return report

        try:
            offline_tokenizer(folder)
            guard.check()
            phase = time.perf_counter()
            for path, count, _ in partitions["drugsfda"]:
                progress(f"Building approval lookup: {path.name}")
                build_approvals(db, path, count, guard, progress)
            timings["approvals"] = round(time.perf_counter() - phase, 3)
            if args.stage in ("match", "all"):
                phase = time.perf_counter()
                selected_parts = partitions["label"][:args.limit_partitions]
                for path, count, _ in selected_parts:
                    progress(f"Matching all label records: {path.name}")
                    match_partition(db, path, count, guard, progress)
                    report_now()
                timings["matching"] = round(time.perf_counter() - phase, 3)
            matching_complete = all(completed(db, "label:" + path.name)[1]
                                    for path, _, _ in partitions["label"])
            if matching_complete:
                baseline_records = read_json(
                    REPO / "data/processed" / baseline["snapshot_id"] / "records.json"
                )
                write_json(folder / "baseline_comparison.json", baseline_comparison(db, baseline_records))
            if args.stage in ("size", "all"):
                if not matching_complete:
                    raise ValueError("Finish matching all partitions before sizing label versions")
                phase = time.perf_counter()
                for path, count, source_hash in partitions["label"]:
                    progress(f"Counting selected-label passages: {path.name}")
                    size_partition(
                        db, path, count, source_hash, snapshot_id, config["chunking"], guard, progress
                    )
                    report_now()
                timings["sizing"] = round(time.perf_counter() - phase, 3)
                expected = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
                actual = db.execute("SELECT COUNT(*) FROM sized").fetchone()[0]
                if actual != expected:
                    raise ValueError(f"Selected/sized label mismatch: {actual} != {expected}")
                status = "complete"
            else:
                status = "matching_complete" if matching_complete else "partial_matching"
        except (KeyboardInterrupt, Paused) as exc:
            db.rollback()
            status, error = "paused", str(exc) or "Interrupted by user"
        except Exception as exc:
            db.rollback()
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            verification = verify_baseline(baseline)
            write_json(folder / "baseline_verification.json", verification)
            if not verification["unchanged"]:
                status, error = "failed", "Baseline changed; inspect baseline_verification.json"
            timings["total"] = round(time.perf_counter() - started, 3)
            report = report_now()
            print((folder / "sizing_report.txt").read_text(encoding="utf-8"), flush=True)
            db.close()
        if status in ("failed", "paused"):
            raise SystemExit(2 if status == "paused" else 1)
        return report


def main():
    """Parse analysis stages; this command has no embedding or activation option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "match", "size", "status"), default="all")
    parser.add_argument("--limit-partitions", type=int, default=None, help="Trial: only with --stage match")
    args = parser.parse_args()
    if args.limit_partitions is not None and (args.stage != "match" or args.limit_partitions < 1):
        parser.error("--limit-partitions must be positive and requires --stage match")

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Bulk sizing is offline; network access is disabled")

    sys.addaudithook(deny_network)
    run(args)


if __name__ == "__main__":
    main()
