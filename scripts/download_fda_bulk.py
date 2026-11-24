"""Download pinned FDA label/approval ZIPs without changing the active corpus.

Run from the repository root with its existing Python environment. This command
only downloads and verifies sources; it never chunks, embeds, or builds an index.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
import zipfile

import requests


REPO = Path(__file__).resolve().parents[1]
MANIFEST_URL = "https://api.fda.gov/download.json"
BUFFER = 1024 * 1024
MIN_FREE = 80 * 1024**3
MAX_WORKSPACE = 40 * 1024**3


def read_json(path):
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomically replace a JSON artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    """Hash file contents with bounded memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER), b""):
            value.update(block)
    return value.hexdigest()


def check_space(workspace):
    """Enforce the destination disk reserve and workspace allowance."""
    if shutil.disk_usage(workspace).free < MIN_FREE:
        raise RuntimeError("Paused: fewer than 80 GiB free on the destination drive")
    used = sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file())
    if used >= MAX_WORKSPACE:
        raise RuntimeError("Paused: bulk workspace reached its 40 GiB allowance")


def baseline_inventory():
    """Read and fingerprint the active baseline; never open it for writing."""
    pointer = REPO / "results/search_latest.json"
    index = REPO / read_json(pointer)["run_path"].replace("\\", "/")
    descriptor = read_json(index)
    snapshot = REPO / "data/processed" / descriptor["snapshot_id"]
    files = {REPO / "config/default.json", REPO / "data/processed/latest.json", pointer, index}
    files.update(snapshot.glob("*.json"))
    files.add(index.parent / "chunks.json")
    for envelope in read_json(snapshot / "records.json"):
        files.add(REPO / envelope["provenance"]["raw_file"])
    for folder in (REPO / "data/chroma", REPO / "data/cache"):
        files.update(p for p in folder.rglob("*") if p.is_file())
    chunks = read_json(index.parent / "chunks.json")
    return {
        "snapshot_id": descriptor["snapshot_id"],
        "index": str(index.relative_to(REPO)),
        "drug_names": descriptor["coverage"]["covered_names"],
        "label_records": len({c["document_id"] for c in chunks}),
        "passages": len(chunks),
        "configuration": read_json(REPO / "config/default.json"),
        "files": {
            str(p.relative_to(REPO)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(files)
        },
    }


def verify_baseline(baseline):
    """Compare protected baseline files with their original hashes."""
    changed = [
        name for name, value in baseline["files"].items()
        if not (REPO / name).is_file() or sha256(REPO / name) != value["sha256"]
    ]
    return {"unchanged": not changed, "changed_files": changed}


def validate_partition(partition, endpoint):
    """Require an official HTTPS download for the requested endpoint."""
    parsed = urlparse(partition["file"])
    if (
        parsed.scheme != "https" or parsed.netloc != "download.open.fda.gov"
        or not parsed.path.startswith(f"/drug/{endpoint}/")
        or not parsed.path.endswith(".json.zip") or parsed.query or parsed.fragment
    ):
        raise ValueError("Unexpected FDA partition URL")
    return Path(parsed.path).name


def inspect_zip(path):
    """Verify ZIP CRCs and report sizes without extracting files."""
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC validation failed: {path.name}")
        members = archive.infolist()
        if not members or not any(m.filename.endswith(".json") for m in members):
            raise ValueError("FDA ZIP contains no JSON file")
        return {
            "bytes": path.stat().st_size,
            "uncompressed_bytes": sum(m.file_size for m in members),
            "sha256": sha256(path),
            "zip_crc_verified": True,
        }


def download_partition(session, partition, endpoint, workspace, state):
    """Download or recover one partition, then persist verified completion."""
    name = validate_partition(partition, endpoint)
    destination = workspace / "downloads" / name
    partial = destination.with_suffix(destination.suffix + ".part")
    metadata = partial.with_suffix(partial.suffix + ".json")
    url = partition["file"]
    previous = state.get(name)
    if not destination.exists() and partial.exists() and metadata.exists():
        saved = read_json(metadata)
        if saved.get("url") == url and saved.get("expected_bytes") == partial.stat().st_size:
            inspect_zip(partial)
            partial.replace(destination)
            metadata.unlink()
    if destination.exists():
        checked = inspect_zip(destination)
        if previous and (checked["sha256"] != previous["sha256"] or previous["url"] != url):
            raise RuntimeError(f"Completed download changed: {name}; investigate before continuing")
        print(f"Verified existing {name}", flush=True)
    else:
        for attempt in range(5):
            check_space(workspace)
            saved = read_json(metadata) if metadata.exists() else {}
            offset = partial.stat().st_size if partial.exists() else 0
            validator = saved.get("etag") or saved.get("last_modified")
            resume = offset and validator and saved.get("url") == url
            headers = {"Accept-Encoding": "identity"}
            if resume:
                headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
            try:
                with session.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    append = bool(resume and response.status_code == 206)
                    expected = None
                    if append:
                        actual_validator = (
                            response.headers.get("ETag") or response.headers.get("Last-Modified")
                        )
                        content_range = response.headers.get("Content-Range", "")
                        if actual_validator != validator or not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server did not confirm the requested file identity/range")
                        expected = int(content_range.rsplit("/", 1)[1])
                    elif response.status_code != 200:
                        raise RuntimeError(f"Unexpected download response {response.status_code}")
                    else:
                        offset = 0
                        if response.headers.get("Content-Length"):
                            expected = int(response.headers["Content-Length"])
                    write_json(metadata, {
                        "url": url, "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "expected_bytes": expected,
                    })
                    print(f"{'Resuming' if append else 'Downloading'} {name} at {offset:,} bytes", flush=True)
                    last = time.monotonic()
                    with partial.open("ab" if append else "wb") as stream:
                        for block in response.iter_content(BUFFER):
                            if block:
                                stream.write(block)
                                offset += len(block)
                            if time.monotonic() - last >= 8:
                                check_space(workspace)
                                print(f"  {name}: {offset / 1024**2:.1f} MiB", flush=True)
                                last = time.monotonic()
                    if expected is not None and partial.stat().st_size != expected:
                        raise requests.ConnectionError("Incomplete HTTP body; rerun to resume")
                checked = inspect_zip(partial)
                partial.replace(destination)
                metadata.unlink(missing_ok=True)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                print(f"Network interruption; retry {attempt + 1}/4", flush=True)
                time.sleep(min(2**attempt, 16))
    state[name] = {**checked, "url": url, "endpoint": endpoint, "manifest_records": partition["records"]}
    write_json(workspace / "download_state.json", state)
    print(f"Completed {name}: {checked['bytes']:,} bytes; ZIP verified", flush=True)


def main():
    """Pin sources, download sequentially, and verify baseline preservation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Download only this many files for the initial trial")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == REPO or workspace.is_relative_to(REPO) or "onedrive" in str(workspace).lower():
        parser.error("Use a separate workspace outside the repository and OneDrive")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "downloads").mkdir(exist_ok=True)
    lock = (workspace / "download.lock").open("a+b")
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_space(workspace)
    baseline_path = workspace / "baseline.json"
    if not baseline_path.exists():
        write_json(baseline_path, baseline_inventory())
    baseline = read_json(baseline_path)
    print(f"Preserved baseline: {baseline['drug_names']} names, {baseline['label_records']} records, "
          f"{baseline['passages']} passages", flush=True)
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "DrugAssist-source-download/0.1"
            manifest_path = workspace / "fda_manifest.json"
            if not manifest_path.exists():
                response = session.get(MANIFEST_URL, timeout=(15, 90))
                response.raise_for_status()
                response.json()  # Validate JSON before pinning its original bytes.
                temporary = manifest_path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(manifest_path)
            manifest = read_json(manifest_path)
            state_path = workspace / "download_state.json"
            state = read_json(state_path) if state_path.exists() else {}
            partitions = [
                (endpoint, partition)
                for endpoint in ("label", "drugsfda")
                for partition in manifest["results"]["drug"][endpoint]["partitions"]
            ]
            names = [validate_partition(part, endpoint) for endpoint, part in partitions]
            if len(set(names)) != len(names):
                raise ValueError("Manifest contains duplicate destination filenames")
            selected = partitions[:args.limit] if args.limit is not None else partitions
            for endpoint, partition in selected:
                download_partition(session, partition, endpoint, workspace, state)
            report = {
                "manifest_sha256": sha256(manifest_path),
                "exports": {
                    endpoint: manifest["results"]["drug"][endpoint]["export_date"]
                    for endpoint in ("label", "drugsfda")
                },
                "expected_files": len(partitions),
                "verified_files": len([name for name in names if name in state]),
                "complete": all(name in state for name in names),
                "compressed_bytes": sum(state[name]["bytes"] for name in names if name in state),
                "uncompressed_bytes": sum(
                    state[name]["uncompressed_bytes"] for name in names if name in state
                ),
                "source_records": {
                    endpoint: manifest["results"]["drug"][endpoint]["total_records"]
                    for endpoint in ("label", "drugsfda")
                },
                "embeddings_requested": 0,
                "index_built": False,
            }
            write_json(workspace / "download_report.json", report)
            print(json.dumps(report, indent=2), flush=True)
    finally:
        verification = verify_baseline(baseline)
        write_json(workspace / "baseline_verification.json", verification)
        print("Baseline verification: " + json.dumps(verification), flush=True)
        lock.close()
    if not verification["unchanged"]:
        raise RuntimeError("Baseline files changed during the run; see baseline_verification.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted safely. Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
