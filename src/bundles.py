"""Export public FDA data and model assets, or check an extracted bundle. No API calls."""

import argparse
import hashlib
import json
import platform
import sqlite3
import sys
import tempfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY = ("config/default.json", "pyproject.toml", "uv.lock")
SNAPSHOT_FILES = ("documents.json", "records.json", "manifest.json", "verification.json", "quarantine.json")


def read_json(path):
    """Read a UTF-8 JSON document from a Path object."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    """Return the SHA-256 checksum of a file's bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compatibility_hash(path):
    """Git may check out text with CRLF on Windows and LF on other systems."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def local_path(root, relative):
    """Resolve a relative data path and reject paths outside the repository root."""
    path = root / relative.replace("\\", "/")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Data path leaves the repository")
    return path.resolve()


def active_files(root):
    """Use the active descriptor, not historical snapshots or private runtime stores."""
    root = root.resolve()
    pointer = root / "results/search_latest.json"
    descriptor = local_path(root, read_json(pointer)["run_path"])
    if not descriptor.is_relative_to(root / "results/search_indexes"):
        raise ValueError("Active descriptor must be inside results/search_indexes")
    index = read_json(descriptor)
    snapshot = local_path(root, "data/processed/" + index["snapshot_id"])
    if snapshot.parent != root / "data/processed":
        raise ValueError("Invalid snapshot identifier")
    files = {p.relative_to(root).as_posix(): p for p in (
        pointer, descriptor, descriptor.parent / "chunks.json",
        *(snapshot / name for name in SNAPSHOT_FILES),
    )}
    # Include the original public response bodies referenced by this snapshot.
    for response in read_json(snapshot / "manifest.json")["responses"]:
        path = local_path(root, response["raw_file"])
        if not path.is_relative_to(root / "data/raw") or path.suffix != ".body":
            raise ValueError("Unexpected raw response path")
        if sha256(path) != response["body_sha256"]:
            raise ValueError("Raw response checksum mismatch")
        files[path.relative_to(root).as_posix()] = path
    model = index["selection"]["reranker"]
    repository = "models--" + model["model"].replace("/", "--")
    folder = local_path(root, f"data/models/{repository}/snapshots/{model['revision']}")
    if not any((folder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError("Download the pinned reranker weights before exporting")
    for path in folder.rglob("*"):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    config_target = (folder / "config.json").relative_to(root).as_posix()
    if config_target not in files:
        # CrossEncoder 3.0.1 can cache AutoConfig separately from model weights.
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model["model"], "config.json", revision=model["revision"])
        if not isinstance(cached, str):
            raise FileNotFoundError("Pinned reranker config.json is not cached; load the model first")
        files[config_target] = Path(cached)
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required bundle file is missing: {path.name}")
    return files, index


def export_bundle(root, output):
    """Export active public data and model assets with a checksummed manifest."""
    root = root.resolve()
    files, index = active_files(root)
    database = root / "data/chroma/chroma.sqlite3"
    for path in database.parent.rglob("*"):
        if path.is_file() and path.suffix in {".bin", ".pickle"}:
            files[path.relative_to(root).as_posix()] = path
    signatures = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        backup = Path(scratch) / "chroma.sqlite3"
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as source:
            database_version = source.execute("PRAGMA data_version").fetchone()[0]
            with closing(sqlite3.connect(backup)) as target:
                source.backup(target)
            files["data/chroma/chroma.sqlite3"] = backup
            temporary = output.with_suffix(output.suffix + ".tmp")
            try:
                manifest = {
                    "format": 1,
                    "runtime": {
                        "platform": platform.system().lower() + "-" + platform.machine().lower(),
                        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "chromadb": version("chromadb"),
                    },
                    "snapshot_id": index["snapshot_id"],
                    "collection": index["selection"]["embedding"]["collection_name"],
                    "compatibility_encoding": "utf8-lf",
                    "compatibility": {name: compatibility_hash(root / name) for name in COMPATIBILITY},
                    "files": {},
                }
                with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=1) as archive:
                    for name, path in sorted(files.items()):
                        manifest["files"][name] = sha256(path)
                        archive.write(path, name)
                    latest = json.dumps({"snapshot_id": index["snapshot_id"]}).encode()
                    manifest["files"]["data/processed/latest.json"] = hashlib.sha256(latest).hexdigest()
                    archive.writestr("data/processed/latest.json", latest)
                    archive.writestr("data/bundle.json", json.dumps(manifest, indent=2))
                if source.execute("PRAGMA data_version").fetchone()[0] != database_version:
                    raise RuntimeError("Chroma changed during export; stop writers and retry")
                if any((files[n].stat().st_size, files[n].stat().st_mtime_ns) != sig
                       for n, sig in signatures.items()):
                    raise RuntimeError("Source files changed during export; stop writers and retry")
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"Created {output.name} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("Excluded: accounts, credentials, embedding/query cache, environment and history reports.")


def check_bundle(root):
    """Verify bundle compatibility, file checksums, and the active passage count."""
    root = root.resolve()
    manifest_path = root / "data/bundle.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("format") != 1:
            raise ValueError("Unsupported data bundle format")
        runtime = manifest.get("runtime")
        if runtime and (
            runtime["platform"] != platform.system().lower() + "-" + platform.machine().lower()
            or runtime["python"] != f"{sys.version_info.major}.{sys.version_info.minor}"
            or runtime["chromadb"] != version("chromadb")
        ):
            raise ValueError("Bundle runtime differs. Use setup --from-source in a clean checkout.")
        for name in COMPATIBILITY:
            checksum = compatibility_hash if manifest.get("compatibility_encoding") == "utf8-lf" else sha256
            if checksum(root / name) != manifest["compatibility"][name]:
                raise ValueError(
                    f"Bundle and repository differ: {name}. Use the matching repository version."
                )
        receipt = root / "results/private/bundle-checked.sha256"
        checked = receipt.exists() and receipt.read_text() == sha256(manifest_path)
        for name, expected in manifest["files"].items():
            # Chroma can update bookkeeping/index files during normal use.
            # Verify their original checksums once; check live record counts below on every launch.
            if checked and name.startswith("data/chroma/"):
                continue
            if sha256(local_path(root, name)) != expected:
                raise ValueError(f"Bundle checksum mismatch: {name}. Re-extract the original bundle.")
    descriptor = local_path(root, read_json(root / "results/search_latest.json")["run_path"])
    index = read_json(descriptor)
    folder = local_path(root, "data/processed/" + index["snapshot_id"])
    chunks = read_json(descriptor.parent / "chunks.json")
    if not (folder / "documents.json").is_file():
        raise FileNotFoundError("Saved FDA sections are missing; extract the prepared data bundle")
    database = root / "data/chroma/chroma.sqlite3"
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        count = db.execute(
            "SELECT count(*) FROM embeddings e JOIN segments s ON e.segment_id=s.id "
            "JOIN collections c ON s.collection=c.id WHERE c.name=? AND s.scope='METADATA'",
            (index["selection"]["embedding"]["collection_name"],),
        ).fetchone()[0]
    if count != len(chunks):
        raise ValueError("Chroma record count differs from the active passage inventory")
    if manifest_path.exists():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(sha256(manifest_path))
    print(f"Prepared collection found: {count} passages. No downloading or indexing needed.")


def main():
    """Dispatch bundle export or validation from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local data without exporting")
    parser.add_argument("--output", type=Path, default=ROOT / "drugassist-data.zip")
    args = parser.parse_args()
    try:
        if args.check:
            check_bundle(ROOT)
        else:
            export_bundle(ROOT, args.output)
    except (OSError, ValueError, KeyError, sqlite3.Error, RuntimeError) as exc:
        parser.exit(1, f"Data bundle: {exc}\n")


if __name__ == "__main__":
    main()
