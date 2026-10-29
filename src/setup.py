"""Prepare an installation in isolation, validate it, and activate its index last."""

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import requests
from dotenv import dotenv_values

from src.bundles import COMPATIBILITY, check_bundle, local_path, sha256
from src.utils.config import CODE_ROOT, ROOT, load_config, read_json

POINTER = "results/search_latest.json"
RECEIPT = "results/private/bundle-checked.sha256"


def model_folder(root, settings):
    """Resolve the workspace path for a specific model and pinned revision."""
    return local_path(root, "data/models/models--" + settings["model"].replace("/", "--")
                      + "/snapshots/" + settings["revision"])


def check_models(root, settings):
    """Raise FileNotFoundError when required pinned model assets are incomplete."""
    folder = model_folder(root, settings)
    required = ("config.json", "tokenizer_config.json", "vocab.txt")
    if not all((folder / name).is_file() for name in required) or not any(
        (folder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError("Pinned reranker files are incomplete. Run setup to prepare model assets.")


def prepare_models(root, settings):
    """Download missing pinned model assets and verify their required files."""
    try:
        check_models(root, settings)
        return
    except FileNotFoundError:
        pass
    from huggingface_hub import snapshot_download

    snapshot_download(
        settings["model"], revision=settings["revision"], cache_dir=str(root / "data/models"),
        allow_patterns=["config.json", "model.safetensors", "tokenizer*.json", "special_tokens_map.json",
                        "vocab.txt"],
    )
    check_models(root, settings)


def check_installation(root, config, *, require_models=True):
    """Validate downloaded or locally built data against configuration and lineage."""
    from src.retrieval.search import SearchSession

    root = root.resolve()
    descriptor = local_path(root, read_json(root / POINTER)["run_path"])
    if not descriptor.is_relative_to(root / "results/search_indexes"):
        raise ValueError("Active index must be inside results/search_indexes")
    index = read_json(descriptor)
    selection = index["selection"]
    # Older prepared descriptors retain a historical research configuration. Their
    # selected parameters describe the actual indexed corpus and active retrievers.
    checks = [(config.get(name), selection.get(name)) for name in ("chunking", "reranker")]
    if "embedding" in config:
        checks.append((config["embedding"], selection["embedding"]["embedding"]))
    if "retrieval" in config:
        checks.extend([
            ({key: config["retrieval"][key] for key in ("k1", "b")}, selection["bm25"]),
            ({key: config["retrieval"][key] for key in ("fusion", "rrf_k", "alpha")}, selection["hybrid"]),
            (config["retrieval"]["candidates"], index["configuration"]["retrieval"]["candidates"]),
        ])
    if "chroma" in config:
        checks.append((config["chroma"], index["configuration"]["chroma"]))
    if "ingestion" in config:
        checks.append(
            (config["ingestion"]["lookup_names"], index["configuration"]["ingestion"]["lookup_names"])
        )
    if "fusion_settings" in config:
        checks.append((config["fusion_settings"], index["fusion_settings"]))
    if any(expected is not None and expected != actual for expected, actual in checks):
        raise ValueError("Installed index and current configuration differ; use a separate clean checkout.")
    check_bundle(root)
    # Check text identifiers and corpus lineage without invoking an embedding provider.
    SearchSession(run_path=descriptor, method="bm25")
    if require_models:
        check_models(root, index["selection"]["reranker"])


def distribution(root):
    """Return a validated release asset matching the current platform and Python."""
    spec = read_json(root / "config/distribution.json")
    if spec.get("format") != 1:
        raise ValueError("Unsupported distribution descriptor")
    platform_key = platform.system().lower() + "-" + platform.machine().lower()
    try:
        asset = spec["assets"][platform_key]
    except KeyError:
        raise ValueError(
            f"No tested bundle for {platform_key}. Run fda-rag setup --from-source instead."
        ) from None
    if asset.get("python") and asset["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError(
            f"The prepared bundle requires Python {asset['python']}; use --from-source otherwise."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", "")):
        raise ValueError("The bundle checksum is not configured. Run setup --from-source instead.")
    if not asset["url"].startswith(
        "https://github.com/suvrazastrovision/drug_rag/releases/download/"
    ) or "/latest/" in asset["url"]:
        raise ValueError("Bundle URL must identify a versioned project release")
    return asset


def download_bundle(asset, output):
    """Download a pinned bundle and reject a mismatched SHA-256 checksum."""
    print("Downloading the pinned prepared collection...", flush=True)
    try:
        with requests.get(asset["url"], stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with output.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
    except requests.RequestException:
        raise RuntimeError(
            "Bundle download failed. Retry, or explicitly run setup --from-source "
            "to build fresh data (embedding charges apply)."
        ) from None
    if sha256(output) != asset["sha256"]:
        raise ValueError("Downloaded bundle checksum mismatch; nothing was installed")


def public_member(name):
    """Only public stores and activation pointers may be supplied by an archive."""
    if "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"..", "."} for part in name.split("/")):
        return False
    return name in {POINTER, "data/bundle.json", "data/processed/latest.json"} or any(
        name.startswith(prefix) for prefix in (
            "data/raw/", "data/processed/", "data/chroma/", "data/models/", "results/search_indexes/",
        )
    )


def extract_bundle(archive, stage):
    """Extract allowed archive members and verify the complete manifest inventory."""
    try:
        with ZipFile(archive) as zipped:
            seen = set()
            for info in zipped.infolist():
                name = info.filename.rstrip("/")
                if not public_member(name) or name.casefold() in seen:
                    raise ValueError(f"Unsafe or duplicate archive path: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Archive symlinks are not supported")
                seen.add(name.casefold())
                target = local_path(stage, name)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
    except BadZipFile:
        raise ValueError("Downloaded bundle is not a valid ZIP archive") from None
    manifest = read_json(stage / "data/bundle.json")
    expected = set(manifest["files"]) | {"data/bundle.json"}
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if expected != actual or not all(public_member(name) for name in expected):
        raise ValueError("Archive members differ from the public bundle manifest")
    for name, checksum in manifest["files"].items():
        if sha256(local_path(stage, name)) != checksum:
            raise ValueError(f"Bundle member checksum mismatch: {name}")


@contextmanager
def setup_lock(root):
    """Hold an exclusive setup lock and remove it when the operation exits."""
    lock = root / "results/private/setup.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = lock.open("x")
    except FileExistsError:
        raise RuntimeError(
            "Setup is already running. If it was interrupted, remove "
            "results/private/setup.lock."
        ) from None
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def install_stage(stage, root, *, from_source=False):
    """Replace files transactionally; preserve unrelated files and activate last."""
    files = [p for p in stage.rglob("*") if p.is_file() and public_member(p.relative_to(stage).as_posix())]
    # This receipt is generated locally by check_bundle, never accepted from an
    # archive. Preserve it before the native smoke check's bookkeeping changes
    # would otherwise be mistaken for corruption on the next launch.
    receipt = stage / RECEIPT
    if receipt.is_file():
        if receipt.read_text() != sha256(stage / "data/bundle.json"):
            raise ValueError("Staged bundle validation receipt does not match its manifest")
        files.append(receipt)
    if from_source:
        files.append(stage / "reports/data_checks.json")
    files.sort(key=lambda p: (p.relative_to(stage).as_posix() == POINTER, p.as_posix()))
    changed = []
    backup_root = stage.parent / "rollback"
    try:
        for source in files:
            relative = source.relative_to(stage).as_posix()
            target = local_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = local_path(backup_root, relative)
            existed = target.exists()
            if existed:
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
            changed.append((target, backup, existed))
            shutil.copyfile(source, target)
    except BaseException:
        for target, backup, existed in reversed(changed):
            target.unlink(missing_ok=True)
            if existed:
                backup.replace(target)
        raise


def source_worker(config_path):
    """Run only in a child process whose workspace points to staging."""
    from src.maintenance import maintain

    config = load_config(config_path)
    prepare_models(ROOT, config["reranker"])
    maintain(config, "all")
    check_installation(ROOT, config)


def smoke_installation(root):
    """Load the actual native index/model in an offline process, without any query or API charge."""
    root = root.resolve()
    env = dict(os.environ, DRUGASSIST_WORKSPACE=str(root), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(root / "data/models"), HF_HUB_CACHE=str(root / "data/models"),
               TRANSFORMERS_CACHE=str(root / "data/models"))
    for name in ("OPENAI_API_KEY", "FDA_API_KEY"):
        env.pop(name, None)
    try:
        subprocess.run(
            [sys.executable, "-c", ("from src.services.search_service import build_search_service; "
             "service = build_search_service(); print(service.health()); service.close()")],
            cwd=root, env=env, check=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError(
            "Native index/model readiness check failed or timed out; setup did not "
            "activate a new installation."
        ) from None


def build_source(stage, config_path):
    """Prepare source data in a staged child process using configured API credentials."""
    env = dict(os.environ, DRUGASSIST_WORKSPACE=str(stage))
    for name, value in dotenv_values(CODE_ROOT / ".env").items():
        if name in {"OPENAI_API_KEY", "FDA_API_KEY"} and value and not env.get(name):
            env[name] = value
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env or the environment before building from source")
    # Setup downloads required assets; ordinary application startup can then stay offline.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", "from src.setup import source_worker; import sys; source_worker(sys.argv[1])",
         str(config_path)], env=env, cwd=CODE_ROOT, check=False,
    )
    if result.returncode:
        raise RuntimeError("Source preparation failed; the active installation was not changed")


def setup(*, from_source=False, config_path=None, root=ROOT):
    """Prepare or verify local data and models, activating a validated installation last.

    Reuse a compatible installation or download its pinned prepared bundle.
    With from_source=True, build a staged collection using configured API keys;
    embedding requests can incur charges. Use config_path to override the default
    configuration and root to select the installation workspace. Preserve the
    active installation if staging or validation fails.
    """
    root = Path(root).resolve()
    config = load_config(config_path)
    with setup_lock(root):
        has_data = (root / "data/chroma/chroma.sqlite3").is_file()
        if has_data:
            # Refuse to overwrite an incompatible installation, including private user state.
            check_installation(root, config, require_models=False)
            prepare_models(root, config["reranker"])
            smoke_installation(root)
            print("Existing compatible installation and model assets are ready.")
            return
        with tempfile.TemporaryDirectory(prefix=".drugassist-setup-", dir=root) as scratch:
            scratch = Path(scratch)
            stage = scratch / "stage"
            stage.mkdir()
            if from_source:
                print("Building fresh FDA data. OpenAI document embedding charges apply.", flush=True)
                for name in COMPATIBILITY:
                    target = local_path(stage, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / name, target)
                build_source(stage, Path(config_path or root / "config/default.json").resolve())
            else:
                asset = distribution(root)
                archive = scratch / "bundle.zip"
                download_bundle(asset, archive)
                extract_bundle(archive, stage)
                for name in COMPATIBILITY:
                    target = local_path(stage, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / name, target)
            check_installation(stage, config)
            smoke_installation(stage)
            install_stage(stage, root, from_source=from_source)
    print("DrugAssist is ready. Start the backend and frontend using the README commands.")
