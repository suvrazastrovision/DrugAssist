"""First-run installs use staged public data and never a paid implicit fallback."""

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
import requests
from test_share_data import prepared as bundle_data  # noqa: F401 -- Register the shared synthetic fixture.

from src import bundles
from src import setup as installer
from src.setup import smoke_installation
from src.utils.config import write_json


@pytest.fixture
def release(request, tmp_path, monkeypatch):
    prepared = request.getfixturevalue("bundle_data")
    pointer = bundles.read_json(prepared / installer.POINTER)
    descriptor = bundles.local_path(prepared, pointer["run_path"])
    index = bundles.read_json(descriptor)
    config = {"reranker": index["selection"]["reranker"]}
    index["configuration"] = config
    write_json(descriptor, index)
    write_json(prepared / "config/default.json", config)
    folder = installer.model_folder(prepared, config["reranker"])
    (folder / "tokenizer_config.json").write_text("{}")
    (folder / "vocab.txt").write_text("test")
    archive = tmp_path / "release.zip"
    bundles.export_bundle(prepared, archive)
    recipient = tmp_path / "recipient"
    recipient.mkdir()
    for name in bundles.COMPATIBILITY:
        target = recipient / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((prepared / name).read_bytes())
    # The real corpus identifier validation is covered by SearchSession tests; this fixture
    # intentionally contains only a minimal SQLite schema instead of a real vector index.
    monkeypatch.setattr("src.retrieval.search.SearchSession", Mock())
    # Placeholder model files cannot be loaded; native subprocess behavior is
    # exercised separately below rather than using real downloaded weights.
    monkeypatch.setattr(installer, "smoke_installation", Mock())
    asset = {"url": "https://github.com/suvrazastrovision/drug_rag/releases/download/data-v0.1.0/data.zip",
             "sha256": bundles.sha256(archive)}
    key = installer.platform.system().lower() + "-" + installer.platform.machine().lower()
    write_json(recipient / "config/distribution.json", {"format": 1, "assets": {key: asset}})
    return recipient, archive, config


def run_setup(root, **kwargs):
    installer.setup(root=root, config_path=root / "config/default.json", **kwargs)


def test_bundle_setup_installs_and_then_reuses_without_provider_calls(release, monkeypatch):
    root, archive, _ = release
    download = Mock(side_effect=lambda asset, output: output.write_bytes(archive.read_bytes()))
    monkeypatch.setattr(installer, "download_bundle", download)
    monkeypatch.setattr(installer, "build_source", Mock(side_effect=AssertionError("Unexpected paid build")))
    run_setup(root)
    assert (root / "data/chroma/chroma.sqlite3").is_file()
    assert not (root / "data/auth").exists()
    assert not (root / ".env").exists()
    run_setup(root)
    assert download.call_count == 1
    assert installer.smoke_installation.call_count == 2


def test_native_bookkeeping_and_later_launch_preserve_bundle_validation(release, monkeypatch):
    root, archive, _ = release
    monkeypatch.setattr(
        installer, "download_bundle", lambda asset, output: output.write_bytes(archive.read_bytes())
    )

    def smoke(stage):
        assert (stage / installer.RECEIPT).read_text() == bundles.sha256(stage / "data/bundle.json")
        with closing(sqlite3.connect(stage / "data/chroma/chroma.sqlite3")) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS bookkeeping(value TEXT)")

    monkeypatch.setattr(installer, "smoke_installation", smoke)
    run_setup(root)
    assert (root / installer.RECEIPT).read_text() == bundles.sha256(root / "data/bundle.json")
    run_setup(root)
    (root / "data/processed/snapshot/documents.json").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_setup(root)


def test_download_failure_never_builds_or_overwrites_private_files(release, monkeypatch):
    root, _, _ = release
    private = root / "data/auth/users.sqlite3"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private accounts")
    monkeypatch.setattr(installer, "download_bundle", Mock(side_effect=RuntimeError("offline")))
    build = Mock()
    monkeypatch.setattr(installer, "build_source", build)
    with pytest.raises(RuntimeError, match="offline"):
        run_setup(root)
    build.assert_not_called()
    assert private.read_bytes() == b"private accounts"
    assert not (root / installer.POINTER).exists()


def test_source_setup_uses_stage_and_leaves_published_report_alone(release, monkeypatch):
    root, archive, _ = release
    published = root / "reports/expanded_60/method_comparison.json"
    published.parent.mkdir(parents=True)
    published.write_bytes(b"published benchmark")

    def build(stage, config_path):
        # Source builds have no bundle receipt; the shared readiness check still applies.
        with ZipFile(archive) as zipped:
            zipped.extractall(stage)
        (stage / "data/bundle.json").unlink()
        write_json(stage / "reports/data_checks.json", {"snapshot_id": "snapshot"})

    monkeypatch.setattr(installer, "build_source", build)
    monkeypatch.setattr(installer, "download_bundle", Mock(side_effect=AssertionError("Unexpected download")))
    run_setup(root, from_source=True)
    assert (root / "data/chroma/chroma.sqlite3").is_file()
    assert not (root / "data/bundle.json").exists()
    assert published.read_bytes() == b"published benchmark"
    assert bundles.read_json(root / "reports/data_checks.json")["snapshot_id"] == "snapshot"


def test_failed_source_build_preserves_existing_pointer(release, monkeypatch):
    root, _, _ = release
    write_json(root / installer.POINTER, {"run_path": "old"})
    monkeypatch.setattr(installer, "build_source", Mock(side_effect=RuntimeError("build failed")))
    with pytest.raises(RuntimeError, match="build failed"):
        run_setup(root, from_source=True)
    assert bundles.read_json(root / installer.POINTER)["run_path"] == "old"


def test_incompatible_existing_installation_is_not_replaced(release, monkeypatch):
    root, archive, config = release
    monkeypatch.setattr(
        installer, "download_bundle", lambda asset, output: output.write_bytes(archive.read_bytes())
    )
    run_setup(root)
    config["reranker"]["revision"] = "different"
    write_json(root / "config/default.json", config)
    download = Mock()
    monkeypatch.setattr(installer, "download_bundle", download)
    with pytest.raises(ValueError, match="configuration differ"):
        run_setup(root)
    download.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "data/auth/users.sqlite3",
        installer.RECEIPT,
        ".env",
        "C:/escape",
        "data/chroma/../../auth",
        "data\\chroma\\file",
    ],
)
def test_unsafe_archive_members_are_rejected(tmp_path, name):
    if "\\" in name:
        # Windows' ZIP writer normalizes backslashes, so test the path policy directly.
        assert not installer.public_member(name)
        return
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(name, b"bad")
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        installer.extract_bundle(archive, stage)


def test_corrupt_member_is_rejected_before_installation(release, tmp_path):
    _, archive, _ = release
    changed = tmp_path / "changed.zip"
    with ZipFile(archive) as original, ZipFile(changed, "w") as zipped:
        for name in original.namelist():
            zipped.writestr(name, b"modified" if name.endswith("chunks.json") else original.read(name))
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.extract_bundle(changed, stage)


def test_platform_mismatch_is_actionable(release, monkeypatch):
    root, _, _ = release
    monkeypatch.setattr(installer.platform, "system", lambda: "Unsupported")
    with pytest.raises(ValueError, match="--from-source"):
        installer.distribution(root)


def test_missing_model_asset_is_detected(release, monkeypatch):
    root, archive, config = release
    monkeypatch.setattr(
        installer, "download_bundle", lambda asset, output: output.write_bytes(archive.read_bytes())
    )
    run_setup(root)
    (installer.model_folder(root, config["reranker"]) / "vocab.txt").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        installer.check_models(root, config["reranker"])


def test_noncanonical_root_is_resolved_before_validation(release, monkeypatch):
    root, archive, config = release
    monkeypatch.setattr(
        installer, "download_bundle", lambda asset, output: output.write_bytes(archive.read_bytes())
    )
    run_setup(root)
    (root / "existing").mkdir()
    installer.check_installation(root / "existing/..", config)


def test_archive_symlinks_are_rejected(tmp_path):
    from zipfile import ZipInfo

    info = ZipInfo("data/models/link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    archive = tmp_path / "symlink.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(info, "../../.env")
    with pytest.raises(ValueError, match="symlinks"):
        installer.extract_bundle(archive, tmp_path / "stage")


def test_archive_checksum_and_interrupted_download(tmp_path, monkeypatch):
    response = SimpleNamespace(
        raise_for_status=lambda: None, iter_content=lambda **kwargs: iter([b"corrupt"]),
    )
    from contextlib import nullcontext

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: nullcontext(response))
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.download_bundle({"url": "https://example.invalid", "sha256": "0" * 64}, tmp_path / "zip")
    monkeypatch.setattr(requests, "get", Mock(side_effect=requests.Timeout))
    with pytest.raises(RuntimeError, match="explicitly"):
        installer.download_bundle({"url": "https://example.invalid"}, tmp_path / "zip")


def test_install_failure_rolls_back_and_activates_pointer_last(tmp_path, monkeypatch):
    root, stage = tmp_path / "root", tmp_path / "stage"
    for folder in (root, stage):
        write_json(folder / installer.POINTER, {"run_path": str(folder)})
        write_json(folder / "data/processed/latest.json", {"snapshot_id": str(folder)})
        write_json(folder / "data/bundle.json", {"stage": str(folder)})
        receipt = folder / installer.RECEIPT
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(bundles.sha256(folder / "data/bundle.json"))
    before = (root / installer.POINTER).read_bytes()
    before_receipt = (root / installer.RECEIPT).read_bytes()
    original = installer.shutil.copyfile
    order = []

    def fail_at_pointer(source, target):
        order.append(Path(source).relative_to(stage).as_posix())
        if order[-1] == installer.POINTER:
            raise OSError("interrupted")
        return original(source, target)

    monkeypatch.setattr(installer.shutil, "copyfile", fail_at_pointer)
    with pytest.raises(OSError, match="interrupted"):
        installer.install_stage(stage, root)
    assert order[-1] == installer.POINTER
    assert (root / installer.POINTER).read_bytes() == before
    assert installer.RECEIPT in order
    assert (root / installer.RECEIPT).read_bytes() == before_receipt
    assert json.loads((root / "data/processed/latest.json").read_text())["snapshot_id"] == str(root)


@pytest.mark.parametrize("from_source", [False, True])
@pytest.mark.parametrize("failure", [subprocess.CalledProcessError(1, "native-check"),
                                   subprocess.TimeoutExpired("native-check", 120)])
def test_native_failure_never_activates_staged_installation(release, monkeypatch, from_source, failure):
    root, archive, _ = release
    write_json(root / installer.POINTER, {"run_path": "previous-index"})
    before = (root / installer.POINTER).read_bytes()
    private = root / "data/auth/users.sqlite3"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private accounts")
    monkeypatch.setattr(
        installer, "download_bundle", lambda asset, output: output.write_bytes(archive.read_bytes())
    )

    def build(stage, config_path):
        with ZipFile(archive) as zipped:
            zipped.extractall(stage)
        (stage / "data/bundle.json").unlink()
        write_json(stage / "reports/data_checks.json", {"snapshot_id": "snapshot"})

    monkeypatch.setattr(installer, "build_source", build)
    monkeypatch.setattr(installer, "smoke_installation", smoke_installation)
    monkeypatch.setattr(installer.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match="readiness check failed or timed out"):
        run_setup(root, from_source=from_source)
    assert (root / installer.POINTER).read_bytes() == before
    assert private.read_bytes() == b"private accounts"
    assert not (root / "data/chroma").exists()
    assert not (root / "results/private/setup.lock").exists()


def test_native_smoke_uses_offline_workspace_without_provider_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("FDA_API_KEY", "private-test-key")
    run = Mock()
    monkeypatch.setattr(installer.subprocess, "run", run)
    smoke_installation(tmp_path)
    options = run.call_args.kwargs
    assert options["cwd"] == tmp_path.resolve()
    assert options["check"] is True and options["timeout"] == 120
    assert options["env"]["DRUGASSIST_WORKSPACE"] == str(tmp_path.resolve())
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in options["env"] and "FDA_API_KEY" not in options["env"]


def test_mismatched_staged_receipt_is_rejected_before_installation(tmp_path):
    stage, root = tmp_path / "stage", tmp_path / "root"
    write_json(stage / "data/bundle.json", {})
    receipt = stage / installer.RECEIPT
    receipt.parent.mkdir(parents=True)
    receipt.write_text("invalid")
    with pytest.raises(ValueError, match="receipt does not match"):
        installer.install_stage(stage, root)
    assert not root.exists()
