"""Filesystem isolation allows temporary stores and rejects runtime access."""

import os
import sqlite3
import tempfile
from pathlib import Path

import conftest as guards
import pytest


def test_temporary_directory_cleanup_allows_runtime_folder_names(tmp_path):
    # Linux rmtree opens these child names with dir_fd. Resolving them against
    # the repository cwd instead would incorrectly reject safe cleanup.
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        for name in ("data", "results", "reports"):
            folder = Path(scratch) / name
            folder.mkdir()
            (folder / "fixture.txt").write_text("temporary")
    assert not Path(scratch).exists()


@pytest.mark.parametrize("operation", [
    lambda path: path.read_text(),
    lambda path: os.open(path, os.O_RDONLY),
    lambda path: path.mkdir(),
    lambda path: os.listdir(path),
    lambda path: os.scandir(path),
    lambda path: sqlite3.connect(path),
])
def test_runtime_paths_are_still_blocked(tmp_path, monkeypatch, operation):
    protected = tmp_path / "private-runtime"
    monkeypatch.setattr(guards, "_protected", [protected])
    with pytest.raises(AssertionError, match="Tests must use temporary stores"):
        operation(protected)


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="Requires directory-relative open")
def test_directory_relative_access_uses_descriptor_instead_of_cwd(tmp_path, monkeypatch):
    protected = tmp_path / "private-runtime"
    protected.mkdir()
    (protected / "fixture.txt").write_text("private")
    descriptor = os.open(protected, os.O_RDONLY)
    try:
        monkeypatch.setattr(guards, "_protected", [protected])
        with pytest.raises(AssertionError, match="Tests must use temporary stores"):
            os.open("fixture.txt", os.O_RDONLY, dir_fd=descriptor)
        with pytest.raises(AssertionError, match="Tests must use temporary stores"):
            os.mkdir("child", dir_fd=descriptor)
    finally:
        os.close(descriptor)
