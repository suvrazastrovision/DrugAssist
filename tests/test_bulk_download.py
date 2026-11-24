"""Recovery and integrity checks for the source-only FDA downloader."""

from io import BytesIO
import zipfile

import pytest
import requests

from scripts import download_fda_bulk as bulk


def zip_body():
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labels.json", '{"results":[{"set_id":"one"}]}')
    return output.getvalue()


class Response:
    def __init__(self, status, headers, body, interrupt=False):
        self.status_code, self.headers, self.body = status, headers, body
        self.interrupt = interrupt

    def __enter__(self):
        """Open a simulated HTTP response."""
        return self

    def __exit__(self, *args):
        """Close a simulated HTTP response."""
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body
        if self.interrupt:
            raise requests.ConnectionError("interrupted")


@pytest.mark.parametrize("resumable", [True, False])
def test_interrupted_download_resumes_or_restarts_without_duplicate_bytes(tmp_path, monkeypatch, resumable):
    body = zip_body()
    cut = len(body) // 2
    monkeypatch.setattr(bulk, "check_space", lambda path: None)
    monkeypatch.setattr(bulk.time, "sleep", lambda delay: None)
    (tmp_path / "downloads").mkdir()
    partition = {"file": "https://download.open.fda.gov/drug/label/sample.json.zip", "records": 1}
    headers = {"Content-Length": str(len(body))}
    if resumable:
        headers["ETag"] = '"snapshot-a"'

    class Session:
        calls = 0

        def get(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response(200, headers, body[:cut], interrupt=True)
            if resumable:
                assert kwargs["headers"]["Range"] == f"bytes={cut}-"
                assert kwargs["headers"]["If-Range"] == '"snapshot-a"'
                return Response(206, {
                    "ETag": '"snapshot-a"', "Content-Range": f"bytes {cut}-{len(body)-1}/{len(body)}",
                }, body[cut:])
            assert "Range" not in kwargs["headers"]
            return Response(200, headers, body)

    session, state = Session(), {}
    bulk.download_partition(session, partition, "label", tmp_path, state)
    destination = tmp_path / "downloads/sample.json.zip"
    assert destination.read_bytes() == body
    assert state[destination.name]["zip_crc_verified"]
    bulk.download_partition(session, partition, "label", tmp_path, state)
    assert session.calls == 2  # A completed file is verified locally, never requested again.


def test_completed_partial_recovers_without_http(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk, "check_space", lambda path: None)
    (tmp_path / "downloads").mkdir()
    body = zip_body()
    url = "https://download.open.fda.gov/drug/label/sample.json.zip"
    (tmp_path / "downloads/sample.json.zip.part").write_bytes(body)
    bulk.write_json(tmp_path / "downloads/sample.json.zip.part.json", {
        "url": url, "expected_bytes": len(body),
    })
    bulk.download_partition(None, {"file": url, "records": 1}, "label", tmp_path, {})
    assert (tmp_path / "downloads/sample.json.zip").read_bytes() == body


def test_corrupted_completed_file_is_not_trusted(tmp_path):
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads/sample.json.zip").write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        bulk.download_partition(None, {
            "file": "https://download.open.fda.gov/drug/label/sample.json.zip", "records": 1,
        }, "label", tmp_path, {})


def test_low_disk_space_stops_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 0})())
    with pytest.raises(RuntimeError, match="80 GiB"):
        bulk.check_space(tmp_path)


def test_manifest_cannot_redirect_download_outside_fda():
    with pytest.raises(ValueError, match="Unexpected FDA"):
        bulk.validate_partition({"file": "https://example.com/drug/label/sample.json.zip"}, "label")
