"""Fresh source acquisition must not require a bundled snapshot."""

from unittest.mock import Mock

import pytest

from src import maintenance
from src.utils.config import write_json


@pytest.mark.parametrize("stale_checkpoint", [False, True])
def test_acquire_without_local_snapshot(tmp_path, monkeypatch, stale_checkpoint):
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    if stale_checkpoint:
        write_json(tmp_path / "reports/data_checks.json", {"snapshot_id": "missing"})
    monkeypatch.setattr(maintenance, "load_snapshot", Mock(side_effect=FileNotFoundError))
    snapshot = {"manifest": {"snapshot_id": "new"}}
    ingest = Mock(return_value=snapshot)
    monkeypatch.setattr(maintenance, "ingest", ingest)
    monkeypatch.setattr(maintenance, "coverage_report", lambda *args: {
        "covered_names": 1, "requested_names": 1, "full_fda_label_sets": 1,
    })
    monkeypatch.setattr(maintenance, "quality_report", lambda *args: {})
    monkeypatch.setattr(maintenance, "verify_snapshot", lambda *args: {})
    maintenance.acquire({"ingestion": {"lookup_names": ["test"]}})
    assert ingest.call_args.kwargs["base_snapshot"] is None
