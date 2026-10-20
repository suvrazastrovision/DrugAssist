"""Maintain an append-only corpus; a new benchmark binding is required afterward."""

import argparse

from src.ingestion.data_loader import ingest, load_snapshot
from src.ingestion.incremental import merge_snapshots
from src.utils.config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", help="Base snapshot ID; defaults to current latest before acquisition")
parser.add_argument("--config")
args = parser.parse_args()
base = load_snapshot(args.base)
update = ingest(load_config(args.config))
merged = merge_snapshots(base, update, set_latest=True)
print("Incremental snapshot", merged["manifest"]["snapshot_id"])
print("Historical versions retained; rebind and independently review benchmark evidence before evaluation")
