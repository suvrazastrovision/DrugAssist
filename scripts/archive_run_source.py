"""Archive exact experiment code/configuration without data caches or credentials."""

import json
import zipfile
from pathlib import Path

from src.utils.config import digest

ROOT = Path(__file__).resolve().parents[1]
pointer = json.loads((ROOT / "results/latest.json").read_text(encoding="utf-8"))
run_path = ROOT / pointer["run_path"]
run = json.loads(run_path.read_text(encoding="utf-8"))
for name, expected in run["source_files"].items():
    if digest((ROOT / name).read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Experiment source changed before archival: " + name)
with zipfile.ZipFile(run_path.parent / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    files = [ROOT / name for name in run["source_files"]]
    files.extend(p for p in (ROOT / "config").glob("*.json"))
    files.append(ROOT / "data/benchmarks/query_specs.json")
    files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "README.md", "ANNOTATION_PROTOCOL.md"))
    for path in files:
        archive.write(path, arcname=str(path.relative_to(ROOT)))
print("Archived exact source and configuration for", run["run_id"])
