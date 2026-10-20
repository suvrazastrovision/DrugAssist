"""Execute with the project interpreter, not an unrelated global Jupyter kernel."""

import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", action="append", help="Optional notebook stem; repeat to select several")
args = parser.parse_args()
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    if args.notebook and path.stem not in args.notebook:
        continue
    nb = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        nb, km=manager, timeout=300, resources={"metadata": {"path": str(ROOT)}}, allow_errors=False
    )
    client.execute()
    nbformat.write(nb, path)
    print("Executed", path.name, flush=True)
