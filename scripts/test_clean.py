"""Run tests twice in fresh source copies with no prepared data, credentials or caches."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    args, pytest_args = parser.parse_known_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    # Include new implementation files during local development, as well as tracked files.
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    ).decode().split("\0")
    for run in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory(prefix="drugassist-clean-") as scratch:
            root = Path(scratch)
            for name in paths:
                path = Path(name)
                if not name or path.parts[0] not in {
                    "src", "tests", "scripts", "frontend", "config", "pyproject.toml", "uv.lock",
                }:
                    continue
                source = ROOT / path
                if source.is_file():
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            env = dict(os.environ)
            for name in ("OPENAI_API_KEY", "FDA_API_KEY", "DRUGASSIST_WORKSPACE", "DRUGASSIST_AUTH_DB"):
                env.pop(name, None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            print(f"Clean offline test run {run}/{args.runs}", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_args],
                cwd=root, env=env, check=False,
            )
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
