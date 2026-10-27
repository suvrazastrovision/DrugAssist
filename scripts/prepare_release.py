"""Export, restore-check and pin a release asset locally; this does not publish it."""

import argparse
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from src.bundles import COMPATIBILITY, export_bundle, sha256
from src.setup import check_installation, extract_bundle, smoke_installation
from src.utils.config import ROOT, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Immutable GitHub Release tag, e.g. data-v0.1.0")
    args = parser.parse_args()
    if not all(char.isalnum() or char in "-_." for char in args.tag):
        parser.error("Use a simple version tag containing letters, digits, hyphens, dots or underscores")
    platform_key = platform.system().lower() + "-" + platform.machine().lower()
    asset_name = f"drugassist-data-{platform_key}-py{sys.version_info.major}{sys.version_info.minor}.zip"
    output = ROOT / "results/private/release" / asset_name
    export_bundle(ROOT, output)
    with tempfile.TemporaryDirectory(prefix="drugassist-release-check-") as scratch:
        stage = Path(scratch).resolve()
        extract_bundle(output, stage)
        for name in COMPATIBILITY:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        check_installation(stage, load_config())
        smoke_installation(stage)
    descriptor = {"format": 1, "assets": {platform_key: {
        "url": f"https://github.com/suvrazastrovision/drug_rag/releases/download/{args.tag}/{asset_name}",
        "sha256": sha256(output),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }}}
    write_json(ROOT / "config/distribution.json", descriptor)
    print("Restoration verified. Upload this exact file to the pinned release:", output)
    print("Commit config/distribution.json with the matching code. This script did not publish anything.")


if __name__ == "__main__":
    main()
