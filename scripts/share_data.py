"""Export or check the public bundle; implementation shared with first-run setup."""

from src.bundles import (  # noqa: F401 -- Preserve the script's importable engineering API.
    COMPATIBILITY,
    ROOT,
    SNAPSHOT_FILES,
    ZipFile,
    active_files,
    check_bundle,
    export_bundle,
    local_path,
    main,
    read_json,
    sha256,
)

if __name__ == "__main__":
    main()
