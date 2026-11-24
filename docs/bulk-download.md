# Local FDA source downloads

This stage saves FDA source ZIPs locally. It does not change the 60-drug app,
normalize records, generate passages, request embeddings, or build an index.
The existing chunking and retrieval configuration remains in place.

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe scripts/download_fda_bulk.py --workspace "$env:LOCALAPPDATA\DrugAssist\bulk"
```

For a first-partition trial, append `--limit 1`. Run again without that option
to finish. The downloader reuses the saved manifest rather than selecting a
different FDA release midway through a run.

## Files on your PC

The default destination above is outside OneDrive and the repository:

- `fda_manifest.json`: the original pinned FDA download manifest.
- `downloads/`: label and Drugs@FDA approval ZIPs, with no bulk extraction.
- `download_state.json`: per-file hashes, sizes and ZIP-validation results.
- `download_report.json`: totals, export dates and completion status.
- `baseline.json`: original baseline configuration, counts and file hashes.
- `baseline_verification.json`: comparison of protected baseline files after a run.
- `download.lock`: operating-system lock preventing concurrent writers; the
  file remains after exit, but its lock is automatically released.

The source-record totals are before approval filtering. Uncompressed sizes
come from ZIP member metadata; these files are not extracted to disk.

## Interrupting and resuming

Press Ctrl+C, then rerun the same command. Partial files use `.part` names.
Byte-range resume requires an unchanged ETag or Last-Modified value and a valid
server range response. If the server ignores the range or does not provide a
validator, the partition restarts. Completed files are verified and skipped.
Network failures are retried; ZIP-integrity or baseline changes stop the run
for investigation. SHA-256 hashes record locally downloaded bytes; they are
not signatures or FDA-published cryptographic checksums.

The process pauses if the destination drive has under 80 GiB free or the bulk
workspace reaches 40 GiB. Keep this workspace dedicated to this download run.
Use a new workspace when deliberately acquiring a newer release.

Avoid running indexing or maintenance against the baseline during the download.
The baseline inventory includes Chroma and embedding-cache files, so another
process updating them can cause preservation verification to report changes.

Approval matching, sequential normalization, token sizing and sample indexing
are subsequent stages and are not implemented by this command.
