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

## Free count-only sizing

The separate analyzer scans the complete download, selects the newest
approval-matched record per label set, and counts the existing normalizer and
chunker's actual embedding inputs. It does not save passage text or vectors,
change the active app, or make network connections. It uses the checked-in
`cl100k_base` tokenizer asset, so it needs no API credentials or new dependencies.

First try matching one partition:

```powershell
.venv/Scripts/python.exe scripts/analyze_fda_bulk.py --workspace "$env:LOCALAPPDATA\DrugAssist\bulk" --stage match --limit-partitions 1
```

Then resume matching and count the full selected collection:

```powershell
.venv/Scripts/python.exe scripts/analyze_fda_bulk.py --workspace "$env:LOCALAPPDATA\DrugAssist\bulk"
```

Run the same command after Ctrl+C to resume. Reading a compressed partition
starts at its beginning again, but committed records and label-set counts are
skipped without duplication. Source hashes, code and configuration must match
the saved run. There is no sampling or drug-name limit in a complete report.
Only label records with an explicit NDA/ANDA/BLA application match to at least
one approved (`AP`) Drugs@FDA submission qualify. Tentative-only applications
are excluded. This is application-level approval evidence, not verification of
current marketing status or independent approval of each label's text.

Version selection sorts numeric version, effective date, then canonical source
hash. Labels without a set ID are reported but cannot be selected. Exact
duplicates and superseded records remain represented in the audit database.
The original ZIPs preserve all source records.

Find the outputs in `%LOCALAPPDATA%\DrugAssist\bulk\analysis`:

- `sizing_report.txt` and `sizing_report.json`: counts, token totals, estimated
  serialized passage sizes, raw vector/cache payload, and storage scenarios.
- `analysis.sqlite3`: approvals, matching audit, selected record locations,
  per-label counts, unique input hashes and transactional checkpoints.
- `baseline_comparison.json`: whether original full-label records satisfy the
  same approval rule and overlap the downloaded/selected collection.
- `baseline_verification.json`: protected original files remain unchanged.
- `progress.json`: the most recent progress update.

Use `--stage status` to read the latest persisted report. During a run,
`progress.json` has more recent progress than the last partition-level report.
Use `--stage match` or `--stage size` to run the stages independently; sizing
requires matching to have finished across every partition first.

Analysis has a 2 GiB **additional** storage allowance, an 80 GiB free-space
reserve, and a 4 GiB process peak-RAM stop threshold. The SQLite database is
limited to approximately 1 GiB to leave space for its rollback journal and
reports. A resource stop rolls back uncommitted work and saves a paused report.
The existing ZIP downloads do not count toward the new 2 GiB allowance.

Passage/normalized JSONL sizes describe canonical compact UTF-8 serialization
with one newline per record; those files are counted, not written. Cache totals
deduplicate the exact identity/section/text input, using SHA-256 hashes. Token
counts include repeated passage inputs as well as distinct-input totals.
The source SHA in the hypothetical passage provenance refers to the source ZIP.

Raw vectors use the configured dimension count and four bytes per float32.
Storage scenarios apply explicitly hypothetical Chroma overhead multipliers;
they are not measured storage forecasts or upper bounds. This report cannot
certify cloud RAM or query latency. A separately cost-bounded representative
index benchmark is required before selecting the cheapest server under the
EUR 50/month hosting ceiling. No paid benchmark or rental is initiated here.

After complete sizing, prepare nested samples targeting 2,000 and 10,000 passages:

```powershell
.venv/Scripts/python.exe scripts/prepare_bulk_benchmark.py --workspace "$env:LOCALAPPDATA\DrugAssist\bulk"
```

This writes `analysis/benchmark_plan.json` containing whole label sets sampled
from four passage-count ranges, their source locations, and sample token totals.
Targets may be exceeded to retain complete labels. Tokens are an upper bound
before duplicate/cache savings, suitable for pricing a sample before any paid
requests. This preparation does not run indexing or embeddings.
