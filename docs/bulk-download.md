# FDA bulk download and sizing

## Status

Downloads and sizing completed on **2026-11-25**. The 60-drug app and retrieval
settings remain unchanged. No bulk embeddings or new index have been created.

| Result | Total |
|---|---:|
| Source label records scanned | 265,009 |
| Approval-matched label sets | 41,240 |
| Passages counted | 2,016,926 |
| Embedding-input tokens, including repeats | 474,191,969 |
| Downloaded ZIPs | 1.76 GiB |
| Analysis artifacts, including backup and sample plan | Approximately 285 MiB |

These are snapshot figures; consult generated reports after future runs.

## Location and outputs

Workspace: `%LOCALAPPDATA%\DrugAssist\bulk`, outside the repository and OneDrive.
Keep the entire workspace for verification and resume.

| Path relative to workspace | Purpose |
|---|---|
| `downloads/` | Label and Drugs@FDA source ZIPs; no extraction |
| `fda_manifest.json`, `download_state.json` | Pinned release and download checkpoints |
| `download_report.json` | Download completion, source counts and sizes |
| `analysis/sizing_report.txt`, `analysis/sizing_report.json` | Readable and detailed sizing results |
| `analysis/analysis.sqlite3` | Matching audit and resumable sizing checkpoints |
| `analysis/progress.json` | Latest progress during analysis |
| `baseline_verification.json`, `analysis/baseline_verification.json` | Baseline preservation checks for each stage |
| `analysis/baseline_comparison.json` | Original labels compared with the approval rule |
| `analysis/benchmark_plan.json` | Prepared sample records, token counts and optional cost estimates |

## Commands

Run in PowerShell from the repository root, using the existing virtual environment.
Steps 1, 2 and 4 are already complete for this workspace.

```powershell
$bulkWorkspace = Join-Path $env:LOCALAPPDATA 'DrugAssist\bulk'

# 1. Download sources; reuse verified files.
.venv/Scripts/python.exe scripts/download_fda_bulk.py --workspace $bulkWorkspace

# 2. Match approvals and count all passages/tokens.
.venv/Scripts/python.exe scripts/analyze_fda_bulk.py --workspace $bulkWorkspace

# 3. Read the latest saved sizing report.
.venv/Scripts/python.exe scripts/analyze_fda_bulk.py --workspace $bulkWorkspace --stage status

# 4. Prepare samples without indexing or embeddings.
.venv/Scripts/python.exe scripts/prepare_bulk_benchmark.py --workspace $bulkWorkspace
```

Sample preparation replaces `benchmark_plan.json`. To include cost estimates,
supply `--usd-per-million` and `--pricing-source` with a reviewed rate and URL.

## Matching and sizing rules

Labels require an explicit NDA/ANDA/BLA application-number match to an approved
(`AP`) Drugs@FDA submission. Tentative-only applications are excluded. This is
application-level approval evidence, not current marketing status or approval
of each label's text.

Partitions are processed sequentially, selecting the newest matched version per
label set. Existing normalization and recursive **256-token chunks with
32-token overlap** use `cl100k_base`. Token counts include identity and section
text; distinct-input totals support cache sizing.

Analysis runs offline and calculates passage-file and vector sizes without
creating those files.

## Resume and limits

Press Ctrl+C, then rerun the same command. Analysis rereads the interrupted
partition and skips committed work. Resume with unchanged pinned sources, code
and configuration. Use a separate workspace for a newer FDA release.

| Guard | Limit |
|---|---:|
| Free disk reserve for download and analysis | 80 GiB |
| Total download-workspace allowance | 40 GiB |
| Additional analysis storage, including checkpoint backups | 2 GiB |
| Analysis process peak-RAM stop threshold | 4 GiB |

Avoid baseline indexing, maintenance or embedding-cache updates during these
stages; preservation checks compare those files.

## Next step

Benchmark the prepared **2,076- and 10,070-passage samples** using the existing
Chroma/BM25/fusion/reranker pipeline. Measure indexing and searching separately.
Preparation makes no paid requests; review costs and set an embedding spending
limit first.

Storage scenarios are hypothetical; full-index RAM and latency remain
unmeasured. The 335 MiB peak RAM describes the final analysis invocation.
Choose a server within the **EUR 50/month hosting budget** after benchmarking.
