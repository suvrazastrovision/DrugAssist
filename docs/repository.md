# Repository and saved data

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

## Project files

| Location | Role |
|---|---|
| `config/default.json` | Drug list, models, indexing and search settings |
| `docs/` | Usage, architecture, storage and research guides |
| `frontend/app.py` | Streamlit signup, login, search and answer display |
| `notebooks/` | Data inspection and saved comparison results |
| `scripts/` | Maintenance helpers, method comparison, notebook execution and artifact checks |
| `scripts/share_data.py` | Export public prepared data/models; check restored bundles |
| `start.ps1` | Windows dependency setup, key prompt and two-server launcher |
| `src/` | Shared data, retrieval, evaluation, API and authentication implementation |
| `src/query_parser.py` | Drug aliases and topic vocabulary rules |
| `src/services/answer_service.py` | FDA context, one model call and citation checks |
| `tests/` | Automated implementation checks |

## Local data and databases

| Location | Contents |
|---|---|
| `data/raw/` | Original API responses and provenance metadata |
| `data/processed/` | Normalized snapshots and complete saved sections |
| `data/chroma/` | Chroma database and vector-index files |
| `data/cache/` | Reusable document and question embeddings |
| `data/models/` | Downloaded local model assets |
| `data/auth/users.sqlite3` | Separate accounts and sessions |

- Search uses full FDA records; MORVS summaries/lookups may be shortened.
- Fingerprints trace versions, not medical accuracy or approval.
- Table relationships: [database schema](database-schema.md).

## Indexes and reports

| Location | Purpose |
|---|---|
| `results/search_latest.json` | Select the active index |
| `results/search_indexes/<index-id>/` | Index descriptor, passages and settings |
| `reports/data_checks.json` | Data-quality/coverage checks and maintenance acquisition checkpoint |
| `reports/expanded_60/method_comparison.json` | Questions, expected sections and results for notebook 3 |
| `reports/api_contract.json` | Recorded external API capabilities |

- API: runs without `reports/`; notebook 3 needs its saved comparison.
- Maintenance: uses `data_checks.json`, falling back to the active dataset when absent.

## Reproduce or transfer the project

- A fresh clone can run `fda-rag setup` to download a pinned release bundle, or
  `fda-rag setup --from-source` to acquire data and create embeddings locally.
- The default path never automatically falls back to paid source rebuilding.
- Transfer source data, models, active index and comparison alongside code/configuration; large artifacts are Git-ignored.
- Fresh downloads may differ from saved versions.
- VS Code hides generated environments/caches; files remain on disk.
- Commands and evaluation limits: [Research](research.md).

## Export a prepared bundle — repository owner

Stop app servers and ingestion/indexing jobs, then run:

```powershell
uv run --locked --no-sync python scripts/share_data.py
```

- Share the generated `drugassist-data.zip` alongside the matching repository version; the ZIP is Git-ignored.
- Includes the active snapshot, referenced raw responses, index descriptors, complete Chroma store and pinned reranker.
- Excludes credentials, accounts, query/embedding cache, reports and Python environment.
- Export backs up SQLite and rejects source changes during packaging.
- Startup verifies checksums once, then immutable files and Chroma counts on later runs.
- Recipients follow the [quick start](../README.md#quick-start); they create their own accounts and use their own API keys.

## Prepare and publish a release asset

Stop services and ingestion jobs. From the matching code checkout, run:

```sh
uv run --locked --no-sync python scripts/prepare_release.py --tag data-v0.1.0
```

This exports the public collection into `results/private/release/`, restores it into a
temporary directory, checks readiness, and writes `config/distribution.json` with its
exact URL, checksum, platform and Python version. It does not publish anything.
The restoration check loads the real reranker and native Chroma index in an offline
child process, without issuing a question or using an API key.

1. Review the exported contents and rerun the offline tests.
2. Commit the matching code, configuration, lockfile and distribution descriptor.
3. Create the immutable GitHub Release tag specified above at that commit.
4. Upload the exact ZIP printed by the command; do not recompress or replace it.
5. Run `fda-rag setup` in a clean checkout on the advertised platform and test startup.
6. Use a new release tag when data/configuration changes. Only advertise platforms
   whose restored native indexes have been tested. Other platforms use `--from-source`.

Archive checksums refer to the exact uploaded bytes. Generated archives stay outside
Git history. Keep published comparison reports separate from source-built collections.
