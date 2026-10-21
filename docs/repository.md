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

- Transfer source data, models, active index and comparison alongside code/configuration; large artifacts are Git-ignored.
- Fresh downloads may differ from saved versions.
- VS Code hides generated environments/caches; files remain on disk.
- Commands and evaluation limits: [Research](research.md).
