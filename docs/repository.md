# Repository and saved data

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

This guide explains the main folders, saved artifacts and local files needed to reproduce a search or research result.

See the [annotated database schema](database-schema.md) for Chroma tables, vector-index files, the embedding cache and account relationships.

## Files worth knowing

| Location | Purpose |
|---|---|
| `docs/` | Detailed usage, app flow, evaluation and repository guides |
| `notebooks/` | The reviewer-facing explanation and results |
| `src/` | Shared retrieval, evaluation, API and authentication code |
| `frontend/app.py` | Minimal Streamlit signup, login and search screens |
| `src/auth/` | Separate account database and session authentication |
| `config/default.json` | Drug names, models and search settings |
| `data/benchmarks/query_specs.json` | Curated question templates for building the original benchmark; app vocabulary rules live in `src/query_parser.py` |
| `data/raw/` and `data/processed/` | Original downloads and saved datasets |
| `data/cache/`, `data/chroma/`, `data/models/` | Cached embeddings, search index and local models |
| `data/benchmarks/benchmark.json` | Original test questions and draft source evidence |
| `results/search_latest.json` | Pointer to the active 60-drug index |
| `results/latest.json` | Pointer to the original saved comparison |
| `reports/data_checks.json` | Dataset ID, drug coverage, data quality, file integrity and name-search checks |
| `reports/expanded_60/method_comparison.json` | Saved questions, source references and results used by notebook 3 |
| `reports/api_contract.json` | Recorded external data API capabilities |
| `tests/` | Checks that the implementation behaves as intended |

VS Code workspace settings hide generated environment, Python cache, test cache, lint cache and package metadata folders in the Explorer. The files stay on disk and remain available to the app and tools.

Recent summaries and name-lookup records come from MORVS. Full records come from FDA and supply the searchable text. Some MORVS text is shortened. Source files and fingerprints are retained so results can be checked. An FDA label record does not by itself establish product approval or medical accuracy.

Large local datasets, models and indexes are ignored by Git. To reproduce the saved results on another computer, transfer those files as well as the repository. Fresh downloads can differ from the saved versions.

The API runs without `reports/`. Notebook 3 reads the saved method comparison; the maintenance command uses `data_checks.json` as an acquisition checkpoint and can fall back to the active saved dataset when it is absent. The reports support review and reproducibility. Keep the source data, settings, benchmark evidence and saved experiment results as well.
