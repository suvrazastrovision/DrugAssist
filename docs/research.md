# Notebooks and evaluation

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

## Read the notebooks

- Run cells in order using `.venv`; notebooks read saved files without API calls.
- Stop servers/notebooks before installing research and test tools:

```powershell
uv sync --locked --extra research --extra dev
```

| Notebook | Purpose |
|---|---|
| [1. Data quality](../notebooks/01_data_quality_analysis.ipynb) | Check downloaded files, missing fields and source integrity |
| [2. Explore and search](../notebooks/02_exploratory_data_analysis.ipynb) | Inspect section lengths, passage splitting and a local search example |
| [3. Compare search methods](../notebooks/03_retrieval_pipeline_benchmark.ipynb) | Display saved six-method results for 60 drugs |

## What the comparison measures

- **60 questions:** one/drug; 20 uses, 20 dosage, 20 warnings.
- **Reference:** automatically matched FDA drug identity and section.
- **Scope:** unfiltered retrieval, without the app's score cutoff.
- **Timing:** excludes model loading, question embedding and index building.
- **Limits:** no independent relevance judgments, untouched test set or clinical/answer-quality validation.

## Maintain and check the project

| Command suffix | Action |
|---|---|
| `maintain` or `maintain --step verify` | Offline source coverage and name-search checks |
| `maintain --step acquire` | Reuse matching labels; fetch gaps through MORVS/openFDA |
| `maintain --step index` | Build using `config/default.json`; embedding charges may apply |
| `maintain --step all` | Acquire, index and verify |

Examples:

```powershell
uv run --locked --no-sync fda-rag maintain
uv run --locked --no-sync fda-rag maintain --step all
```

Verification checks indexed label coverage and matching-label retrieval for each configured drug.

### Refresh results and notebook outputs

```powershell
uv run --locked --no-sync python scripts/compare_search_methods.py
uv run --locked --no-sync python scripts/execute_notebooks.py
```

- Results: `reports/expanded_60/method_comparison.json`.
- Comparison can incur embedding usage and runs the local reranker; matching completed runs are reused.
- Reuse checks index/settings, questions and comparison-module hash, not every code/dependency change.
- Notebook execution refreshes displays from saved results.

### Check implementation behavior

```powershell
uv run --locked --no-sync pytest -q
uv run --locked --no-sync ruff check src tests scripts frontend
```

- Tests use temporary stores and simulated responses; external HTTP/`.env` loading are disabled.
- Human review remains necessary for relevance, factual support, qualifications and citation completeness.
