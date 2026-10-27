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
| [3. Compare search methods](../notebooks/03_retrieval_pipeline_benchmark.ipynb) | Compare six methods: 60 baseline + 10 hard questions across 60 drugs |

## What the comparison measures

- **60 baseline questions:** unchanged; 20 uses, 20 dosage, 20 warnings. Hits match FDA identity and section.
- **10 hard questions:** 3 vocabulary, 3 constraint, 2 multi-section and 2 near-match challenges.
- **Hard references:** 25 required evidence items, anchored to exact saved text, character offsets and source hashes before retrieval.
- **Hard scores:** any required evidence in top 10, mean requirement coverage and complete-evidence success; duplicate chunks earn no extra credit.
- **Separate groups:** baseline section matches and hard evidence coverage have different meanings; their percentages are not combined.
- **Scope:** unfiltered retrieval, without the app's score cutoff.
- **Timing:** excludes model loading, question embedding and index building.
- **Limits:** source anchors are repository-reviewed, not independently clinically reviewed or exhaustive relevance judgments. Equivalent evidence may be missed.
- Ten authored challenges are a small stress test, without validated difficulty, an untouched test set or clinical/answer-quality validation.

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
- The original collection/index is reused. New questions need no Chroma rebuild or document re-embedding.
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

For cache-independent verification, run `uv run --locked --no-sync python scripts/test_clean.py`.
It runs the suite twice in fresh source copies without prepared data, reports or credentials.
Tokenizer/model caches start empty in each process; the actual tokenizer vocabulary is
supplied by the checked-in test asset and checked against its upstream SHA-256. Cache behavior
tests populate and reopen their own temporary stores. Tests cannot read the repository's
runtime stores or the user's default model/tokenizer caches. Dependency installation is a
separate network-enabled step. `live` tests are opt-in with `pytest -m live`.
