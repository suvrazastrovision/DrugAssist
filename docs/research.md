# Notebooks and evaluation

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

Data checks, source-section matching and human relevance review answer different questions. The saved comparisons should be read with their stated limitations.

## Read the notebooks

Open them in order and run their cells from top to bottom. They use saved local files without API calls.

Install notebook tools with `uv sync --locked --extra research`, then select the Python interpreter in `.venv`.

| Notebook | Main question | Data used |
|---|---|---|
| [1. Data quality](../notebooks/01_data_quality_analysis.ipynb) | Are the downloaded files usable, and what is missing? | Current 60-drug collection |
| [2. Explore and search](../notebooks/02_exploratory_data_analysis.ipynb) | How do labels become passages, and how can I search them? | Current 60-drug collection |
| [3. Compare search methods](../notebooks/03_retrieval_pipeline_benchmark.ipynb) | How often did each method find the requested section? | Current 60-drug collection / 60 simple questions |

Notebook 3 compares all six methods using one direct section-finding question per drug: 20 about uses, 20 about dosage instructions and 20 about warnings. Expected matches come from FDA identity and section fields. Finding an expected section does not prove that a passage answers a medical question. This automatic check is separate from the original 80-question draft benchmark, which still needs independent human review.

## Maintain and check the project

```powershell
# Check all configured drug names offline. This is also the default maintenance step.
uv run --locked fda-rag maintain

# Fetch missing labels, build the index, then check it. Indexing can use the embedding API.
uv run --locked fda-rag maintain --step all

# Individual steps are available with --step acquire, index or verify.

# Run or resume the six-method comparison used by notebook 3.
uv run --locked --extra research python scripts/compare_search_methods.py

# Execute all three notebooks using this project's Python environment.
uv run --locked --extra research python scripts/execute_notebooks.py

# Run implementation checks.
uv run --locked --extra research --extra dev pytest -q
uv run --locked --extra research --extra dev ruff check src tests scripts frontend
```

Acquisition reuses existing matching labels; a direct FDA name search fills gaps left by MORVS. New indexing can incur embedding API usage. The original benchmark remains separate from the expanded search index.

Notebook execution and the full `benchmark` command require `--extra research`.

The first comparison run also embeds its new questions and runs the local reranker. Later notebook runs read the saved results without API calls. The separate local embedding comparison model is optional and downloads only when explicitly requested with `benchmark --compare-embeddings`; the saved results remain available without that download.

For independent relevance review, see [ANNOTATION_PROTOCOL.md](../ANNOTATION_PROTOCOL.md). Detailed metrics, configuration and source archives remain beside the original saved experiment. Keep these when sharing claims about its results.

Draft review exports are optional. Generate them only when needed with `uv run --locked python scripts/export_annotation_review.py`; the questions and draft evidence remain in `data/benchmarks/benchmark.json`.
