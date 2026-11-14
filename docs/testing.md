# Tests and validation

[README](../README.md) | [Setup](setup.md) | [Usage](usage.md) | [Tests](testing.md) | [Troubleshooting](troubleshooting.md)

## Offline checks

After [installing the project](setup.md), stop servers and notebooks, then run:

```sh
uv sync --locked --inexact --extra dev --python 3.11
uv run --locked --no-sync python scripts/test_clean.py
uv run --locked --no-sync ruff check src tests scripts frontend
```

The suite runs twice in fresh source copies with isolated caches, temporary stores and simulated
providers. External networking is blocked; no data setup or API key is needed. Dependency installation
still needs internet or cached packages. For one ordinary run: `uv run --locked --no-sync pytest -q`.

## Continuous integration

[GitHub Actions](https://github.com/suvrazastrovision/DrugAssist/actions) runs these offline tests
and lint checks on pushes/PRs, using Windows/Linux and Python 3.11/3.12. Open a run to see results;
macOS is not in the CI matrix.

Ruff enforces code-style and PEP 257 docstring checks through `ruff.toml`, inheriting the
project's 110-character line limit. Public application interfaces require docstrings;
test and fixture names are exempt from missing-docstring checks.

## Manual online check

Start the app with internet and a valid API key (charges apply):

1. Open [health](http://127.0.0.1:8000/health); expect `"status": "ready"`. This checks the index, not API access.
2. Sign up/log in; submit a new drug-label question with answer writing unchecked and inspect the passages.
3. Enable **Write a cited answer**, submit again and review the answer and sources.

No automated `live` tests currently exist; `pytest -m live` does not run this online check.

## Optional tracing checks

To include the real Langfuse SDK's in-memory exporter check:

```sh
uv sync --locked --inexact --extra dev --extra observability --python 3.11
uv run --locked --no-sync pytest -q tests/test_observability.py
```

These tests use synthetic inputs and fake or in-memory exporters; they do not send traces to
Langfuse Cloud or make paid model calls. The SDK-specific check is skipped when the optional
package is absent. For dataset-based answer assessment, see [App autograding](evaluation.md).
