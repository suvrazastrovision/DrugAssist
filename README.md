# DrugAssist

Search FDA drug labels with hybrid retrieval, reranking and optional cited answers.
The prepared collection contains **60 drug names, 118 FDA labels and 8,601 passages**.

[Notebook 3](notebooks/03_retrieval_pipeline_benchmark.ipynb) compares six retrieval methods on **60 baseline + 10 hard questions**, reporting each group separately. See [evaluation details](docs/research.md).

## Quick start

Install Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Python
3.11 or 3.12. Clone the repository, then install dependencies:

```sh
git clone https://github.com/suvrazastrovision/drug_rag.git
cd drug_rag
uv sync --locked --inexact
```

Create `.env` from [.env.example](.env.example) once (`Copy-Item .env.example .env`
in PowerShell, or `cp .env.example .env` in Bash). Enter your own `OPENAI_API_KEY`.
New query embeddings and optional written answers can incur OpenAI charges.

### Choose a setup path

| Path | Command | Requirements |
|---|---|---|
| Download prepared data (default) | `uv run --locked --no-sync fda-rag setup` | Internet; published release asset for your platform/Python |
| Build fresh data | `uv run --locked --no-sync fda-rag setup --from-source` | Internet and your OpenAI key; document embedding charges apply |

The default downloads an exact versioned GitHub Release asset and verifies its SHA-256,
member checksums, configuration, runtime, source inventory and model files before activation.
No personal ZIP handoff is needed. The matching asset must first be published by the maintainer;
until it is published, use the source-build command. A failed download never starts a paid build.

The prepared native index currently targets **Windows x64, Python 3.11**. On Linux, macOS,
other architectures, or Python 3.12, use `--from-source`. These source-build paths have automated
engineering coverage; live provider setup must still be checked on each target platform.

Fresh builds use direct openFDA lookups if MORVS is unavailable, build an isolated collection,
and download the pinned reranker. They can take several minutes and produce different labels
from the saved benchmark. The saved benchmark does not evaluate a rebuilt collection.

Setup reuses a compatible existing installation. For a different configuration or fresh rebuild,
use a separate checkout. Stop services before setup. Credentials, accounts and query caches are
excluded from downloaded bundles and are preserved during installation.

### Windows launcher

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

For the source-build path:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1 -FromSource
```

The launcher installs dependencies, requests your key if needed, runs setup, starts both
services and opens the browser. Keep the terminal open; **Ctrl+C** stops both services.

### Manual startup on all platforms

After setup, open two terminals in the repository folder.

**Backend:**

```sh
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Frontend:**

```sh
uv run --locked --no-sync streamlit run frontend/app.py
```

- App: http://127.0.0.1:8501 - sign up, log in and ask a drug-label question.
- API reference: http://127.0.0.1:8000/docs
- Stop: press **Ctrl+C** in each terminal.
- Uncheck **Write a cited answer** for evidence-only search.

### Run tests without prepared data or existing caches

```sh
uv sync --locked --extra dev
uv run --locked --no-sync python scripts/test_clean.py
uv run --locked --no-sync ruff check src tests scripts frontend
```

The test runner performs two runs in fresh source copies, excluding prepared data, reports,
`.env`, and runtime caches. Each process uses empty temporary model/tokenizer cache paths.
Tests use a checked-in, hash-verified real tokenizer vocabulary, temporary databases, and
simulated providers. External HTTP and non-loopback sockets are blocked. No API key or ZIP is required.

For a single ordinary run: `uv run --locked --no-sync pytest -q -p no:cacheprovider`.
Tests marked `live` are excluded by default; select them explicitly with `pytest -m live`.
Installing dependencies requires internet or an available package cache; test execution does not.

## Documentation

| Guide | Details |
|---|---|
| [Run and search](docs/usage.md) | Accounts, settings, API, answer writing and terminal search |
| [Application flow](docs/app-flow.md) | Data preparation, retrieval, reranking and answer writing |
| [Database schema](docs/database-schema.md) | Chroma, embedding cache and account tables |
| [Notebooks and evaluation](docs/research.md) | Data checks, comparisons, maintenance and scientific limits |
| [Repository and saved data](docs/repository.md) | Folder roles, required artifacts and data transfer |
| [Configuration](config/default.json) | Drug list, models, chunking and retrieval settings |

Outputs are informational, not personalized medical advice. Scores are relevance rankings, not confidence. Review the cited FDA text; the saved comparison does not validate clinical answer quality.
