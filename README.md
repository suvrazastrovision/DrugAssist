# DrugAssist

Search FDA drug labels, get an optional cited answer and read the supporting passages.
The app uses **hybrid retrieval with cross-encoder reranking**, followed by one optional LLM writing step. Outputs are informational, not personalized medical advice.

The saved collection covers **60 selected drug names**, **118 full FDA labels** and **8,601 passages**. It is a sample of FDA data. See the [drug list and settings](config/default.json).

## Quick start

**Windows, using prepared data:**

1. Install [uv](https://docs.astral.sh/uv/) and clone/download this repository.
2. Get `drugassist-data.zip` from the repository owner and extract it **into the repository root**, merging `data/` and `results/`.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

- First run installs locked dependencies and asks for **your own OpenAI API key**, saved in local `.env`.
- Checks the prepared collection, starts both servers and opens the browser. No FDA download or document re-embedding.
- Sign up locally, log in and ask a question. New question embeddings and written answers use your API key.
- Keep the terminal open; **Ctrl+C** stops both servers. Logs: `results/private/startup/`.
- Requires internet for installation and OpenAI requests. The ZIP is shared separately, not stored in Git.

For manual startup, see [Run and search](docs/usage.md#start-the-app). Use the matching repository version; the launcher checks configuration, project metadata and `uv.lock`. Prepared Chroma files are intended for compatible Windows environments.

## Share prepared data

**Repository owner:** stop app servers and ingestion/indexing jobs before exporting:

```powershell
uv run --locked --no-sync python scripts/share_data.py
```

- Creates `drugassist-data.zip` in the repository root; share it alongside the matching repository version.
- Includes the active FDA snapshot, referenced raw responses, index descriptors, complete Chroma store and pinned reranker assets.
- Excludes `.env`, accounts, embedding/query cache, reports and Python environment. Recipients create fresh accounts/cache.
- SQLite is backed up consistently; export fails if source data changes during packaging.
- Includes checksums and configuration/lockfile hashes. Startup verifies original files once, immutable files and Chroma counts on later runs.

**Write a cited answer** adds Patient or Clinician wording above the evidence. Uncheck it for evidence-only search. See [answer writing](docs/usage.md#cited-answers). API reference: **http://127.0.0.1:8000/docs**.

## Documentation

| Guide | What it explains |
|---|---|
| [Run and search](docs/usage.md) | Setup, accounts, API, terminal search and filters |
| [App flow diagram](docs/app-flow.md) | Data preparation, login, hybrid reranking and result display |
| [Database schema](docs/database-schema.md) | Annotated Chroma, embedding-cache and account-storage diagram |
| [Notebooks and evaluation](docs/research.md) | Notebook walkthroughs, scientific limits, checks and maintenance |
| [Repository and saved data](docs/repository.md) | Folder roles, local databases, reports and reproducibility |

## Interpreting results

Results are chunks of label sections and can end before the full section does. Scores rank passages; they are not medical confidence values.

Written answers use expanded saved context and checked source IDs. These checks do not prove factual support; review the cited FDA text. If writing fails, the retrieved passages remain available.

The 60-question comparison checks drug and section matching. It does not establish clinical answer quality; that needs independent human review. See [evaluation details](docs/research.md).
