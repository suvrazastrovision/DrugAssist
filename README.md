# DrugAssist

Search FDA drug labels, get an optional cited answer and read the supporting passages.
The app uses **hybrid retrieval with cross-encoder reranking**, followed by one optional LLM writing step. Outputs are informational, not personalized medical advice.

The saved collection covers **60 selected drug names**, **118 full FDA labels** and **8,601 passages**. It is a sample of FDA data. See the [drug list and settings](config/default.json).

## Quick start

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Configure your private `.env` using [.env.example](.env.example); hybrid search needs `OPENAI_API_KEY` and the local reranker assets.

Stop app servers and notebook executions before installing dependencies:

```powershell
uv sync --locked
```

Start the app in two terminals:

```powershell
# Terminal 1 - backend
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2 - frontend
uv run --locked --no-sync streamlit run frontend/app.py
```

Open **http://localhost:8501**, sign up, log in and enter a drug-label question. The API reference is at **http://127.0.0.1:8000/docs**.

**Write a cited answer** adds a short summary above the evidence, with Patient or Clinician wording. It uses your existing `OPENAI_API_KEY`; model and token limits are in `config/default.json`. Uncheck it for evidence-only search. See [answer writing](docs/usage.md#cited-answers).

The app uses saved local data, models and a Chroma index. These large files are ignored by Git. For a new machine, see [data transfer](docs/repository.md) and [collection preparation](docs/research.md#maintain-and-check-the-project).

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
