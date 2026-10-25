# DrugAssist

Search FDA drug labels with hybrid retrieval, reranking and optional cited answers.
The prepared collection contains **60 drug names, 118 FDA labels and 8,601 passages**.

[Notebook 3](notebooks/03_retrieval_pipeline_benchmark.ipynb) compares six retrieval methods on **60 baseline + 10 hard questions**, reporting each group separately. See [evaluation details](docs/research.md).

## Quick start

1. **Install prerequisites:** [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11 or 3.12. Have your own OpenAI API key and internet access.

2. **Clone and enter the repository** on Windows, Linux or macOS:

   ```sh
   git clone https://github.com/suvrazastrovision/drug_rag.git
   cd drug_rag
   ```

3. **Add prepared data:** get the matching `drugassist-data.zip` from the owner. Extract its `data/` and `results/` folders into the repository root, merging existing folders. Confirm `data/chroma/chroma.sqlite3` exists. The ZIP is shared separately from Git.

4. **Choose a startup method:**
   - **Windows:** use the launcher below, or start both services manually.
   - **Linux/macOS:** follow the manual setup and terminal commands below.

### Windows launcher

Run in PowerShell from the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

The launcher installs dependencies, requests your API key once, starts both services and opens the browser. Keep the terminal open; **Ctrl+C** stops both services.

### Manual setup — Windows, Linux and macOS

Stop running app servers/notebooks before dependency setup:

```sh
uv sync --locked --inexact
```

Create `.env` from [.env.example](.env.example) **once**:

| Platform | Command |
|---|---|
| Windows PowerShell | `Copy-Item .env.example .env` |
| Linux/macOS terminal | `cp .env.example .env` |

Edit `.env`, enter your key and point the model loader to the prepared assets:

```dotenv
OPENAI_API_KEY=your-own-api-key
HF_HUB_CACHE=./data/models
```

Then check the extracted collection:

```sh
uv run --locked --no-sync python scripts/share_data.py --check
```

### Start backend and frontend from terminals

Open **two terminals**, both in the repository folder. These commands work in PowerShell, Bash and Zsh.

**Terminal 1 — backend:**

```sh
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend:**

```sh
uv run --locked --no-sync streamlit run frontend/app.py
```

- **App:** http://127.0.0.1:8501 — sign up, log in and ask a drug-label question.
- **API reference:** http://127.0.0.1:8000/docs
- **Stop manual servers:** press **Ctrl+C in each terminal**.
- **Answers:** uncheck **Write a cited answer** for evidence-only search.
- **Usage:** first setup may take several minutes; new question embeddings and answers can incur OpenAI charges.
- **Compatibility:** use matching code/configuration and a compatible Chroma bundle. Windows startup has been tested; Linux/macOS runtime and transferred native indexes have not been verified.

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
