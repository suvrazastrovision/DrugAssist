# DrugAssist

Search FDA drug labels and get optional AI-written answers with supporting sources.
Prepared collection: **60 drug names · 118 FDA labels · 8,601 passages**.

[Setup](#quick-start) · [Tests](#tests) · [Troubleshooting](#troubleshooting) · [Guides](#documentation)

## Quick start

Use **PowerShell on Windows** or **Terminal on macOS/Linux**. Run commands one at a time;
wait for each to finish and resolve any error before continuing. First setup needs internet.

### 1. Install the tools

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/):

**Windows:**

```powershell
winget install --id=astral-sh.uv -e
```

**macOS / Linux:**

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reopen your terminal; check `git --version` and `uv --version`.
uv downloads Python when needed, so no separate Python installation is required.

### 2. Download the app

Use a local folder outside OneDrive or other cloud-sync folders:

```sh
git clone https://github.com/suvrazastrovision/drug_rag.git
cd drug_rag
uv sync --locked --inexact --python 3.11
```

Downloads may take several minutes. Python 3.11–3.12 is supported; these steps use 3.11.
Run the remaining commands inside `drug_rag`.

### 3. Set up and launch

Have your own [OpenAI API key](https://platform.openai.com/api-keys) ready.
New semantic searches, written answers and source-build embeddings can incur API charges.

Cloning the code does not include the searchable data. Choose either setup path:

| Path | What happens | Use on |
|---|---|---|
| Download prepared data | The launcher downloads the ZIP automatically; nobody needs to send it to you. Requires a published data release. | Windows x64 + Python 3.11 |
| Build from source | Downloads FDA labels and builds the index locally; no prepared ZIP is needed. Takes longer and incurs embedding charges. | macOS/Linux, or Windows as an alternative |

**Windows x64 — automatic ZIP download:**

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

The launcher asks for your key (entry is hidden), saves it in `.env`, downloads prepared data
and opens the app. Wait for `DrugAssist is ready: http://127.0.0.1:8501`.

Prepared data requires **Windows x64 + Python 3.11** and the published
[data release](https://github.com/suvrazastrovision/drug_rag/releases/tag/data-v0.1.0).

**Windows alternative — build from source without a ZIP:**

If the release is unavailable, or you prefer to build the data yourself, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1 -FromSource
```

**macOS / Linux — build from source without a ZIP:**

```sh
cp .env.example .env
```

Open `.env` in a text editor. Set `OPENAI_API_KEY=your-key-here`, save and keep it private.
`FDA_API_KEY` is optional. Build the collection:

```sh
uv run --locked --no-sync fda-rag setup --from-source
```

This downloads FDA data and the local model, then builds the index. It may take several minutes
and produce different data from the saved benchmark. Other architectures and Python 3.12 also
need a source build; dependency availability varies. Live setup needs checking on each target platform.

Open **two terminals**, both inside `drug_rag`:

```sh
# Terminal 1: search API
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```sh
# Terminal 2: browser interface
uv run --locked --no-sync streamlit run frontend/app.py --server.address 127.0.0.1
```

Manual setup/startup also works on Windows; use `Copy-Item .env.example .env` to create `.env`.

### 4. Use and reopen

- Open [DrugAssist](http://127.0.0.1:8501), **sign up**, then **log in** (password: at least 8 characters).
- Try **“What warnings are listed for warfarin?”** and review the FDA sources.
- Uncheck **Write a cited answer** for evidence only; new semantic searches still use the API.
- Keep terminals open. Press **Ctrl+C** to stop (in both terminals for manual startup).
- Next time, reopen the repository folder and repeat the launcher or the two server commands.
  Your key/account are saved; setup reuses compatible data.

New semantic questions go to OpenAI; answer writing also sends supporting passages.
[Usage and settings](docs/usage.md) · [Local API reference](http://127.0.0.1:8000/docs)

## Tests

**Offline tests — all three operating systems:** after step 2, stop servers/notebooks and run:

```sh
uv sync --locked --inexact --extra dev --python 3.11
uv run --locked --no-sync python scripts/test_clean.py
uv run --locked --no-sync ruff check src tests scripts frontend
```

The suite runs twice in fresh source copies with isolated caches, temporary stores and simulated
providers. External networking is blocked; no data setup or API key is needed. Dependency installation
still needs internet or cached packages. For one ordinary run: `uv run --locked --no-sync pytest -q`.

**Online CI:** [GitHub Actions](https://github.com/suvrazastrovision/drug_rag/actions) runs these offline tests
and lint checks on pushes/PRs, using Windows/Linux and Python 3.11/3.12. Open a run to see results;
macOS is not in the CI matrix.

**Real online check:** start the app with internet and a valid API key (charges apply):

1. Open [health](http://127.0.0.1:8000/health); expect `"status": "ready"`. This checks the index, not API access.
2. Sign up/log in; submit a new drug-label question with answer writing unchecked and inspect the passages.
3. Enable **Write a cited answer**, submit again and review the answer and sources.

No automated `live` tests currently exist; `pytest -m live` does not run this online check.

## Troubleshooting

| Problem | Try this |
|---|---|
| `uv` / `git` not recognized | Reopen the terminal; check the installation links above. |
| Installation: `Access is denied` | Stop servers/notebooks, pause folder syncing and retry. Persistent incomplete metadata may require recreating `.venv`. |
| Prepared download fails | Check release availability and platform/Python requirements; explicitly choose a source build if needed. |
| Port already in use | Stop the earlier app. Windows launcher alternatives: `-ApiPort 8001 -UiPort 8502`. |
| Search/answer fails | Check API key and billing/access. Launcher logs: `results/private/startup/`. |

Stop services before changing dependencies/data. Use a separate checkout for a fresh configuration.
Prepared downloads verify checksums, preserve credentials/accounts and never automatically fall back to a paid build.

## Documentation

| Guide | Contents |
|---|---|
| [Usage](docs/usage.md) | Accounts, answers, settings, API and terminal search |
| [App flow](docs/app-flow.md) | Data preparation and search pipeline |
| [Database schema](docs/database-schema.md) | Labels, caches and accounts |
| [Research](docs/research.md) | Notebooks, evaluation and maintenance |
| [Repository and data](docs/repository.md) | Files, data transfer and release publishing |
| [Configuration](config/default.json) / [Environment](.env.example) | Drug list, models and settings |
| [Benchmark notebook](notebooks/03_retrieval_pipeline_benchmark.ipynb) | Six methods; 60 baseline + 10 hard questions, reported separately |

For information, not personalized medical advice. Scores rank relevance, not correctness or confidence.
Review FDA sources. The saved benchmark does not validate clinical answer quality or a rebuilt collection.

## License

Project code is available under the [MIT License](LICENSE).
Third-party dependencies, model weights and datasets retain their own licenses and terms.
