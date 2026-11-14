# Set up DrugAssist

[README](../README.md) | [Setup](setup.md) | [Usage](usage.md) | [Tests](testing.md) | [Troubleshooting](troubleshooting.md)

Use **PowerShell on Windows** or **Terminal on macOS/Linux**. Run commands one at a time;
wait for each to finish and resolve any error before continuing. First setup needs internet.

## 1. Install the tools

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

## 2. Download the app

Use a local folder outside OneDrive or other cloud-sync folders:

```sh
git clone https://github.com/suvrazastrovision/DrugAssist.git DrugAssist
cd DrugAssist
uv sync --locked --inexact --python 3.11
```

Downloads may take several minutes. Python 3.11–3.12 is supported; these steps use 3.11.
Run the remaining commands inside `DrugAssist`.

## 3. Set up and launch

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
[data release](https://github.com/suvrazastrovision/DrugAssist/releases/tag/data-v0.1.0).

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

Open **two terminals**, both inside `DrugAssist`:

```sh
# Terminal 1: search API
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```sh
# Terminal 2: browser interface
uv run --locked --no-sync streamlit run frontend/app.py --server.address 127.0.0.1
```

Manual setup/startup also works on Windows; use `Copy-Item .env.example .env` to create `.env`.

## 4. Use and reopen

- Open [DrugAssist](http://127.0.0.1:8501), **sign up**, then **log in** (password: at least 8 characters).
- Try **“What warnings are listed for warfarin?”** and review the FDA sources.
- Uncheck **Write a cited answer** for evidence only; new semantic searches still use the API.
- Keep terminals open. Press **Ctrl+C** to stop (in both terminals for manual startup).
- Next time, reopen the repository folder and repeat the launcher or the two server commands.
  Your key/account are saved; setup reuses compatible data.

New semantic questions go to OpenAI; answer writing also sends supporting passages.
[Usage and settings](usage.md) · [Local API reference](http://127.0.0.1:8000/docs)
