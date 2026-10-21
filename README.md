# DrugAssist

Search FDA drug labels using hybrid retrieval and reranking, with optional cited answers.
The prepared collection covers **60 drug names, 118 FDA labels and 8,601 passages**.

## Quick start

1. **On Windows, install** [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). Internet access and your own OpenAI API key are required.

2. **Clone the repository** and enter its folder:

   ```powershell
   git clone https://github.com/suvrazastrovision/drug_rag.git
   cd drug_rag
   ```

3. **Add the prepared data.** Get `drugassist-data.zip` from the repository owner, matching this repository version. Extract its `data/` and `results/` folders beside `start.ps1`, merging existing folders. Check that `data/chroma/chroma.sqlite3` now exists.

   The ZIP is shared separately; cloning alone does not include the data or models.

4. **Start the app** from PowerShell in the repository folder:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start.ps1
   ```

   The launcher installs dependencies, checks the prepared collection and asks for your API key once, saving it in local `.env`. It starts both servers without downloading labels or rebuilding document embeddings. The first launch may take several minutes.

5. **Use the app.** Your browser opens at **http://127.0.0.1:8501**. Sign up, log in and ask a drug-label question. Leave **Write a cited answer** checked for a summary, or uncheck it for passages only.

- **Stop:** keep the terminal open; press **Ctrl+C** to stop both servers.
- **API usage:** new question embeddings and written answers can incur OpenAI charges.
- **Troubleshooting:** stop existing servers/notebooks before retrying dependency setup. Logs: `results/private/startup/`; [manual startup and settings](docs/usage.md).
- **Compatibility:** the launcher checks configuration, project metadata and `uv.lock`. Use the matching data bundle and repository version in a compatible Windows environment.

## Documentation

| Guide | Details |
|---|---|
| [Run and search](docs/usage.md) | Manual startup, accounts, API, answer settings and terminal search |
| [Application flow](docs/app-flow.md) | How data preparation, retrieval, reranking and answer writing connect |
| [Database schema](docs/database-schema.md) | Chroma, embedding cache and account tables |
| [Notebooks and evaluation](docs/research.md) | Data checks, saved comparisons, maintenance and scientific limits |
| [Repository and saved data](docs/repository.md) | Folder roles, required artifacts and data transfer |
| [Drug list and configuration](config/default.json) | Selected drugs, models, chunking and retrieval settings |

Interactive API reference, while running: **http://127.0.0.1:8000/docs**.

## Read results carefully

Passages can end mid-section. Scores are relevance rankings, not medical confidence. Review cited FDA text: format and source-ID checks do not prove factual support. The saved comparison checks drug/section matching, not clinical answer quality. Outputs are informational, not personalized medical advice.
