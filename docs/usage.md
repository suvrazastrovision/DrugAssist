# Run and use DrugAssist

[README](../README.md) | [Run and search](usage.md) | [App flow](app-flow.md) | [Research](research.md) | [Repository guide](repository.md)

## Start the app

For prepared-data setup on Windows, use the [single-command launcher](../README.md#quick-start). Manual startup follows below.

1. Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).
2. Configure private settings using [`.env.example`](../.env.example). Keep `OPENAI_API_KEY` in the ignored `.env`.
3. Stop servers and notebooks before installing dependencies:

```powershell
uv sync --locked
```

Start each server in its own terminal. `--no-sync` avoids changing the running environment.

```powershell
# Terminal 1: backend
uv run --locked --no-sync uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: frontend
uv run --locked --no-sync streamlit run frontend/app.py
```

- Open **http://localhost:8501**, sign up, then log in.
- Enter a drug-label question; inspect passages, dates, scores and FDA links.
- Both servers default to localhost; remote API URLs require HTTPS.

## Search behavior

| Step | Current behavior |
|---|---|
| Parse | Recognize FDA drug aliases and a small dictionary of label topics and everyday phrases |
| Filter | Match document and section exactly; ambiguous attributes remain unset |
| Retrieve | Combine 25% BM25 and 75% semantic scores; keep up to 30 candidates |
| Rerank | Score question–passage pairs with the pinned local MiniLM cross-encoder |
| Return | Keep top `k` passages passing the experimental raw-score cutoff of `0.5` |

- Questions stay unchanged; spelling is not corrected. “Unwanted effects” maps to `adverse_reactions`.
- Conditions are **not database filters**. Natural-language manufacturer, route, version and date constraints are not interpreted.
- Out-of-scope questions or missing evidence return `no_match: true`; filters are not relaxed.
- Scores rank passages. **A score of 0.5 is not 50% confidence**; the cutoff is uncalibrated.
- Uncached semantic questions require the embedding provider; saved vectors are reused.
- Passage expanders show chunks, which can end mid-list. Answer mode can supply more context.

See the [complete pipeline](app-flow.md) and [storage layout](database-schema.md).

## Cited answers

- Leave **Write a cited answer** checked; select **Patient** or **Clinician** wording.
- Output: **Answer**, optional **Details**, **Important warnings**, and **What this answer covers**.
- **Supporting FDA context** shows cited excerpts, label dates and FDA links.
- Audience changes wording, not evidence; it does not verify professional roles.
- Uncheck answer writing for evidence alone.

### Writer settings

Defaults live in [`config/default.json`](../config/default.json), under `answer`.

| Setting | Default |
|---|---|
| Provider / model | OpenAI Responses / `gpt-5-mini` |
| Reasoning | `minimal`; use `null` for models without this option |
| Context | Up to 4 sources, 1,600 tokens each |
| Combined instructions, question and evidence | 6,000-token budget |
| Output / read timeout | 1,500 tokens / 60 seconds |
| Generation attempts | One request; no automatic retry |

- Context uses exact saved text; covered passages are skipped and overlapping windows may merge. Long sections remain partial; products/versions stay separate.
- Budgets use `cl100k_base`; provider counts can differ.
- Prompts request preserved units/negation, citations and abstention. Compliance is not guaranteed.
- Questions/context go to OpenAI with `store=false`; other retention policies still apply. API charges may apply.
- No local answer history. Successful responses record the reported model snapshot; aliases may change.

### Answer states and citations

| `answer_status` | Meaning |
|---|---|
| `generated` | One direct-answer statement; optional details and warnings |
| `insufficient_evidence` | No written answer; a note explains missing evidence |
| `unavailable` | Generation failed or output was invalid; evidence sufficiency remains undetermined |

- No-match retrieval skips generation. Generation failures preserve retrieved passages.
- Statements cite local IDs such as `S1`; multiple excerpts can come from one label.
- Citation `chunk_id` identifies the anchor; text/offsets describe expanded context. `context_partial` flags incomplete coverage.
- Missing adverse-reactions text does not mean no side effects. The app may suggest a warnings question; filters stay unchanged.
- Structure/source IDs are checked; **factual support needs human review**. The comparison does not evaluate written answers.

## API reference

Interactive reference: **http://127.0.0.1:8000/docs**.

| Endpoint | Access | Purpose |
|---|---|---|
| `POST /auth/signup` | Public | Create account |
| `POST /auth/login` | Public | Receive 60-minute session |
| `GET /auth/me` | Logged in | Check session |
| `POST /auth/logout` | Logged in | Revoke session |
| `POST /search` | Logged in | Retrieve FDA evidence |
| `POST /answer` | Logged in | Retrieve evidence and request a cited answer |
| `GET /health` | Public | Check saved Chroma index and counts |

- Signup/login accept `email` and `password`.
- Protected requests use `Authorization: Bearer <session_token>`.
- Search: nonblank `query` up to 2,000 characters; `k` from 1–20, default 5.
- Answer: same inputs plus `audience`: `patient` (default) or `clinician`.
- Responses include evidence, scores, filters and provenance; answer mode adds statements, citations and status fields.
- Model text is displayed literally. Auth/search/answer responses use `Cache-Control: no-store`.

| HTTP status | Meaning |
|---|---|
| `200` | Valid search/answer, including no-match or unavailable writing |
| `201` / `204` | Account created / logout successful |
| `401` / `409` | Invalid session / duplicate email |
| `422` / `429` | Invalid input / authentication attempt limit |
| `503` | Retrieval service or required embedding provider unavailable |

## Accounts and connection settings

- Accounts: `data/auth/users.sqlite3`; Argon2 password hashes and SHA-256 session-token hashes.
- Passwords: 8–128 characters. Sessions: 60 minutes; logout revokes the current session.
- Streamlit holds tokens server-side and rechecks sessions. Refreshing may require login.
- Signup/login: 10 attempts/IP/10 minutes; in-memory limiter, one backend worker.
- Local MVP: no email verification, password reset or recovery. See [database details](database-schema.md#accounts-and-sessions).

| Environment variable | Purpose / default |
|---|---|
| `OPENAI_API_KEY` | New OpenAI embeddings and answer writing |
| `DRUGASSIST_API_URL` | Backend URL: `http://127.0.0.1:8000` |
| `DRUGASSIST_REQUEST_TIMEOUT` | Frontend request timeout: 120 seconds |
| `DRUGASSIST_AUTH_DB` | Override account database path |

## Terminal search

```powershell
# Word-based search: no embedding API call
uv run --locked --no-sync fda-rag search "Find apixaban ingredient information" --method bm25 --k 3

# Interactive search
uv run --locked --no-sync fda-rag search
```

- Methods: `keyword`, `bm25`, `dense`, `hybrid_rrf`, `hybrid_normalized`, `hybrid_rerank`.
- Default: `api.method` (`hybrid_rerank`); local reranker assets required.
- `--method selected`, or `--run` without a method, uses the saved descriptor's default.
- Exact filters combine: `--section`, `--set-id`, `--document-id`.
- Add `--json` for structured output; enter `/quit` to exit interactive mode.
- CLI search bypasses app authentication, parsing, scope checks and the relevance cutoff.

For maintenance and tests, see [Research](research.md#maintain-and-check-the-project).

Provider references: [Responses](https://developers.openai.com/api/docs/guides/text), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [model](https://developers.openai.com/api/docs/models/gpt-5-mini).
