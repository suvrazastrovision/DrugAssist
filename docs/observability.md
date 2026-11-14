# Optional observability

[README](../README.md) | [Setup](setup.md) | [Usage](usage.md) | [Tests](testing.md) | [Troubleshooting](troubleshooting.md)

Stop the API before installing the optional tracing dependency:

```sh
uv sync --locked --inexact --extra observability
```

Create a Langfuse Cloud project and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_BASE_URL` in `.env` (EU: `https://cloud.langfuse.com`). Set
`DRUGASSIST_OBSERVABILITY_ENABLED=true` and restart the API. To disable tracing,
set it to `false` and restart. The default is disabled.

Search produces a retrieval trace; answers group retrieval and any OpenAI answer call
under one trace. Recorded fields include timings, retrieval method, requested/result
counts, audience, no-match reason, answer status, and generation model/token usage.
Questions, answers, passages, filters, account identifiers and raw errors are excluded.
Auth and health requests are not traced. Telemetry is exported in the background and
drained on API shutdown; missing configuration, a missing SDK, or telemetry failures
do not prevent search or answers. The initial integration covers these service calls,
not embedding usage, ingestion, evaluation judging or the CLI search pipeline.

## Verify tracing

1. Restart the API after configuring the environment.
2. Submit a question in the app with **Write a cited answer** enabled.
3. Open your Langfuse project's **Tracing** view and refresh after a few seconds.
4. Open the `answer` trace. It should contain `retrieval` and, when the model is called, `generation` observations.

Inspect durations, result counts, answer status, model and token usage. Question and answer
text remain empty by design. An insufficient-evidence response can skip generation entirely.

If traces are missing, follow the [troubleshooting guide](troubleshooting.md#langfuse-traces-are-missing).
