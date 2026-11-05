# App evaluation and autograding

[README](../README.md) | [Research](research.md) | [Usage](usage.md)

An **evaluation** runs test cases, grades outputs and compares versions. An **autograder**
computes the grades: Python checks retrieval and citations; an optional model judge checks
factual support, completeness, relevance and abstention. Valid citations do not prove correctness.

This evaluates the same services as `/search` and `/answer`, including parsing, filters,
score cutoffs and answer generation. App settings, endpoints and document vectors are unchanged.
No additional framework or service is required.

## 1. Prepare cases once

Run from the repository root:

```powershell
uv run --locked --no-sync fda-rag evaluate prepare
```

This verifies saved sources offline and creates `results/private/evaluation/dataset.json`
with **30 draft cases**: 10 existing complex challenges, 15 focused questions sharing their
evidence, and 5 negative cases for unsupported topics, fictitious drugs and unavailable information.

Cases pin the snapshot, document versions, source hashes, exact quotes and offsets. Each fact
can have alternative evidence occurrences; one must be covered fully. Duplicate passages earn
no extra credit. These are diagnostic cases, not independent clinical assessments or an untouched
holdout. They start in `development` or `regression` and need human review.

Reuse the file after preparation. For another dataset, use `--output PATH.json`; `--snapshot ID`
selects another saved snapshot. Missing required source anchors cause preparation to fail.

## 2. Run the app pipeline

Start with five development cases:

```powershell
uv run --locked --no-sync fda-rag evaluate run --split development --limit 5 --allow-api --output results/private/evaluation/search-baseline.json
```

The default is `search` with the configured retrieval method and `k=5`. Omit `--split` and
`--limit` for all cases; use `--k` to change the result count. Prepared index/model assets and
the configured API key are required. Services run directly, without an account/session token;
endpoint authentication remains unchanged.

**Costs:** `--allow-api` is required because uncached query embeddings and generated answers may
incur charges. Document embeddings are not rebuilt.

Collect answers with structural checks:

```powershell
uv run --locked --no-sync fda-rag evaluate run --mode answer --split development --limit 5 --allow-api --output results/private/evaluation/answers.json
```

Generated answers remain `review_required` until semantic grading. Correct expected abstentions
can pass without a judge because they make no answer claims.

To collect and judge together, replace `JUDGE_MODEL` with a supported model you can access,
preferably a pinned snapshot calibrated against human judgments:

```powershell
uv run --locked --no-sync fda-rag evaluate run --mode answer --split development --limit 5 --allow-api --judge-model JUDGE_MODEL --output results/private/evaluation/answer-baseline.json
```

The judge uses `OPENAI_API_KEY`, structured output, `store=false`, token/time limits and no
automatic retries. It treats inputs as untrusted and checks statements against their own cited
windows, keeping expected facts separate. Supported statements require exact source quotes.
Missing grades, invented quotes, incomplete responses and refusals are errors. Structured output
checks format, not truth; see the [official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

## 3. Regrade saved answers

```powershell
uv run --locked --no-sync fda-rag evaluate grade --responses results/private/evaluation/answers.json --output results/private/evaluation/regraded.json
```

Deterministic regrading is offline and needs the matching saved source snapshot. It does not
initialize Chroma, embedding models or the answer writer. Add `--judge-model JUDGE_MODEL
--allow-api` for semantic grading without regenerating answers or query embeddings.

Valid judgments are cached by full input, model settings and grading instructions in
`results/private/evaluation/judge_cache/`; errors are not cached. Only uncached judgments incur
new model calls. Rolling model aliases retain their cached judgments; use a pinned or new model
ID for a fresh assessment.

Reports separate captured embedding/answer tokens from new judge usage. On replay, captured
usage belongs to the original run; cached judgments are not new usage. Token counts are not
estimated EUR charges and may omit usage from failed requests.

## 4. Understand reports

Commands save JSON and Markdown reports, including per-case failures, source/code fingerprints,
index/settings, prompts, requested/resolved judge models, package versions, latency and usage.
Latency includes query embedding and generation, but excludes index/model loading.

| Status | Meaning |
|---|---|
| `passed` | All applicable checks passed; check whether semantic grading ran |
| `failed` | An output or evidence requirement failed |
| `error` | App/judge unavailable or judgment invalid; not a quality judgment |
| `review_required` | Structural checks passed; generated answer needs semantic assessment |

| Measurement | Meaning |
|---|---|
| Evidence hit rate at k | Answerable cases with any required evidence retrieved |
| Mean evidence coverage | Average fraction of required facts whose full spans were retrieved |
| Complete evidence rate at k | Answerable cases with all required evidence retrieved |
| Mean reciprocal rank | Average inverse rank where the first full requirement is covered |
| Cited evidence coverage | Coverage in expanded citation windows, separate from retrieved chunks |
| Semantic checks | Statement support, completeness, relevance and appropriate abstention |
| Critical errors | Reversed warnings, dose/unit errors, changed qualifications or blended products |

Every authored fact must be covered for its deterministic evidence check to pass. Answerable
cases with app/schema errors count as zero retrieval coverage; negative cases are excluded
from coverage averages. Exact spans can miss equivalent evidence, so review failures.

Completed cases are checkpointed. Interrupted runs stay `running` and cannot be compared or
regraded as complete runs. Commands refuse to overwrite datasets/reports; choose a new `.json`
output path. Exit codes: **0** completed checks, **1** failures/errors, **2** invalid input or
pending semantic review. An automated pass does not certify medical correctness.

## 5. Compare versions

Collect a candidate using the same dataset, selected cases, mode, k and judge configuration:

```powershell
uv run --locked --no-sync fda-rag evaluate compare results/private/evaluation/answer-baseline.json results/private/evaluation/answer-candidate.json
```

Comparison rejects mismatched snapshots, datasets, cases, modes, k, grading code or judge
settings/resolved models. App implementation/settings may differ and are recorded. Lost passes,
new failed checks, lower evidence/semantic scores and additional critical errors are regressions.
Comparison exits **1** for regressions/current service errors, or **2** for pending semantic review.

## 6. Review before release

1. Have a person verify questions, source versions, expected facts and acceptable evidence;
   medical accuracy/safety cases need appropriate domain expertise.
2. Blindly grade sample answers, including omissions, wrong units/negation/formulations and
   appropriate/unnecessary refusals. Compare human and model decisions by criterion, recording
   disagreements and false passes. Self-grading models may share the answer model's blind spots.
3. Tune the rubric on development cases, then freeze it. Rubric changes require regrading;
   different grader fingerprints cannot be compared directly.
4. Add independently authored, reviewed `holdout` cases, preferably from different drug/product
   groups. Keep them separate from tuning and existing challenges. Run with `--dataset PATH
   --split holdout`. Set `review_status` to `human_reviewed` only after actual review, then create
   new baselines because the dataset hash changes.

CI runs offline engineering tests without paid calls. Start live evaluations manually and use
model grades as release gates only after calibration. Keep regression snapshots frozen; assess
live-source changes with a new dataset/snapshot. The notebook benchmark remains a separate,
unfiltered retrieval comparison.

Evaluation artifacts, including JSON and Markdown reports, are saved in the Git-ignored `results/private/evaluation/` directory by default; use `--output PATH.json` to choose another report location and exclude private user questions from shared reports.
