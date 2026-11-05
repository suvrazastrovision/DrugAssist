# App evaluation and autograding

[README](../README.md) | [Research](research.md) | [Usage](usage.md)

An **evaluation** is a dataset, an app run, its grades, and a comparison with previous runs.
An **autograder** computes those grades. Python checks grade retrieval and citation integrity;
an optional model judge assesses factual support, completeness, relevance and abstention.
Valid citation IDs alone do not establish factual support or medical correctness.

The evaluation command calls the same search and answer services as `/search` and `/answer`.
It includes the app's parser, filters, score cutoff, context construction and answer writer.
It does not change endpoints, prompts, chunking, embedding settings, retrieval settings or the
active collection. New query embeddings can incur charges; document vectors are not rebuilt.
No evaluation framework or additional service is required.

## 1. Prepare a frozen dataset once

From the repository root:

```powershell
uv run --locked --no-sync fda-rag evaluate prepare
```

This is offline: it verifies the active snapshot's saved sources and creates
`results/private/evaluation/dataset.json`. The file contains 30 **draft diagnostic cases**:

- 10 existing complex retrieval challenges, also used in the original benchmark.
- 15 focused questions derived from their evidence requirements.
- 5 negative questions covering unsupported topics, fictitious drugs and unavailable personal information.

The dataset pins the snapshot, document versions, source hashes, exact quotes and character
offsets. Each expected fact can have several acceptable evidence occurrences. A complete
occurrence must be covered; duplicate retrieved passages do not earn additional credit.

These are not 30 independent clinical assessments. Focused questions share evidence with
the complex challenges. Every case starts in `development` or `regression`; no case is
claimed to be an untouched holdout. Reference descriptions and source quotes require human
review before using their scores to make release decisions.

Preparation refuses to overwrite a file. Reuse the prepared file, or create another with
`--output results/private/evaluation/dataset-v2.json`. Use `--snapshot SNAPSHOT_ID` to select
another saved snapshot; preparation fails when its required source anchors are missing.

## 2. Collect a small retrieval run

```powershell
uv run --locked --no-sync fda-rag evaluate run --split development --limit 5 --allow-api --output results/private/evaluation/search-baseline.json
```

The default mode is `search`, with the app's configured retrieval method and `k=5`.
`--allow-api` is required because an uncached question embedding may call the configured
provider. An existing API key and prepared index/model assets are needed, as when running the app.
No account or session token is needed: this command invokes the services directly, while
normal endpoint authentication remains unchanged.

Inspect the JSON or adjacent Markdown report. To evaluate all 30 cases, omit `--split` and
`--limit`. Use the same dataset, selection and `--k` when comparing runs. Loading models and
the index is excluded from per-question latency; question embedding and generation are included.

## 3. Collect answers and optionally grade their meaning

To collect answers with deterministic checks only:

```powershell
uv run --locked --no-sync fda-rag evaluate run --mode answer --split development --limit 5 --allow-api --output results/private/evaluation/answers.json
```

The answer writer uses the existing app configuration. Generated answers that pass structural
checks remain **review_required** until semantic grading is performed. Correct expected
abstentions can pass deterministic checks because they make no answer claims.

To collect and judge in one run, replace `JUDGE_MODEL` with a supported model ID you have
access to, preferably a pinned snapshot that you have calibrated for this task:

```powershell
uv run --locked --no-sync fda-rag evaluate run --mode answer --split development --limit 5 --allow-api --judge-model JUDGE_MODEL --output results/private/evaluation/answer-baseline.json
```

The judge uses the existing `OPENAI_API_KEY` and the Responses API. It uses structured
output, `store=false`, bounded input/output tokens, a timeout, and no automatic retries.
It treats the question, evidence and answer as untrusted data. It receives the actual cited
windows plus separate expected facts; it cannot count gold references as evidence the
answer actually saw. Supported statements require literal evidence quotes from their own
cited sources. Missing grades, invented quotes, incomplete responses and refusals are errors.

The judge is optional and can be wrong. A model grading itself can share the answer model's
blind spots; compare candidate judges with human labels rather than choosing on cost alone.
See [official OpenAI documentation on structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
for the API contract. Structured output validates shape, not the truth of a judgment.

## 4. Regrade saved answers without generating them again

Deterministic regrading is offline and does not initialize Chroma, embedding models, or the
answer writer. It requires the matching saved source snapshot for source verification:

```powershell
uv run --locked --no-sync fda-rag evaluate grade --responses results/private/evaluation/answers.json --output results/private/evaluation/regraded.json
```

Add `--judge-model JUDGE_MODEL --allow-api` to judge the same saved answers. That pays for
uncached judgments only, without regenerating answers or question embeddings. Valid judgments
are cached under `results/private/evaluation/judge_cache/`, keyed by the full input, model
settings and grading instructions. Errors are not cached. If using a rolling model alias,
its existing cache entries stay frozen; use a pinned model or a new model ID for a fresh assessment.

Reports record captured embedding and answer tokens when the provider reports them, and new
judge tokens separately. Captured tokens in a regraded report describe the original collection,
not new billing. Cached judgment usage is not counted as new usage. Token totals are not a euro
estimate and may omit provider usage from failed requests.

## 5. Read the grades

| Grade | Meaning |
|---|---|
| `passed` | All applicable automated checks passed; inspect which semantic checks ran |
| `failed` | An output or evidence requirement failed |
| `error` | The app/judge was unavailable or returned an invalid judgment; no quality inference follows |
| `review_required` | Structural checks passed, but a generated answer has no semantic assessment |

| Measurement | Meaning |
|---|---|
| Evidence hit rate at k | Fraction of answerable cases with any required evidence represented |
| Mean evidence coverage | Mean fraction of required facts whose full evidence spans were retrieved |
| Complete evidence rate at k | Fraction of answerable cases with all required evidence represented |
| Mean reciprocal rank | Average inverse rank at which the first full evidence requirement is covered |
| Cited evidence coverage | Required evidence covered by expanded citation windows, reported separately from retrieved chunks |
| Statement support | Judge checks every claim against its actual citations |
| Completeness | Judge checks every expected fact, including qualifications |
| Relevance and abstention | Judge checks responsiveness and whether answering/refusing was appropriate |
| Critical errors | Explicit flags for reversed warnings, dose/unit errors, changed qualifications or blended products |

Answerable cases with app/schema errors contribute zero retrieval coverage; negative cases
are excluded from evidence-coverage averages. Overall failure and error counts remain separate.
Every authored requirement must be covered for its deterministic evidence check to pass;
there is no hidden arbitrary aggregate accuracy threshold. Source-span grading can miss valid
equivalent evidence, so review failures before extending the accepted references.

Reports record dataset and code fingerprints, source snapshot, index/settings, prompts,
requested/resolved judge model, package versions, latency, usage and per-question responses.
Each completed case is checkpointed. Interrupted runs remain `running` and are not comparable
or accepted for regrading as completed runs. New commands refuse to overwrite earlier reports.

Command exit codes are `0` for completed checks without failures or pending semantic reviews,
`1` for failed/error cases, and `2` for invalid inputs or generated answers still needing review.
A successful automated grade is not a certification of medical correctness.

## 6. Compare a change against a baseline

Collect another run using the same dataset, mode, cases, k, and judge configuration, then:

```powershell
uv run --locked --no-sync fda-rag evaluate compare results/private/evaluation/answer-baseline.json results/private/evaluation/answer-candidate.json
```

Comparison rejects different snapshots, datasets, selected cases, modes, k values, grading
code, judge settings or resolved judge models. Changing app implementation/settings is allowed
and is recorded. New failed checks, lower evidence coverage, lost passes, and additional
critical errors are regressions. The command exits `1` for regressions or current service errors.
Review pending semantic assessments separately; they are not evidence of answer quality.

## 7. Review the dataset and calibrate the judge

1. Have a person review questions, exact source versions, accepted evidence and expected facts.
   Cases involving medical accuracy or safety need appropriate domain expertise.
2. Independently grade a sample of saved answers before looking at the model's grades.
   Include correct answers, missing qualifications, wrong units/negation, wrong formulations,
   appropriate abstentions and unnecessary refusals. Deliberately altered answers can supplement
   real outputs; identify them as synthetic diagnostic cases.
3. Compare human and model decisions separately for support, completeness, relevance, abstention
   and critical errors. Record disagreements and false passes; do not rely on one average score.
4. Revise the grading rubric on development cases and freeze it. Regrade saved answers after
   rubric changes; reports with different grader fingerprints cannot be directly compared.
5. Add independently authored `holdout` cases to a separate reviewed dataset, preferably using
   different drug/product groups from development. Do not copy these existing challenges into
   a holdout or repeatedly tune against it. Run with `--dataset PATH --split holdout` before release.
6. Set `review_status` to `human_reviewed` only after actual review. This changes the dataset hash;
   create new baselines rather than silently updating previous results.

The existing CI runs the new offline engineering tests with the rest of `pytest`. It does not
make paid calls or claim clinical validation. Start live evaluations manually; enable a model
grade as a release gate only after calibration. Keep regression data frozen. Evaluate changes
to live source data in a separate run with a newly prepared dataset/snapshot.

All datasets, responses and judge caches default to the already Git-ignored `results/private/`
directory. Keep private user questions out of shared reports. The existing notebook benchmark
is unchanged and continues to describe unfiltered retrieval separately from these app evaluations.
