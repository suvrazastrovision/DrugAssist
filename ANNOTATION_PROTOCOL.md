# Review the expected search results

The 80-question benchmark contains draft judgments. A matching phrase or a high search score does not prove that a passage answers the whole question.

1. Keep the saved dataset, questions and development/test groups fixed while reviewing. Read the difficulty definitions in `data/benchmarks/benchmark.json`.
2. A qualified biomedical or regulatory reviewer should read the full saved FDA sections. Check the drug, formulation, label date, units, conditions and safety details.
3. Search for all plausible supporting sections, including candidates from every method. Hide method names and scores during review. Grade each section **2** (fully useful), **1** (partly useful), or **0** (not useful). An unreviewed section is not automatically irrelevant.
4. Record grades in each question's `qrels`, using `<document_id>:<section>` as the key. For useful sections, keep the exact quoted evidence, character positions, document and label IDs, source hashes, date and URL in `evidence`. Use the saved text: a live URL may show a newer label.
5. For questions without support, check the collection independently. Use `answerable=false`, no positive grades, and a clear `unsupported_reason`. Failure to find an answer in the first ten results is not enough.
6. Have a second reviewer check the judgments where possible. Record actual disagreements and how they were resolved. Never invent reviewer identities, approvals or agreement statistics.
7. For completed reviews, set `judgment_status=reviewed`, the real `reviewer`, an ISO `reviewed_at` timestamp, and meaningful `review_notes`. Set `judgments_exhaustive` accurately. Once all questions are reviewed, set the benchmark `status=reviewed`, update its version/content ID, and preserve the original draft.
8. Validate and evaluate:

```powershell
uv run --locked fda-rag verify-benchmark
uv run --locked fda-rag benchmark
```

The CLI checks recorded review information and source consistency; it cannot certify reviewer expertise. For a stronger final claim, use a new independently reviewed test set that has not already influenced decisions. The expanded 60-drug collection needs evidence bound to its own saved dataset before evaluation.
