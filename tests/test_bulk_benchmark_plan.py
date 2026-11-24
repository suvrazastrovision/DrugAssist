"""Benchmark selection stays deterministic, stratified and bounded to whole labels."""

from scripts.prepare_bulk_benchmark import select_samples


def test_samples_are_nested_stratified_and_independent_of_input_order():
    rows = [{"set_id": f"set-{n}", "stratum": n % 4 + 1,
             "passages": 10 + n, "embedding_tokens": (10 + n) * 270} for n in range(40)]
    samples = select_samples(rows, (100, 500))
    assert samples == select_samples(list(reversed(rows)), (100, 500))
    small, large = samples
    assert small["strata_represented"] == [1, 2, 3, 4]
    assert small["actual_passages"] >= 100
    assert large["actual_passages"] >= 500
    assert large["records"][:small["label_sets"]] == small["records"]
    assert large["embedding_input_tokens_upper_bound"] == sum(
        row["embedding_tokens"] for row in large["records"]
    )


def test_tiny_corpus_is_exhausted_without_duplication():
    rows = [{"set_id": "one", "stratum": 1, "passages": 7, "embedding_tokens": 123}]
    for sample in select_samples(rows):
        assert sample["actual_passages"] == 7
        assert sample["label_sets"] == 1
    assert select_samples([])[0]["records"] == []
