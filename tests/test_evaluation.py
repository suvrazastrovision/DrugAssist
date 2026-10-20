import math

import pytest

from src.evaluation.benchmarking import tune_abstention
from src.evaluation.metrics import retrieval_metrics
from src.evaluation.statistics import holm_adjust, paired_comparison


def test_metrics_against_hand_calculation():
    result = retrieval_metrics(["x", "a", "b"], {"a": 2, "b": 1}, [3])
    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["recall@3"] == 1
    assert result["mrr@3"] == 0.5
    assert result["map@3"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert result["ndcg@3"] == pytest.approx((3 / math.log2(3) + 0.5) / (3 + 1 / math.log2(3)))


def test_metrics_collapse_duplicate_units():
    assert retrieval_metrics(["a", "a", "a"], {"a": 2}, [3])["precision@3"] == pytest.approx(1 / 3)


def test_unsupported_is_excluded_from_recall():
    assert all(v is None for v in retrieval_metrics(["x"], {}).values())


def test_empty_results_and_missing_judgments():
    assert retrieval_metrics([], {"a": 2}, [1])["ndcg@1"] == 0
    assert retrieval_metrics(["unknown"], {"a": 2}, [1])["precision@1"] == 0


def test_abstention_calibration_only_uses_passed_dev_rows():
    rows = [{"answerable": True, "top_score": 0.8}, {"answerable": False, "top_score": 0.2}]
    tuned = tune_abstention(rows)
    assert tuned["balanced_accuracy"] == 1 and tuned["threshold"] == 0.8


def test_paired_bootstrap_reproducibility_and_effect():
    stats = paired_comparison([1, 1, 1, 1], [0, 0, 0, 0], ["a", "a", "b", "b"], samples=100, permutations=200)
    assert stats == paired_comparison(
        [1, 1, 1, 1], [0, 0, 0, 0], ["a", "a", "b", "b"], samples=100, permutations=200
    )
    assert stats["mean_difference"] == stats["ci_low"] == stats["ci_high"] == 1
    assert stats["clusters"] == 2


def test_holm_family_adjustment():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
