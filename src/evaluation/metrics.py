"""Standard ranked IR metrics with explicit cutoffs and graded relevance.

Binary relevance uses grade >= 1; nDCG uses gains 2**grade - 1. Duplicate
evaluation units are collapsed before scoring. Unsupported queries return None
for effectiveness metrics and are evaluated separately for evidence rejection.
"""

import math


def unique_ranking(ranking):
    return list(dict.fromkeys(ranking))


def retrieval_metrics(ranking, judgments, ks=(1, 3, 5, 10)):
    ranking = unique_ranking(ranking)
    relevant = {key for key, grade in judgments.items() if grade > 0}
    output = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        if not relevant:
            output.update(
                {f"{name}@{k}": None for name in ("precision", "recall", "hit_rate", "mrr", "map", "ndcg")}
            )
            continue
        gains = [judgments.get(key, 0) for key in ranking[:k]]
        hits = [int(g > 0) for g in gains]
        output[f"precision@{k}"] = sum(hits) / k
        output[f"recall@{k}"] = sum(hits) / len(relevant)
        output[f"hit_rate@{k}"] = float(any(hits))
        output[f"mrr@{k}"] = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
        output[f"map@{k}"] = sum(sum(hits[: i + 1]) / (i + 1) for i, hit in enumerate(hits) if hit) / min(
            k, len(relevant)
        )
        dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(gains))
        ideal = sorted(judgments.values(), reverse=True)[:k]
        idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
        output[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return output
