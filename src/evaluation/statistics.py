"""Paired, drug-cluster bootstrap and sign-flip randomization comparisons."""

import numpy as np


def paired_comparison(left, right, groups=None, seed=42, samples=2000, permutations=10000):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not len(left):
        raise ValueError("Paired nonempty arrays required")
    difference = left - right
    groups = list(range(len(difference))) if groups is None else list(groups)
    if len(groups) != len(difference):
        raise ValueError("Each pair needs a cluster ID")
    clusters = list(dict.fromkeys(groups))
    values = [difference[np.array([g == cluster for g in groups])] for cluster in clusters]
    rng = np.random.default_rng(seed)
    estimates = [
        np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))]).mean()
        for _ in range(samples)
    ]
    observed = float(difference.mean())
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice([-1, 1], len(values))
        simulated = np.concatenate([value * sign for value, sign in zip(values, signs, strict=True)]).mean()
        extreme += abs(simulated) >= abs(observed) - 1e-15
    return {
        "mean_difference": observed,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_value": (extreme + 1) / (permutations + 1),
        "clusters": len(clusters),
        "pairs": len(difference),
        "seed": seed,
        "procedure": "paired drug-cluster bootstrap; two-sided cluster sign-flip test",
    }


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted, previous = [0.0] * len(p_values), 0.0
    for i, index in enumerate(order):
        previous = max(previous, min(1.0, float(p_values[index]) * (len(p_values) - i)))
        adjusted[int(index)] = previous
    return adjusted
