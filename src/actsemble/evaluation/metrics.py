"""Success-rate statistics: Wilson intervals and paired bootstrap."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_diff(
    successes_a: list[bool],
    successes_b: list[bool],
    *,
    num_resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap CI for mean(a) - mean(b) resampling PAIRED episodes."""
    a = np.asarray(successes_a, dtype=np.float64)
    b = np.asarray(successes_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired bootstrap needs equal-length 1D outcome lists")
    n = len(a)
    if n == 0:
        raise ValueError("no episodes")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(num_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean_difference": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1 - alpha)),
        "confidence": confidence,
        "num_resamples": num_resamples,
        "num_pairs": n,
    }


def paired_outcome_counts(successes_a: list[bool], successes_b: list[bool]) -> dict:
    """Per-seed win/loss/tie counts for system a versus system b."""
    a = np.asarray(successes_a, dtype=bool)
    b = np.asarray(successes_b, dtype=bool)
    return {
        "a_wins": int(np.sum(a & ~b)),
        "b_wins": int(np.sum(~a & b)),
        "both_succeed": int(np.sum(a & b)),
        "both_fail": int(np.sum(~a & ~b)),
    }
