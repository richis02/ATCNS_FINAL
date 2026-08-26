"""Paired bootstrap confidence intervals over per-image metric deltas."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    deltas: np.ndarray,
    n_reps: int,
    seed: int,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict:
    """deltas: a per-image paired difference (e.g. metric_compressed - metric_original).

    ``statistic`` is whatever summary the caller reports -- np.mean for the classic paired
    bootstrap, np.median when the deltas are visibly non-normal (which they are here: many
    images produce exactly 0, so the distribution has a spike and heavy tails).

    Some metrics (e.g. MCC) are NaN for an image with a degenerate confusion matrix
    (no predicted-positive pixels). Those pairs carry no signal and are dropped before
    bootstrapping -- one NaN must not blank out the whole CI. n_paired_images reflects
    only the non-degenerate pairs actually used.
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    n = len(deltas)
    if n == 0:
        return {
            "n_paired_images": 0,
            "point_estimate": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "interpretation": "undefined (all pairs degenerate)",
        }

    rng = np.random.default_rng(seed)
    resamples = np.empty(n_reps)
    for i in range(n_reps):
        resamples[i] = statistic(deltas[rng.integers(0, n, size=n)])

    ci_low, ci_high = np.quantile(resamples, [alpha / 2, 1 - alpha / 2])
    if ci_low > 0:
        interpretation = "positive (compressed above reference)"
    elif ci_high < 0:
        interpretation = "negative (compressed below reference)"
    else:
        interpretation = "inconclusive (CI straddles 0)"

    return {
        "n_paired_images": n,
        "point_estimate": float(statistic(deltas)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "interpretation": interpretation,
    }


def bootstrap_mean_ci(deltas: np.ndarray, n_reps: int, seed: int, alpha: float = 0.05) -> dict:
    """Back-compatible wrapper (fusion.py / window_strata.py / old tests call this)."""
    result = bootstrap_ci(deltas, n_reps, seed, alpha, statistic=np.mean)
    result["mean_delta"] = result.pop("point_estimate")
    return result
