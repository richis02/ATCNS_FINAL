"""Auxiliary analysis: does CASIA2 itself run out of unique donors/hosts before it runs out of
images? On the Phase 1 manifest (1832 eligible images), host_id/donor_id uniqueness as a function
of sample size -- classical rarefaction (repeated subsampling WITHOUT replacement), not bootstrap.

1832, not the 1833 of the classical lineage's own audit: Phase 1 adds a GEOMETRIC_MISMATCH check
that run did not have, and it excludes exactly one more image. Already a declared finding, not a
drift -- results/<DATASET_RUN_ID>/dataset_reconciliation.csv carries the delta line by line.

Answers a different question from EXPERIMENT_LOG.md 33.2's donor_leakage_fraction jump (45.9% at
n=500 -> 52.9% at n=800): that number is a property of ONE particular split's calibration/test
assignment. This curve is a property of the manifest alone, no split involved, and is the context
that makes the jump readable as "CASIA2's donor pool is thin", or not.

Zero training, zero GPU. Reads the pinned Phase 1 manifest through phase2_common (metadata only --
host_id/donor_id columns -- never an image), same resolver every other Phase 2/3 module uses, so a
missing/moved manifest raises instead of silently falling back (CLAUDE.md 5.11).

Run in .venv (no torch needed):
    python compute_rarefaction_curve.py
"""
from __future__ import annotations

import json
import sys

sys.path[:0] = ["src", "configs"]

import numpy as np
import pandas as pd

import params
import phase2_common as common


def rarefaction_sweep(frame: pd.DataFrame, sizes: list[int], n_reps: int, seed: int) -> pd.DataFrame:
    """host_id / donor_id uniqueness vs. sample size.

    At each n in `sizes`: n_reps subsamples of n rows WITHOUT replacement, mean + 95% percentile
    interval of unique host_id/donor_id counts over the reps. Deliberately NOT stats.bootstrap_ci
    (resampling WITH replacement from a fixed sample): rarefaction asks what a SMALLER sample
    would have shown, bootstrap asks how uncertain a statistic of the GIVEN sample is -- different
    questions, so this is not forced into that function's signature.
    """
    rng = np.random.default_rng(seed)
    hosts = frame["host_id"].to_numpy()
    donors = frame["donor_id"].to_numpy()
    n_total = len(frame)
    rows = []
    for n in sizes:
        if n > n_total:
            raise ValueError("sample size {} exceeds the {} available rows".format(n, n_total))
        host_counts = np.empty(n_reps)
        donor_counts = np.empty(n_reps)
        for rep in range(n_reps):
            idx = rng.choice(n_total, size=n, replace=False)
            host_counts[rep] = len(np.unique(hosts[idx]))
            donor_counts[rep] = len(np.unique(donors[idx]))
        h_low, h_high = np.quantile(host_counts, [0.025, 0.975])
        d_low, d_high = np.quantile(donor_counts, [0.025, 0.975])
        rows.append({
            "n": n,
            "host_unique_mean": float(host_counts.mean()),
            "host_ci95_low": float(h_low), "host_ci95_high": float(h_high),
            "donor_unique_mean": float(donor_counts.mean()),
            "donor_ci95_low": float(d_low), "donor_ci95_high": float(d_high),
        })
    return pd.DataFrame(rows)


def main() -> None:
    frame = common.manifest()  # the 1832 eligible rows, host_id/donor_id already columns
    n_total = len(frame)
    # Fine grid every 50, plus the sample sizes already cited elsewhere in the project (349 frozen
    # test, 500/800 classical-lineage selections, 1082/453 learned pool A/B) so the curve can be
    # read directly against those numbers.
    markers = {349, 500, 800, 1082, 453, n_total}
    grid = sorted({n for n in range(50, n_total, 50)} | markers)
    grid = [n for n in grid if 0 < n <= n_total]
    n_reps = 200
    seed = params.RNG_SEED

    result = rarefaction_sweep(frame, grid, n_reps, seed)
    result.to_csv(params.RESULTS_DIR / "rarefaction_curve.csv", index=False)

    metadata = {
        "n_total": n_total,
        "total_unique_hosts": int(frame["host_id"].nunique()),
        "total_unique_donors": int(frame["donor_id"].nunique()),
        "n_reps": n_reps,
        "seed": seed,
        "grid": grid,
        "markers": sorted(markers),
        "sampling": "without replacement per size (classical rarefaction), not bootstrap",
    }
    (params.RESULTS_DIR / "rarefaction_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("wrote rarefaction_curve.csv + rarefaction_metadata.json to {}".format(params.RESULTS_DIR))
    print("n_total={}  unique_hosts={}  unique_donors={}".format(
        n_total, metadata["total_unique_hosts"], metadata["total_unique_donors"]))


if __name__ == "__main__":
    main()
