"""Qualitative panels for the Tp_S_ copy-move probe (EXPERIMENT_LOG.md 39/40): RGB / ground truth /
prediction / TP-FP-FN confusion overlay, for a handful of DECLARED-RULE examples.

RENDERING ONLY, mirroring notebooks/make_failure_panels.py's own disclaimer: no statistic here is
new. tp/fp/fn/raw_abs_separability are read straight from tp_s_per_image.csv (the frozen probe's
own output); this script re-runs the same frozen scales/thresholds (baseline.frozen_config(), no
RNG, deterministic) purely to draw the prediction mask, which is bit-identical to what the probe
already scored.

CLAUDE.md 5.10: qualitative figures are not proof. The four panels are chosen by a rule fixed
BEFORE looking at any image (median / best / worst / a silent case), not curated for a flattering
result, and a category with no candidate is skipped and reported, never back-filled.

Run:  .venv/bin/python tp_s_qualitative_examples.py [--run-dir results/20260821-121059]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = ["src", "configs"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import params
import detector_baseline as baseline
from tp_s_copy_move_probe import load_pair

CONFUSION_COLORS = {"tp": (0, 200, 0), "fp": (220, 0, 0), "fn": (0, 90, 255)}
ALPHA = 0.55


def confusion_overlay(rgb: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.astype(np.float64).copy()
    for key, (region, color) in {
        "tp": (prediction & mask, CONFUSION_COLORS["tp"]),
        "fp": (prediction & ~mask, CONFUSION_COLORS["fp"]),
        "fn": (~prediction & mask, CONFUSION_COLORS["fn"]),
    }.items():
        overlay[region] = (1 - ALPHA) * overlay[region] + ALPHA * np.array(color)
    return overlay.clip(0, 255).astype(np.uint8)


def select_examples(per_image: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    """Rule fixed before looking at any image. A category with no candidate is skipped, not
    back-filled with a substitute (CLAUDE.md 5.10)."""
    fired = per_image[per_image["fp"] + per_image["tp"] > 0]
    silent = per_image[per_image["silent"]]
    chosen = {}
    if len(fired):
        by_sep = fired.sort_values("raw_abs_separability")
        chosen["typical (median lr_plus among firing images)"] = (
            "median lr_plus", fired.iloc[(fired["lr_plus"].rank(pct=True) - 0.5).abs().idxmin()]
            if fired["lr_plus"].notna().any() else fired.iloc[len(fired) // 2]
        )
        chosen["strongest separation"] = ("max raw_abs_separability", by_sep.iloc[-1])
        chosen["weakest separation (still firing)"] = ("min raw_abs_separability among firing images", by_sep.iloc[0])
    if len(silent):
        chosen["silent (mode A, n_predicted == 0)"] = (
            "first silent image by image_id", silent.sort_values("image_id").iloc[0]
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(params.RESULTS_ROOT / "20260821-121059"))
    parser.add_argument("--out", default=None, help="output PNG path (default: <run-dir>/tp_s_qualitative_examples.png)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    per_image_path, pairing_path = run_dir / "tp_s_per_image.csv", run_dir / "tp_s_pairing.csv"
    for path in (per_image_path, pairing_path):
        if not path.exists():
            raise FileNotFoundError("{} missing: nothing is substituted (CLAUDE.md 5.11).".format(path))

    per_image = pd.read_csv(per_image_path)
    pairing = pd.read_csv(pairing_path).set_index("image_id")

    categories = select_examples(per_image)
    if not categories:
        raise ValueError("no candidate image in any category -- nothing to render")

    frozen = baseline.frozen_config()
    print("re-scoring with the frozen configuration verbatim (results/{}, w=64 two_sided per_tail)".format(
        params.FROZEN_RUN_ID))

    fig, axes = plt.subplots(4, len(categories), figsize=(4.2 * len(categories), 15.5), squeeze=False)
    for column, (label, (rule, row)) in enumerate(categories.items()):
        record = pairing.loc[row["image_id"]]
        rgb, mask = load_pair(record["image_path"], record["mask_path"])
        maps = baseline.score_maps(rgb, frozen["scales"], frozen["threshold_low"], frozen["threshold_high"])
        prediction = maps["binary"]

        axes[0, column].imshow(rgb)
        axes[0, column].set_title("{}\n{}".format(label, row["image_id"][:28]), fontsize=7)
        axes[1, column].imshow(mask, cmap="gray")
        axes[1, column].set_title("ground truth (pasted region only)", fontsize=7)
        axes[2, column].imshow(prediction, cmap="gray")
        axes[2, column].set_title("prediction (frozen classical, verbatim)", fontsize=7)
        axes[3, column].imshow(confusion_overlay(rgb, prediction, mask))
        axes[3, column].set_title(
            "TP green / FP red / FN blue\ntp={} fp={} fn={}  abs_sep={:.3f}".format(
                int(row["tp"]), int(row["fp"]), int(row["fn"]), row["raw_abs_separability"]),
            fontsize=7,
        )
        print("  {:42s} rule: {:42s} -> {}".format(label, rule, row["image_id"]))

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        "Tp_S_ copy-move probe (EXPERIMENT_LOG.md 39/40) -- frozen classical detector, uncurated "
        "examples\nrows: image / ground truth / prediction / confusion overlay. tp/fp/fn/abs_sep "
        "read from the frozen per-image CSV.",
        fontsize=9,
    )
    plt.tight_layout()

    out_path = Path(args.out) if args.out else run_dir / "tp_s_qualitative_examples.png"
    fig.savefig(out_path, dpi=140)
    print("\nwritten to {}".format(out_path))


if __name__ == "__main__":
    main()
