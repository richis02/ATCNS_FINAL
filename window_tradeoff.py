"""The brief's second contribution: window size vs detection accuracy, per compression pipeline.

Re-derives the sliding-window trade-off under the CURRENT frozen classical detector (per_tail
fusion, per-filter scales frozen on calibration, per-image normalization AFTER the fusion,
local_contrast off) -- the configuration of EXPERIMENT_LOG.md 18.6, not the pre-18.3 single-Sobel
one the archived sliding_window_tradeoff.png was measured on.

CALIBRATION ONLY. 151 images x 4 arms x 9 windows. No test image is read, no parameter is chosen
for anything downstream, nothing is re-frozen. Protocol pre-registered in EXPERIMENT_LOG.md 37
BEFORE this ran; the reading rules below are that entry, in code.

What moves and what does not (CLAUDE.md 5.1): the WINDOW is the only axis. Direction stays
"two_sided", the fusion stays per_tail, postprocess stays (0, 0.0), target_fpr stays the
pre-registered 0.10, and ONE threshold per window -- fitted on authentic calibration pixels pooled
across all four arms -- is applied identically to every arm. Re-opening the direction would be a
new candidate axis AND would break the paired comparison.

Primary statistic is micro lr_plus (TPR/FPR): prevalence-invariant and invariant to threshold
aggressiveness. Micro F1 is written next to its OWN realised FPR and positive_ratio and is never
used to order anything -- that is the 11.2 signature, which 36.3 caught for the third time.

NOT a Phase 2 module. It imports phase2_common only for statistical primitives that have no Phase
2 content (cluster bootstrap over images, host-halved optimism gap); the split, the arms and the
detector all come from experiment_scaffold, the classical lineage's own scaffold.

Run in .venv (no torch needed), from the repo root:
    python window_tradeoff.py --smoke     # 2 windows, 12 images -- shakes out code, ~1 min
    python window_tradeoff.py
"""
from __future__ import annotations

import json
import sys

sys.path[:0] = ["src", "configs"]

import numpy as np
import pandas as pd

import experiment_scaffold as scaffold
import params
import phase2_common as common
from forensics import calibration, detector, metrics

# The full trade-off grid. Wider on both ends than the calibration grid ([16..128]): the question
# is where the optimum IS, and an argmax at the edge of its own grid says the optimum may be
# outside it (the reasoning that added 96/128 in iteration 16, applied downwards too).
WINDOWS = [8, 12, 16, 24, 32, 48, 64, 96, 128]

# The frozen geometry (EXPERIMENT_LOG.md 18.6). Literals, not resolved from a run directory, and
# checked against the frozen artefact below -- a resolver that guesses is the 20.4 defect, and a
# literal that silently disagrees with the artefact is its mirror image.
FROZEN_DIRECTION = "two_sided"
FROZEN_RADIUS = 0
FROZEN_MIN_AREA = 0.0
FROZEN_CONTRAST = False
FROZEN_VARIANT = "per_tail"
TARGET_FPR = params.FROZEN_TARGET_FPR  # 0.10, pre-registered
PIXELS_CAP = params.CALIBRATION["calibration_pixels_per_class_per_map"]
FILTER_NAMES = ["laplacian", "log_sigma2", "sobel"]
SCALE_TOLERANCE = 1e-9
THRESHOLD_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------------------------
# Resolvers. CLAUDE.md 5.11: raise on a missing artefact, never substitute a plausible value.


def frozen_artefact(name: str):
    path = params.RESULTS_ROOT / params.FROZEN_RUN_ID / name
    if not path.exists():
        raise FileNotFoundError(
            "{} is missing. This sweep is defined relative to the frozen classical run and "
            "invents no substitute (CLAUDE.md 5.11).".format(path)
        )
    return path


def frozen_metadata() -> dict:
    return json.loads(frozen_artefact("experiment_metadata.json").read_text())


def frozen_selected_images() -> int:
    """The n the frozen split was taken at (500), from the frozen run's own summary.

    Read, never assumed: EXPERIMENT_LOG.md 34.4 left params.DATASET["max_images"] at 800 as the
    default for future exploration of the classical lineage, so build_split()'s default would
    silently produce a DIFFERENT calibration set -- and every number here would then describe a
    detector nobody froze.
    """
    return int(pd.read_csv(frozen_artefact("split_summary.csv"))["selected_images"].iloc[0])


def frozen_calibration_ids() -> set[str]:
    """The 151 calibration image ids of the frozen run, from the grid it actually selected on."""
    return set(pd.read_csv(frozen_artefact("calibration_grid_cv.csv"), usecols=["image_id"])["image_id"])


# ---------------------------------------------------------------------------------------------
# Guards. Each RAISES; a run that trips one is discarded, not patched.


def assert_calibration_set(calibration_ids: list[str]) -> None:
    got, expected = set(calibration_ids), frozen_calibration_ids()
    if got != expected:
        raise ValueError(
            "the calibration set does not match the frozen run: {} images here, {} there, "
            "{} symmetric difference. The split has moved and nothing measured here would be "
            "comparable with the frozen detector.".format(len(got), len(expected), len(got ^ expected))
        )


def assert_geometry_matches_frozen() -> None:
    """The literals above must agree with the frozen run's own record of what it froze."""
    frozen = frozen_metadata()["frozen_configuration"]
    declared = {
        "direction": FROZEN_DIRECTION, "fusion_variant": FROZEN_VARIANT,
        "local_contrast": FROZEN_CONTRAST, "postprocess_radius": FROZEN_RADIUS,
        "postprocess_min_area_frac": FROZEN_MIN_AREA, "target_fpr": TARGET_FPR,
    }
    mismatch = {k: (v, frozen[k]) for k, v in declared.items() if frozen[k] != v}
    if mismatch:
        raise ValueError(
            "declared geometry disagrees with results/{}/experiment_metadata.json: {}".format(
                params.FROZEN_RUN_ID, mismatch)
        )


def assert_filter_scales(scales: dict) -> None:
    """Refit scales must reproduce filter_scales.csv on every window the frozen grid covers.

    This is the proof that the sweep runs the SAME detector the report is about. Windows 8/12/24/48
    have no frozen value and none is invented -- they are simply not checked here.
    """
    frozen = pd.read_csv(frozen_artefact("filter_scales.csv"))
    for window, group in frozen.groupby("window_size"):
        if int(window) not in scales:
            continue
        got = dict(zip(FILTER_NAMES, scales[int(window)]))
        for row in group.itertuples():
            loc, scale = got[row.filter]
            if abs(loc - row.loc) > SCALE_TOLERANCE or abs(scale - row.scale) > SCALE_TOLERANCE:
                raise ValueError(
                    "refit filter scale at w={} {} is ({:.12f}, {:.12f}) but the frozen run "
                    "recorded ({:.12f}, {:.12f}): this is not the frozen detector.".format(
                        window, row.filter, loc, scale, row.loc, row.scale)
                )


def assert_frozen_threshold(window: int, low: float, high: float) -> None:
    """At the frozen window the fitted pair must be the frozen one -- the same operating point.

    Compared against experiment_metadata.json's frozen_configuration, NOT against
    selected_frozen_configuration.csv: that CSV comes out of best_configuration, which aggregates
    the CROSS-VALIDATED grid and reports the MEDIAN of the per-fold threshold fits. The CV chose
    the geometry only; the threshold was then refitted on the whole calibration set pooled over
    all four arms (forensics_pipeline_iter20.py:1030-1033), which is what this sweep reproduces.
    The two differ by 0.072 on the low tail, so comparing against the CV median would fail a
    correct run.
    """
    frozen = frozen_metadata()["frozen_configuration"]
    if window != int(frozen["window_size"]):
        return
    for name, got, want in (("low", low, frozen["threshold_low"]), ("high", high, frozen["threshold_high"])):
        if abs(got - want) > THRESHOLD_TOLERANCE:
            raise ValueError(
                "threshold_{} at w={} is {:.12f} but the frozen run recorded {:.12f}: this sweep "
                "is not at the frozen operating point.".format(name, window, got, want)
            )


def assert_same_grid_and_prevalence(block: pd.DataFrame, window: int) -> None:
    """One pixel grid and one prevalence across the four arms (CLAUDE.md 5.3)."""
    per_arm = block.groupby("arm")[["tp", "fp", "fn", "tn"]].sum()
    total = (per_arm["tp"] + per_arm["fp"] + per_arm["fn"] + per_arm["tn"])
    positives = (per_arm["tp"] + per_arm["fn"])
    if total.nunique() != 1 or positives.nunique() != 1:
        raise ValueError(
            "the arms do not share one pixel grid / prevalence at w={}:\n{}".format(
                window, pd.DataFrame({"pixels": total, "tampered": positives}))
        )


# ---------------------------------------------------------------------------------------------


def score_window(high, low, calibration_ids, arms, window, stratum_of, check_threshold=True):
    """Per-image confusion counts for every arm at one window, at one shared threshold."""
    pool_high = calibration.pool_authentic_scores(
        high, calibration_ids, arms, window, PIXELS_CAP, params.RNG_SEED)
    pool_low = calibration.pool_authentic_scores(
        low, calibration_ids, arms, window, PIXELS_CAP, params.RNG_SEED)
    threshold_low, threshold_high = calibration.threshold_for_target_fpr(
        pool_high, TARGET_FPR, FROZEN_DIRECTION, pool_low)
    if check_threshold:
        assert_frozen_threshold(window, threshold_low, threshold_high)
    del pool_high, pool_low

    rows = []
    for arm in arms:
        for image_id in calibration_ids:
            high_map, mask = high(image_id, arm, window)
            low_map, _ = low(image_id, arm, window)
            prediction = detector.predict_mask(
                high_map, FROZEN_DIRECTION, threshold_low, threshold_high,
                radius=FROZEN_RADIUS, min_area_frac=FROZEN_MIN_AREA, score_map_low=low_map,
            )
            row = metrics.compute_metrics(prediction, mask)
            n_predicted = int(row["tp"] + row["fp"])
            rows.append({
                "window_size": window, "arm": arm, "image_id": image_id,
                "stratum": stratum_of[image_id],
                "tp": int(row["tp"]), "fp": int(row["fp"]),
                "fn": int(row["fn"]), "tn": int(row["tn"]),
                "f1_image": row["f1"],
                # Failure mode A: predicted nothing at all. Mode B: predicted, all of it wrong --
                # the project definition (evaluate_detector.py:147), not a variant of it.
                "silent": n_predicted == 0,
                "off_mask": bool((row["f1"] == 0 or np.isnan(row["f1"])) and n_predicted > 0),
            })
    frame = pd.DataFrame(rows)
    frame.attrs["threshold_low"] = threshold_low
    frame.attrs["threshold_high"] = threshold_high
    return frame


def summarize(per_image: pd.DataFrame, threshold_low: float, threshold_high: float) -> list[dict]:
    """One micro row per (window, arm), with the bootstrap CI on the primary statistic.

    bootstrap_micro_ci resamples IMAGES and recomputes the metric from POOLED counts, with the
    same seed on every cell -- so every arm and every window sees the SAME image resample and the
    comparisons stay paired (CLAUDE.md 5.6: no RNG that differs between the things being compared).
    """
    rows = []
    for (window, arm), block in per_image.groupby(["window_size", "arm"], sort=False):
        micro = metrics.metrics_from_counts(*(int(block[c].sum()) for c in ("tp", "fp", "fn", "tn")))
        lr_ci = common.bootstrap_micro_ci(block, "lr_plus")
        prevalence = micro["prevalence"]
        rows.append({
            "window_size": int(window), "arm": arm, "n_images": len(block),
            "lr_plus": micro["lr_plus"],
            "lr_plus_ci_low": lr_ci["ci95_low"], "lr_plus_ci_high": lr_ci["ci95_high"],
            # Reported ONLY alongside the operating point that produced it. F1 rising while LR+
            # falls is threshold aggressiveness, not localization (11.2, 36.3).
            "f1": micro["f1"], "fpr": micro["fpr"], "positive_ratio": micro["positive_ratio"],
            "precision": micro["precision"], "recall": micro["recall"], "mcc": micro["mcc"],
            "all_positive_f1": 2 * prevalence / (1 + prevalence),
            "n_silent_images": int(block["silent"].sum()),
            "silent_rate": float(block["silent"].mean()),
            "off_mask_images": int(block["off_mask"].sum()),
            "off_mask_rate": float(block["off_mask"].mean()),
            "threshold_low": threshold_low, "threshold_high": threshold_high,
        })
    return rows


def separability(per_image: pd.DataFrame, summary: pd.DataFrame, calibration_ids, arms, frozen_window):
    """The pre-registered reading rule of EXPERIMENT_LOG.md 37.2, applied mechanically.

    1. w* is a RESOLVED optimum only if the PAIRED bootstrap delta lr_plus against the runner-up
       window excludes zero. Paired because both windows are scored on the same images at the same
       prevalence -- an unpaired comparison would throw away exactly the variance that is shared.
    2. Per-pipeline optima DIFFER only if the argmax windows differ, rule 1 holds for the arms
       involved, and the paired arm-vs-arm delta at the frozen window excludes zero.
    3. Otherwise: not resolvable at this sample size. That is the result, not a failed run.
    """
    def cell(window, arm):
        block = per_image[(per_image["window_size"] == window) & (per_image["arm"] == arm)]
        return block.sort_values("image_id").reset_index(drop=True)

    rows = []
    for arm in arms:
        arm_rows = summary[summary["arm"] == arm].sort_values("lr_plus", ascending=False)
        best, runner_up = arm_rows.iloc[0], arm_rows.iloc[1]
        delta = common.bootstrap_micro_delta_ci(
            cell(int(best["window_size"]), arm), cell(int(runner_up["window_size"]), arm), "lr_plus")
        # CLAUDE.md 5.7: the gap of the number actually argmaxed, so lr_plus and not f1.
        gap = common.optimism_gap(
            {int(w): cell(int(w), arm) for w in summary["window_size"].unique()},
            calibration_ids, metric="lr_plus")
        rows.append({
            "arm": arm,
            "argmax_window": int(best["window_size"]), "argmax_lr_plus": best["lr_plus"],
            "runner_up_window": int(runner_up["window_size"]), "runner_up_lr_plus": runner_up["lr_plus"],
            "paired_delta_lr_plus": delta["point_estimate"],
            "delta_ci95_low": delta["ci95_low"], "delta_ci95_high": delta["ci95_high"],
            "optimum_resolved": bool(delta["excludes_zero"]),
            "inner_selected_window": gap["inner_selected"],
            "inner_fit_lr_plus": gap["inner_fit_lr_plus"],
            "inner_check_lr_plus": gap["inner_check_lr_plus"],
            "optimism_gap_lr_plus": gap["optimism_gap_lr_plus"],
            "n_inner_fit": gap["n_inner_fit"], "n_inner_check": gap["n_inner_check"],
        })
    arm_table = pd.DataFrame(rows)

    reference = params.REFERENCE_ARM
    arm_rows = []
    for arm in [a for a in arms if a != reference]:
        delta = common.bootstrap_micro_delta_ci(
            cell(frozen_window, arm), cell(frozen_window, reference), "lr_plus")
        arm_rows.append({
            "window_size": frozen_window, "arm": arm, "versus": reference,
            "paired_delta_lr_plus": delta["point_estimate"],
            "delta_ci95_low": delta["ci95_low"], "delta_ci95_high": delta["ci95_high"],
            "excludes_zero": bool(delta["excludes_zero"]),
        })
    return arm_table, pd.DataFrame(arm_rows)


def verdict(arm_table: pd.DataFrame, arm_vs_arm: pd.DataFrame) -> str:
    resolved = arm_table[arm_table["optimum_resolved"]]
    distinct_optima = arm_table["argmax_window"].nunique() > 1
    if len(resolved) == 0:
        return ("NOT RESOLVABLE. No arm's window optimum survives its own paired bootstrap against "
                "the runner-up window, so the per-pipeline optima are NOT separable at this sample "
                "size and no per-pipeline optimal window is declared (37.2 rule 3).")
    if not distinct_optima:
        return ("PARTIALLY RESOLVED. {} of {} arms have an optimum that clears its runner-up, but "
                "every arm argmaxes at the SAME window, so there is no per-pipeline difference to "
                "declare.".format(len(resolved), len(arm_table)))
    if len(resolved) == len(arm_table) and arm_vs_arm["excludes_zero"].all():
        return ("RESOLVED. Every arm's optimum clears its runner-up, the argmax windows differ, and "
                "every arm-vs-arm paired delta at the frozen window excludes zero (37.2 rule 2).")
    return ("NOT RESOLVABLE as a per-pipeline effect. {} of {} arms have an optimum clearing its "
            "runner-up and the argmax windows differ, but rule 2 needs BOTH conditions on every arm "
            "involved plus a non-zero arm-vs-arm delta, and that does not hold.".format(
                len(resolved), len(arm_table)))


def main() -> None:
    smoke = "--smoke" in sys.argv
    windows = [16, 64] if smoke else WINDOWS

    assert_geometry_matches_frozen()
    n_selected = frozen_selected_images()
    all_records, calibration_ids, _test_ids, stratum_of = scaffold.build_split(max_images=n_selected)
    if smoke:
        calibration_ids = calibration_ids[:12]
        print("SMOKE RUN: {} windows, {} images. Numbers are NOT reportable.".format(
            len(windows), len(calibration_ids)))
    else:
        assert_calibration_set(calibration_ids)
    arms = scaffold.ARMS
    print("split n={} -> {} calibration images, {} arms, {} windows".format(
        n_selected, len(calibration_ids), len(arms), len(windows)))

    deliver, _delivered = scaffold.make_deliver(all_records)
    scales = scaffold.fit_filter_scales(deliver, calibration_ids, arms, windows)
    if not smoke:
        assert_filter_scales(scales)
        print("filter scales reproduce the frozen run on every window the frozen grid covers.")
    high, low, score_cache = scaffold.make_score_providers(all_records, scales, deliver)

    per_image_frames, summary_rows = [], []
    for window in windows:
        # A smoke run fits its threshold on 12 images, so it cannot reproduce the frozen pair and
        # the guard is skipped -- which is also why its numbers are marked NOT reportable above.
        block = score_window(high, low, calibration_ids, arms, window, stratum_of,
                             check_threshold=not smoke)
        assert_same_grid_and_prevalence(block, window)
        per_image_frames.append(block)
        summary_rows += summarize(block, block.attrs["threshold_low"], block.attrs["threshold_high"])
        # ~475 MB of float32 maps per window; the delivered images stay cached, they are the
        # expensive half and they do not depend on the window.
        score_cache.clear()
        print("  w={:3d} done  threshold ({:.4f}, {:.4f})".format(
            window, block.attrs["threshold_low"], block.attrs["threshold_high"]), flush=True)

    per_image = pd.concat(per_image_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    per_image.to_csv(params.RESULTS_DIR / "window_tradeoff_per_image.csv", index=False)
    summary.to_csv(params.RESULTS_DIR / "window_tradeoff.csv", index=False)

    frozen_window = int(frozen_metadata()["frozen_configuration"]["window_size"])
    arm_table, arm_vs_arm = separability(per_image, summary, calibration_ids, arms,
                                         frozen_window if frozen_window in windows else windows[-1])
    arm_table.to_csv(params.RESULTS_DIR / "window_tradeoff_separability.csv", index=False)
    arm_vs_arm.to_csv(params.RESULTS_DIR / "window_tradeoff_arm_deltas.csv", index=False)
    reading = verdict(arm_table, arm_vs_arm)

    # CLAUDE.md 5.9, and the gate is deliberately NOT this sweep's own numbers.
    #
    # 14 of these 36 CALIBRATION cells clear the all-positive baseline on F1, including the frozen
    # w=64 reference arm (0.2638 against 0.2394). Reading that as "the detector beats the trivial
    # baseline" is precisely the mistake 17.5 made and 18.6 recorded: on the TEST set the same
    # frozen configuration scores F1 0.2000 against a 0.2580 baseline, and the difference between
    # those two readings IS optimism_gap_f1 = +0.1436. So the warning is gated on the frozen run's
    # own test verdict -- a frozen artefact -- and the calibration count is reported next to it as
    # a demonstration of the gap rather than as a result.
    clears = summary[summary["f1"] > summary["all_positive_f1"]]
    frozen_meta = frozen_metadata()
    beats_trivial = bool(frozen_meta["detector_beats_trivial_baseline"])
    frozen_gap = float(frozen_meta["selection_objective_optimism_gap_f1"])
    (params.RESULTS_DIR / "window_tradeoff_metadata.json").write_text(json.dumps({
        "question": ("the approved brief's second contribution: the trade-off between window size "
                     "and detection accuracy, per compression pipeline"),
        "protocol": "EXPERIMENT_LOG.md 37, pre-registered before this ran",
        "split": "CALIBRATION ONLY -- no test image is read",
        "frozen_run_id": params.FROZEN_RUN_ID,
        "n_selected_images": n_selected,
        "n_calibration_images": len(calibration_ids),
        "windows": windows, "arms": arms,
        "direction": FROZEN_DIRECTION, "fusion_variant": FROZEN_VARIANT,
        "local_contrast": FROZEN_CONTRAST,
        "postprocess": [FROZEN_RADIUS, FROZEN_MIN_AREA],
        "target_fpr": TARGET_FPR,
        "threshold_pool": "authentic pixels of the calibration images, all four arms, one per window",
        "declared_objective": ("micro lr_plus (TPR/FPR); micro F1 reported only next to its own "
                              "realised FPR and never used to order"),
        "bootstrap_repetitions": params.BOOTSTRAP_REPETITIONS, "seed": params.RNG_SEED,
        "separability_verdict": reading,
        "frozen_detector_beats_trivial_baseline_on_test": beats_trivial,
        "trivial_baseline_gate": (
            "results/{}/experiment_metadata.json -> detector_beats_trivial_baseline (TEST). NOT "
            "this sweep's calibration cells: 18.6 measured F1 0.2000 vs a 0.2580 baseline on test "
            "for the same frozen configuration that clears it here on calibration."
        ).format(params.FROZEN_RUN_ID),
        "frozen_optimism_gap_f1": frozen_gap,
        "n_calibration_cells_clearing_trivial_baseline": int(len(clears)),
        "n_cells": int(len(summary)),
        "calibration_cells_clearing_trivial_baseline": [
            {"window_size": int(r.window_size), "arm": r.arm, "f1": float(r.f1),
             "all_positive_f1": float(r.all_positive_f1), "fpr": float(r.fpr)}
            for r in clears.itertuples()
        ],
        "smoke": smoke,
    }, indent=2))

    print("\n" + "=" * 78)
    print(summary.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("\nper-arm optimum and its optimism gap (selection on one host-half, read on the other):")
    print(arm_table.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("\npaired arm-vs-{} delta lr_plus at w={}:".format(params.REFERENCE_ARM, frozen_window))
    print(arm_vs_arm.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("\nVERDICT (37.2): " + reading)
    print("\ntrivial all-positive baseline: {}/{} CALIBRATION cells clear it on F1{}".format(
        len(clears), len(summary),
        "" if clears.empty else " -> " + ", ".join(
            "w={} {}".format(int(r.window_size), r.arm) for r in clears.itertuples())))
    if not beats_trivial:
        print("\n" + "=" * 78)
        print("WARNING: the frozen detector does NOT beat the trivial all-positive baseline "
              "ON TEST.")
        print("=" * 78)
        print("Read against the line above: the same frozen configuration clears the baseline on")
        print("CALIBRATION here (w={} {}: F1 {:.4f} vs {:.4f}) and misses it on test (0.2000 vs".format(
            frozen_window, params.REFERENCE_ARM,
            *summary[(summary["window_size"] == frozen_window)
                     & (summary["arm"] == params.REFERENCE_ARM)][["f1", "all_positive_f1"]].iloc[0]))
        print("0.2580, EXPERIMENT_LOG.md 18.6). The difference IS optimism_gap_f1 = "
              "{:+.4f}.".format(frozen_gap))
        print("CLAUDE.md 5.9: while this is on, any ordering between windows or arms is RELATIVE")
        print("-- which configuration degrades a weak signal least -- and NOT a measure of")
        print("intrinsic forensic preservation.")
    print("\nwritten to {}".format(params.RESULTS_DIR))


if __name__ == "__main__":
    main()
