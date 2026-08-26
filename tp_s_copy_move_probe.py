"""Tp_S_ copy-move probe: the FROZEN classical detector, unchanged, on data it has never scored.

GOVERNING RULE (verbatim; binding on every Phase 2 module):

    The detector is selected ONLY on its ability to localize tampering in the uncompressed
    reference arm, BEFORE any codec is involved. It is then frozen completely -- weights,
    preprocessing, thresholds, postprocessing, feature extraction, scale fusion,
    hyperparameters. Only then may we ask which codec best preserves what the frozen detector
    can detect. The detector is never modified, reselected, or retuned to favour any codec.

THIS IS NOT REOPENING EXPERIMENT_LOG.md 22 / dataset_validation.py's Tp_S_ exclusion ("Tp_S_
same-image copy-move: out of scope for the splicing-only protocol"). The splicing-only scope of
the whole pipeline (params.DATASET["splicing_only"], phase2_common.manifest()) is untouched. This
is a standalone, read-only empirical check of a hypothesis about WHY that exclusion should hold,
run OUTSIDE the pipeline, on data disjoint from every calibration/test split ever used -- Tp_S_ has
never been decoded or scored by any code in this repo (enumerated but never pixel-checked,
dataset_validation.py:611-615).

No refit. No new candidate configuration. detector_baseline.frozen_config()/tail_maps()/
score_maps() are imported and called VERBATIM (w=64, two_sided, per_tail, radius=0,
min_area_frac=0.0, scales and thresholds read from results/<FROZEN_RUN_ID>/
experiment_metadata.json).

The Tp_D_ (splicing) comparison numbers are NOT recomputed: they are REREAD from
results/<FROZEN_RUN_ID>/heldout_per_image.csv (threshold_set=="primary", arm=="original"), an
already-published artefact. Precedent for this exact move: EXPERIMENT_LOG.md 38.3 / 19.4 --
"rilettura di un artefatto di test gia' pubblicato, non un passaggio di test" (37.4). Declared here
so the same reasoning is not re-argued.

Two questions, both pre-registered in EXPERIMENT_LOG.md 39 BEFORE this script is run on real data:

  1. abs_separability / lr_plus distribution of a ~150-image Tp_S_ sample, against the published
     Tp_D_ reference-arm distribution. Hypothesis: same-image copy-move gives the detector no
     provenance difference to key on, so abs_separability should cluster near 0. Falsification
     criterion: FALSIFICATION_MEDIAN_ABS_SEPARABILITY (below).
  2. What fraction of the frozen detector's False Positives on Tp_S_ fall inside a visually
     duplicated region elsewhere in the same image -- the copy-move SOURCE, which the CASIA2 mask
     never marks. A bounded template-matching heuristic (cv2.matchTemplate), not a claim of exact
     source segmentation.

Run:  .venv/bin/python tp_s_copy_move_probe.py [--n-target 150] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path[:0] = ["src", "configs"]

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import params
import detector_baseline as baseline
from forensics import dataset, detector, metrics, preservation

# Pre-registered BEFORE looking at any Tp_S_ result (EXPERIMENT_LOG.md 39). Symmetric with
# params.FROZEN_TARGET_FPR=0.10 -- no more principled derivation is available, so it is declared
# as a round, UNKNOWN-flagged default rather than picked after seeing the distribution.
FALSIFICATION_MEDIAN_ABS_SEPARABILITY = 0.10

# --------------------------------------------------------------------------------------------
# Listing, sampling, pairing. Tp_S_ never entered any manifest with a real mask_path (the Phase 1
# manifest lists them but never calls dataset.find_mask -- see EXPERIMENT_LOG.md 39), so this goes
# straight to the CASIA2 directories with the same primitives dataset.py already uses for Tp_D_.


def list_tp_s_paths(tp_dir: Path) -> list[Path]:
    return sorted(
        p for p in tp_dir.iterdir()
        if p.suffix.lower() in dataset.IMAGE_EXTS and p.stem.startswith("Tp_S_")
    )


def sample_candidate_paths(paths: list[Path], n_target: int, buffer_frac: float, seed: int) -> list[Path]:
    """A declared, seeded permutation-then-truncate draw (dataset.split_calibration_test's own
    idiom). The buffer accounts for pairing rejections, whose rate on Tp_S_ has never been
    measured -- this run measures it for the first time."""
    n_draw = math.ceil(n_target * (1 + buffer_frac))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order[:n_draw]]


def pair_one(path: Path, gt_index: dict) -> dict:
    """Same pairing rule as Tp_D_ (dataset.find_mask/load_mask_and_check), minus the Phase-1-only
    host/donor Au geometry check, which degenerates for same-image copy-move (host == donor by
    construction) and is not meaningful here."""
    row = {
        "image_id": path.stem, "image_path": str(path), "mask_path": None,
        "mask_match_rule": None, "image_width": None, "image_height": None,
        "tampered_ratio": None, "status": None, "reject_reason": None,
    }
    mask_path, match_rule = dataset.find_mask(path, gt_index)
    if mask_path is None:
        row["status"], row["reject_reason"] = "no_mask", match_rule
        return row
    row["mask_path"], row["mask_match_rule"] = str(mask_path), match_rule

    with Image.open(path) as handle:
        image_size = handle.size  # (width, height)
    row["image_width"], row["image_height"] = image_size

    mask, reject_reason = dataset.load_mask_and_check(mask_path, image_size)
    if mask is None:
        row["status"], row["reject_reason"] = "shape_mismatch", reject_reason
        return row

    ratio = float(mask.mean())
    row["tampered_ratio"] = ratio
    if ratio <= 0.0:
        row["status"], row["reject_reason"] = "empty_mask", "mask has no positive pixels"
        return row
    if ratio >= 1.0:
        row["status"], row["reject_reason"] = "full_mask", "mask is entirely positive"
        return row

    row["status"] = "paired"
    return row


def pair_candidates(paths: list[Path], gt_index: dict) -> pd.DataFrame:
    return pd.DataFrame([pair_one(p, gt_index) for p in paths])


def load_pair(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert("L")) >= 128
    return rgb, mask


# --------------------------------------------------------------------------------------------
# Step 2: does a False Positive sit inside a visually duplicated (unmasked source) region?
# A bounded heuristic surrogate for full copy-move source segmentation, not a claim of exact
# localization -- CORR_THRESHOLD / MIN_AREA_PX are declared defaults, see EXPERIMENT_LOG.md 39.


def fp_components(binary_pred: np.ndarray, mask: np.ndarray, min_area_px: int) -> list[dict]:
    """Connected components of predicted-positive-but-unmasked pixels. Components under
    min_area_px are kept in the output (flagged, not dropped) so their size distribution stays
    visible rather than silently vanishing."""
    fp_mask = binary_pred & ~mask
    if not fp_mask.any():
        return []
    labelled, _n = ndimage.label(fp_mask)
    slices = ndimage.find_objects(labelled)
    out = []
    for component_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        area_px = int((labelled[sl] == component_id).sum())
        out.append({
            "component_id": component_id, "area_px": area_px,
            "row_slice": sl[0], "col_slice": sl[1],
            "below_min_area": area_px < min_area_px,
        })
    return out


def best_offsite_match(gray: np.ndarray, row_slice: slice, col_slice: slice) -> tuple[float, tuple[int, int] | None]:
    """Best normalized-cross-correlation match of the FP patch elsewhere in the same image.

    cv2.matchTemplate requires the template strictly smaller than the search image; a patch over
    half the image in either dimension is degenerate for this search and is reported unchecked
    (nan), never forced through. The trivial match at the patch's own location is suppressed by
    zeroing a window of radius max(patch.shape) around it, so "elsewhere" is enforced, not just
    "not bit-identical".
    """
    patch = gray[row_slice, col_slice]
    patch_h, patch_w = patch.shape
    image_h, image_w = gray.shape
    if patch_h < 8 or patch_w < 8 or patch_h >= image_h // 2 or patch_w >= image_w // 2:
        return float("nan"), None

    response = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
    row0, col0 = row_slice.start, col_slice.start
    radius = max(patch_h, patch_w)
    r0, r1 = max(0, row0 - radius), min(response.shape[0], row0 + radius + 1)
    c0, c1 = max(0, col0 - radius), min(response.shape[1], col0 + radius + 1)
    response[r0:r1, c0:c1] = -1.0

    loc = np.unravel_index(int(np.argmax(response)), response.shape)
    return float(response[loc]), (int(loc[0]), int(loc[1]))


def fp_duplication_components(
    gray: np.ndarray, mask: np.ndarray, binary_pred: np.ndarray, corr_threshold: float, min_area_px: int
) -> list[dict]:
    rows = []
    for comp in fp_components(binary_pred, mask, min_area_px):
        if comp["below_min_area"]:
            rows.append({
                "component_id": comp["component_id"], "area_px": comp["area_px"],
                "best_correlation": float("nan"), "match_row": None, "match_col": None,
                "checked": False, "duplicated": False, "skip_reason": "below_min_area",
            })
            continue
        best_corr, best_loc = best_offsite_match(gray, comp["row_slice"], comp["col_slice"])
        checked = not math.isnan(best_corr)
        rows.append({
            "component_id": comp["component_id"], "area_px": comp["area_px"],
            "best_correlation": best_corr,
            "match_row": best_loc[0] if best_loc else None,
            "match_col": best_loc[1] if best_loc else None,
            "checked": checked,
            "duplicated": bool(checked and best_corr >= corr_threshold),
            "skip_reason": None if checked else "degenerate_patch_size",
        })
    return rows


def summarize_step2(fp_components_df: pd.DataFrame, n_images_with_fp: int, n_images_total: int) -> dict:
    if fp_components_df.empty:
        total_area = dup_area = 0
        n_checked = n_below = n_degenerate = 0
    else:
        total_area = int(fp_components_df["area_px"].sum())
        dup_area = int(fp_components_df.loc[fp_components_df["duplicated"], "area_px"].sum())
        n_checked = int(fp_components_df["checked"].sum())
        n_below = int((fp_components_df["skip_reason"] == "below_min_area").sum())
        n_degenerate = int((fp_components_df["skip_reason"] == "degenerate_patch_size").sum())
    return {
        "n_components_total": int(len(fp_components_df)),
        "n_components_checked": n_checked,
        "n_components_skipped_below_min_area": n_below,
        "n_components_skipped_degenerate_patch_size": n_degenerate,
        "total_fp_area_px": total_area,
        "duplicated_fp_area_px": dup_area,
        "pooled_fp_duplicated_fraction": (dup_area / total_area) if total_area else float("nan"),
        "n_images_zero_fp": int(n_images_total - n_images_with_fp),
    }


# --------------------------------------------------------------------------------------------
# Per-image scoring: one decode, one score-map pass, feeds both questions.


def process_image(
    image_id: str, image_path: str, mask_path: str, scales, threshold_low: float, threshold_high: float,
    corr_threshold: float, min_area_px: int,
) -> tuple[dict, list[dict]]:
    image_rgb, mask = load_pair(image_path, mask_path)
    maps = baseline.score_maps(image_rgb, scales, threshold_low, threshold_high)
    binary = maps["binary"]

    sep = preservation.separability(maps["raw"], mask)
    tp = int((binary & mask).sum())
    fp = int((binary & ~mask).sum())
    fn = int((~binary & mask).sum())
    tn = int((~binary & ~mask).sum())
    lr_plus = metrics.metrics_from_counts(tp, fp, fn, tn)["lr_plus"]

    per_image_row = {
        "image_id": image_id,
        "tampered_ratio": float(mask.mean()),
        "n_tampered": int(mask.sum()),
        "n_authentic": int((~mask).sum()),
        "raw_abs_separability": sep["abs_separability"],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "lr_plus": lr_plus,
        "silent": bool(binary.sum() == 0),
    }

    gray = detector.to_gray(image_rgb).astype(np.float32)
    components = fp_duplication_components(gray, mask, binary, corr_threshold, min_area_px)
    for component in components:
        component["image_id"] = image_id
    return per_image_row, components


# --------------------------------------------------------------------------------------------
# Reread (not recompute) the published Tp_D_ reference-arm numbers.


def load_published_tp_d(reference_run_id: str = params.FROZEN_RUN_ID) -> pd.DataFrame:
    path = params.RESULTS_ROOT / reference_run_id / "heldout_per_image.csv"
    if not path.exists():
        raise FileNotFoundError(
            "{} missing: the published Tp_D_ reference-arm numbers are not recoverable and no "
            "default is substituted (CLAUDE.md 5.11).".format(path)
        )
    frame = pd.read_csv(path)
    frame = frame[(frame["threshold_set"] == "primary") & (frame["arm"] == "original")]
    if len(frame) != 349:
        raise ValueError(
            "expected 349 frozen test-split Tp_D_ rows in {}, got {}: the frozen test set moved "
            "under this reader.".format(path, len(frame))
        )
    return frame[["image_id", "raw_abs_separability", "lr_plus"]].reset_index(drop=True)


def _describe(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p05": float("nan"), "p25": float("nan"), "p75": float("nan"), "p95": float("nan")}
    p05, p25, p75, p95 = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
    return {"n": int(len(values)), "median": float(np.median(values)), "mean": float(values.mean()),
            "p05": float(p05), "p25": float(p25), "p75": float(p75), "p95": float(p95)}


def compare_distributions(tp_s_df: pd.DataFrame, tp_d_df: pd.DataFrame) -> dict:
    """Descriptive only -- no selection decision is taken from this, so no formal test is
    computed. See EXPERIMENT_LOG.md 39 for the pre-registered falsification criterion."""
    return {
        "tp_s_abs_separability": _describe(tp_s_df["raw_abs_separability"].to_numpy(dtype=float)),
        "tp_d_reference_abs_separability": _describe(tp_d_df["raw_abs_separability"].to_numpy(dtype=float)),
        "tp_s_lr_plus": _describe(tp_s_df["lr_plus"].to_numpy(dtype=float)),
        "tp_d_reference_lr_plus": _describe(tp_d_df["lr_plus"].to_numpy(dtype=float)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-target", type=int, default=150)
    parser.add_argument("--buffer-frac", type=float, default=0.2)
    parser.add_argument("--corr-threshold", type=float, default=0.93)
    parser.add_argument("--min-area-px", type=int, default=64)
    parser.add_argument("--seed", type=int, default=params.RNG_SEED)
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N candidates only")
    args = parser.parse_args()

    print(__doc__.split("Run:")[0].strip())
    print("=" * 94)

    all_paths = list_tp_s_paths(params.CASIA_TP_DIR)
    print("Tp_S_ files found: {}".format(len(all_paths)))

    candidates = sample_candidate_paths(all_paths, args.n_target, args.buffer_frac, args.seed)
    if args.limit:
        candidates = candidates[: args.limit]
        print("SMOKE MODE: {} candidates, nothing will be written.".format(len(candidates)))

    gt_index = dataset.build_gt_index(params.CASIA_MASK_DIR)
    pairing_df = pair_candidates(candidates, gt_index)
    paired = pairing_df[pairing_df["status"] == "paired"].reset_index(drop=True)
    print("paired: {}/{} candidates (target {})".format(len(paired), len(candidates), args.n_target))
    print(pairing_df["status"].value_counts().to_string())

    sample_df = paired if args.limit else paired.iloc[: args.n_target]
    if not args.limit and len(sample_df) < args.n_target:
        print("WARNING: only {} of {} target Tp_S_ images paired successfully -- reporting the "
              "actual n, never padding it.".format(len(sample_df), args.n_target))

    frozen = baseline.frozen_config()
    print("\nfrozen configuration read from results/{}: w=64 two_sided per_tail".format(params.FROZEN_RUN_ID))

    per_image_rows, all_components = [], []
    for row in sample_df.itertuples():
        per_image_row, components = process_image(
            row.image_id, row.image_path, row.mask_path,
            frozen["scales"], frozen["threshold_low"], frozen["threshold_high"],
            args.corr_threshold, args.min_area_px,
        )
        per_image_rows.append(per_image_row)
        all_components.extend(components)

    per_image_df = pd.DataFrame(per_image_rows)
    components_df = pd.DataFrame(all_components)

    n_images_with_fp = int((per_image_df["fp"] > 0).sum()) if len(per_image_df) else 0
    step2_summary = summarize_step2(components_df, n_images_with_fp, len(per_image_df))

    tp_d_df = load_published_tp_d()
    distribution_summary = compare_distributions(per_image_df, tp_d_df)

    median_abs_sep = distribution_summary["tp_s_abs_separability"]["median"]
    hypothesis_holds = bool(
        not math.isnan(median_abs_sep) and median_abs_sep < FALSIFICATION_MEDIAN_ABS_SEPARABILITY
    )

    print("\n" + "=" * 94)
    print("STEP 1 -- abs_separability / lr_plus: Tp_S_ (n={}) vs published Tp_D_ reference (n={})".format(
        len(per_image_df), len(tp_d_df)))
    print("=" * 94)
    for key, d in distribution_summary.items():
        print("{:32s} n={:>4d}  median={:+.4f}  mean={:+.4f}  [{:+.4f}, {:+.4f}]".format(
            key, d["n"], d["median"], d["mean"], d["p05"], d["p95"]))
    print("\npre-registered criterion: median raw_abs_separability < {:.2f}".format(
        FALSIFICATION_MEDIAN_ABS_SEPARABILITY))
    print("hypothesis {}: Tp_S_ median = {:.4f}".format(
        "HOLDS" if hypothesis_holds else "FALSIFIED", median_abs_sep))

    print("\n" + "=" * 94)
    print("STEP 2 -- False Positives inside a visually duplicated (unmasked source) region")
    print("=" * 94)
    for key, value in step2_summary.items():
        print("{:44s} {}".format(key, value))

    if args.limit:
        print("\nSMOKE MODE: nothing written.")
        return

    pairing_df.to_csv(params.RESULTS_DIR / "tp_s_pairing.csv", index=False)
    per_image_df.to_csv(params.RESULTS_DIR / "tp_s_per_image.csv", index=False)
    components_df.to_csv(params.RESULTS_DIR / "tp_s_fp_components.csv", index=False)
    (params.RESULTS_DIR / "tp_s_probe_summary.json").write_text(json.dumps({
        "n_target": args.n_target, "n_candidates_drawn": len(candidates),
        "n_paired_success": int(len(sample_df)), "buffer_frac": args.buffer_frac, "seed": args.seed,
        "frozen_run_id": params.FROZEN_RUN_ID, "window_size": 64, "direction": "two_sided",
        "corr_threshold": args.corr_threshold, "min_area_px": args.min_area_px,
        **distribution_summary,
        "falsification_criterion": "median raw_abs_separability < {}".format(
            FALSIFICATION_MEDIAN_ABS_SEPARABILITY),
        "hypothesis_holds": hypothesis_holds,
        **step2_summary,
        "test_images_touched": 0,
    }, indent=2))
    print("\nwritten to {}".format(params.RESULTS_DIR))
    print("test images touched: 0")


if __name__ == "__main__":
    main()
