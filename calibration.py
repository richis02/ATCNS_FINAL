"""Calibration-grid search, MCC-primary configuration selection, and
held-out evaluation.

Decoupled from image I/O: callers supply a ``score_provider(image_id,
pipeline, window_size) -> (score_map, mask_bool)`` callable (the
orchestration notebook wires this to dataset.load_mask_and_check +
compression.apply_compression + detector.detector_score_map, memoized so
each (image, pipeline, window) triple is computed once).

Threshold construction matches the confirmed reference run: an empirical
quantile of pooled AUTHENTIC-pixel scores from calibration images at the
target FPR, not a fixed mean+k*std rule (see detector.apply_threshold).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import metrics as metrics_mod
from .detector import predict_mask

ScoreProvider = Callable[[str, str, int], tuple[np.ndarray, np.ndarray]]

# Candidate identity. radius/min_area_frac joined the key in iteration 15: two candidates sharing
# a (window, direction, threshold) but differing in postprocess are different operating points.
# local_contrast joined in iteration 20. It is a genuine grid axis (unlike the fusion variant):
# the caller runs the whole grid once per setting with the matching provider -- so each setting
# gets its OWN threshold fitted on its OWN authentic-pixel distribution -- and tags the frame with
# this column before best_configuration pools them. _candidate_keys picks it up automatically.
GROUP_KEYS = [
    "scope", "window_size", "target_fpr", "direction", "threshold_low", "threshold_high",
    "radius", "min_area_frac", "local_contrast",
]


def _postprocess_grid(params: dict) -> list[tuple[int, float]]:
    """(0, 0.0) -- no regularization -- is the default, so callers that predate iteration 15 and
    grids built without a 'postprocess' key keep their exact previous behaviour."""
    return [tuple(p) for p in params.get("postprocess", [(0, 0.0)])]


def pool_authentic_scores(
    score_provider: ScoreProvider,
    image_ids: list[str],
    pipelines: list[str],
    window_size: int,
    pixels_cap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pooled = []
    for image_id in image_ids:
        for pipeline in pipelines:
            score_map, mask = score_provider(image_id, pipeline, window_size)
            authentic = score_map[~mask]
            if len(authentic) > pixels_cap:
                idx = rng.choice(len(authentic), size=pixels_cap, replace=False)
                authentic = authentic[idx]
            pooled.append(authentic)
    return np.concatenate(pooled)


def threshold_for_target_fpr(
    pool: np.ndarray, target_fpr: float, direction: str, pool_low: np.ndarray | None = None
) -> tuple[float | None, float | None]:
    """pool_low: the authentic-pixel pool the LOW threshold is a quantile of, when the two tails
    are read from two differently-fused maps (iteration 18's per-tail variant). It has to be the
    pool of the same map the low tail will be thresholded on, or the quantile does not deliver the
    target FPR on it. None -- the default -- takes both quantiles from `pool`, unchanged."""
    low_pool = pool if pool_low is None else pool_low
    if direction == "low":
        return float(np.quantile(low_pool, target_fpr)), None
    if direction == "high":
        return None, float(np.quantile(pool, 1 - target_fpr))
    if direction == "two_sided":
        half = target_fpr / 2
        return float(np.quantile(low_pool, half)), float(np.quantile(pool, 1 - half))
    if direction == "coherence":
        # Both tails at the FULL target FPR, not half each: detector.predict_mask keeps only one of
        # the two masks, so the budget actually spent is target_fpr -- matched to 'low'/'high'.
        # Splitting it here (as 'two_sided' must) would handicap the rule for no reason.
        return float(np.quantile(low_pool, target_fpr)), float(np.quantile(pool, 1 - target_fpr))
    raise ValueError(f"unknown direction: {direction}")


def evaluate_calibration_grid(
    score_provider: ScoreProvider,
    calibration_ids: list[str],
    pipelines: list[str],
    params: dict,
    seed: int,
    pool_ids: list[str] | None = None,
    scopes: list[str] | None = None,
    stratum_of: dict[str, str] | None = None,
    score_provider_low: ScoreProvider | None = None,
) -> pd.DataFrame:
    """calibration_ids are the images each candidate is *scored* on; pool_ids are the images
    each candidate's threshold is *fit* from (defaults to calibration_ids, the original
    in-sample behavior). Keeping these separate is what lets evaluate_calibration_grid_cv
    reuse this function unchanged for out-of-fold scoring.

    scopes defaults to ["shared"] + pipelines (one shared candidate set plus a per-pipeline
    one). Pass scopes=["shared"] when a per-pipeline threshold is not wanted: it is the whole
    point of a frozen-detector comparison that every arm gets the same number, and it also
    cuts the grid by a factor of len(pipelines)+1.

    stratum_of maps image_id -> region-size stratum label; when supplied it is carried onto every
    row so best_configuration can apply the per-stratum lr_plus gate. Omit it and the gate is
    skipped (no strata, nothing to gate on).

    score_provider_low supplies the map (and the threshold pool) for the LOW tail, for iteration
    18's per-tail fusion variant: `max` across filters resolves the high tail, `min` the low one.
    None -- the default -- reads both tails off score_provider and is bit-identical to before.
    The variant is global, not a grid axis: the fusion is determined by the tail being read, not
    fitted, so the candidate count is unchanged and so is the winner's-curse exposure."""
    windows = params["windows"]
    fpr_grid = params["target_fpr_grid"]
    directions = params["directions"]
    postprocess = _postprocess_grid(params)
    pixels_cap = params["calibration_pixels_per_class_per_map"]
    pool_ids = calibration_ids if pool_ids is None else pool_ids

    scopes = (["shared"] + list(pipelines)) if scopes is None else list(scopes)
    rows = []
    for scope in scopes:
        scope_pipelines = pipelines if scope == "shared" else [scope]
        for window in windows:
            pool = pool_authentic_scores(score_provider, pool_ids, scope_pipelines, window, pixels_cap, seed)
            pool_low = None if score_provider_low is None else pool_authentic_scores(
                score_provider_low, pool_ids, scope_pipelines, window, pixels_cap, seed
            )
            for target_fpr in fpr_grid:
                for direction in directions:
                    threshold_low, threshold_high = threshold_for_target_fpr(
                        pool, target_fpr, direction, pool_low
                    )
                    for target_pipeline in scope_pipelines:
                        for image_id in calibration_ids:
                            score_map, mask = score_provider(image_id, target_pipeline, window)
                            score_map_low = None if score_provider_low is None else \
                                score_provider_low(image_id, target_pipeline, window)[0]
                            # The score map is fetched once and reused across the postprocess
                            # variants: the added cost of this axis is morphology, not detection.
                            for radius, min_area_frac in postprocess:
                                pred = predict_mask(
                                    score_map, direction, threshold_low, threshold_high,
                                    radius=radius, min_area_frac=min_area_frac,
                                    score_map_low=score_map_low,
                                )
                                row = metrics_mod.compute_metrics(pred, mask)
                                row.update(
                                    {
                                        "scope": scope,
                                        "window_size": window,
                                        "target_fpr": target_fpr,
                                        "direction": direction,
                                        "threshold_low": threshold_low,
                                        "threshold_high": threshold_high,
                                        "radius": radius,
                                        "min_area_frac": min_area_frac,
                                        "target_pipeline": target_pipeline,
                                        "image_id": image_id,
                                        "stratum": (stratum_of or {}).get(image_id, "ALL"),
                                    }
                                )
                                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_calibration_grid_cv(
    score_provider: ScoreProvider,
    calibration_ids: list[str],
    pipelines: list[str],
    params: dict,
    seed: int,
    n_folds: int = 3,
    scopes: list[str] | None = None,
    stratum_of: dict[str, str] | None = None,
    score_provider_low: ScoreProvider | None = None,
) -> pd.DataFrame:
    """K-fold cross-validated version of evaluate_calibration_grid: each candidate's threshold
    is fit on the K-1 other folds and scored only on its held-out fold, so every calibration
    image contributes exactly one out-of-fold prediction per candidate. best_configuration()
    ranking on this instead of the plain in-sample grid is what penalizes candidates whose
    in-sample calibration_mcc is inflated by overfitting the whole calibration set at once --
    calibration_selection_diagnostic() measures the size of that inflation directly and showed
    it is large (optimism_gap 0.06-0.17 on the reference run).
    ponytail: fixed K rather than leave-one-out -- with ~30 calibration images K=3 keeps each
    fit fold reasonably sized; raise n_folds if the calibration set grows a lot.
    """
    rng = np.random.default_rng(seed)
    shuffled_ids = list(calibration_ids)
    rng.shuffle(shuffled_ids)
    folds = [list(f) for f in np.array_split(shuffled_ids, min(n_folds, len(shuffled_ids))) if len(f) > 0]

    fold_grids = []
    for fold_ids in folds:
        fold_set = set(fold_ids)
        fit_ids = [image_id for image_id in shuffled_ids if image_id not in fold_set]
        fold_grids.append(
            evaluate_calibration_grid(
                score_provider, fold_ids, pipelines, params, seed, pool_ids=fit_ids, scopes=scopes,
                stratum_of=stratum_of, score_provider_low=score_provider_low,
            )
        )
    return pd.concat(fold_grids, ignore_index=True)


def _candidate_keys(grid_df: pd.DataFrame) -> list[str]:
    """Candidate identity, EXCLUDING the fitted threshold.

    This is a fix, not a refactor. The old key included threshold_low/threshold_high, but
    evaluate_calibration_grid_cv fits a different threshold per fold by construction -- so on a
    3-fold CV grid every (window, direction) split into 3 separate groups of ~50 images, and the
    argmax ran over 27 fold-specific candidates instead of 9 pooled ones. The reported
    calibration_mcc came from a single fold. Verified on the reference run's own
    calibration_grid_cv.csv: 3 distinct threshold_low values per (window_size, direction).

    That is the mechanism behind the selection failure this iteration exists to fix: a candidate
    was picked on ~50 images by a 0.009 MCC margin, and the runner-up (w=64, low) turned out ~7x
    better on the test set. The threshold is a nuisance parameter refitted on the full calibration
    set after selection (see the notebook's 'the CV above only chose the geometry'), so pooling
    across folds is both correct and what the CV docstring already claimed to do.
    """
    return [key for key in GROUP_KEYS if key in grid_df.columns and key not in ("threshold_low", "threshold_high")]


def _per_stratum_lr_plus(grid_df: pd.DataFrame, candidate_keys: list[str]) -> pd.Series:
    """Worst (minimum) lr_plus across region-size strata, per candidate.

    lr_plus (TPR/FPR) is the statistic compared across strata because prevalence varies ~32x
    between them (1.1% vs 35.7%), which makes MCC and precision_lift incomparable there.
    Counts are pooled within (candidate, stratum) before the ratio -- a per-image ratio on a
    stratum with 1.1% prevalence is mostly noise.
    """
    per_stratum = grid_df.groupby(candidate_keys + ["stratum"], dropna=False)[["tp", "fp", "fn", "tn"]].sum()
    recall = per_stratum["tp"] / (per_stratum["tp"] + per_stratum["fn"]).replace(0, np.nan)
    fpr = per_stratum["fp"] / (per_stratum["fp"] + per_stratum["tn"]).replace(0, np.nan)
    lr_plus = recall / fpr.replace(0, np.nan)
    # A stratum where lr_plus is undefined (no tampered pixels, or no authentic pixels) carries no
    # evidence either way and must not veto a candidate; only a measured value below 1 does.
    return lr_plus.groupby(level=list(range(len(candidate_keys)))).min()


def best_configuration(grid_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    max_macro_fpr = params["selection_max_macro_fpr"]
    ratio_low, ratio_high = params["selection_positive_ratio_bounds"]
    objective = params.get("selection_objective", "micro_mcc")
    gate_on_strata = params.get("require_positive_lr_plus_per_stratum", False) and "stratum" in grid_df.columns

    group_keys = _candidate_keys(grid_df)
    # evaluate_calibration_grid always emits per-image mcc/f1, but callers may hand over a
    # counts-only frame; derive what the macro aggregation needs rather than requiring it.
    missing = [c for c in ("mcc", "f1") if c not in grid_df.columns]
    if missing:
        derived = grid_df.apply(
            lambda r: pd.Series(metrics_mod.metrics_from_counts(int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"]))),
            axis=1,
        )
        grid_df = grid_df.join(derived[missing])
    # Pool tp/fp/fn/tn across every image in the group into one confusion matrix, then derive
    # MCC/F1/etc. from those pooled counts (micro-average) instead of averaging each image's own
    # MCC (macro-average). A handful of images with few predicted positives makes per-image MCC
    # wildly noisy; pooling first means one lucky/unlucky sparse image can no longer dominate.
    grouped = grid_df.groupby(group_keys, dropna=False)
    pooled_counts = grouped.agg(
        tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), tn=("tn", "sum"), n_images=("tp", "size"),
        # Representative threshold, for reporting only: on the CV grid it is the median of the
        # per-fold fits, on an in-sample grid there is exactly one value so the median IS it.
        threshold_low=("threshold_low", "median"), threshold_high=("threshold_high", "median"),
    )
    pooled_metrics = pooled_counts.apply(
        lambda row: pd.Series(metrics_mod.metrics_from_counts(int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"]))),
        axis=1,
    )
    # MACRO: mean of PER-IMAGE MCC, with undefined counted as 0 rather than dropped. This is
    # load-bearing. pandas' mean drops NaN silently, and per-image MCC is NaN exactly when the
    # candidate predicted no positives on that image -- so a candidate firing on 20 of 151 images
    # would otherwise be scored on its 20 luckiest ones and win. A silent prediction carries zero
    # information about that image and must score zero. (Same trap summarize_heldout documents.)
    macro = pd.DataFrame({
        "calibration_macro_mcc": grouped["mcc"].apply(lambda s: s.fillna(0.0).mean()),
        "calibration_macro_f1": grouped["f1"].apply(lambda s: s.fillna(0.0).mean()),
        "n_mcc_defined": grouped["mcc"].apply(lambda s: int(s.notna().sum())),
    })
    agg = pooled_counts[["n_images", "threshold_low", "threshold_high"]].join(pooled_metrics).join(macro)
    if gate_on_strata:
        agg = agg.join(_per_stratum_lr_plus(grid_df, group_keys).rename("worst_stratum_lr_plus"))
    agg = agg.reset_index()
    agg = agg.rename(
        columns={
            "mcc": "calibration_mcc",
            "balanced_accuracy": "calibration_balanced_accuracy",
            "f1": "calibration_f1",
            "iou": "calibration_iou",
            "fpr": "calibration_fpr",
            "positive_ratio": "calibration_positive_ratio",
        }
    )
    admissible = (
        (agg["calibration_fpr"] <= max_macro_fpr)
        & (agg["calibration_positive_ratio"] >= ratio_low)
        & (agg["calibration_positive_ratio"] <= ratio_high)
    )
    if gate_on_strata:
        # Reject anti-correlated candidates: lr_plus <= 1 in any stratum is worse than chance
        # there. An undefined worst value (NaN) is not evidence of failure, so it does not veto.
        admissible &= ~(agg["worst_stratum_lr_plus"] <= 1.0)
    agg["selection_status"] = np.where(admissible, "fpr_constrained", "rejected_degenerate_or_fpr")

    # Objective first, then the previous keys as tiebreaks so the ranking stays total.
    # "micro_f1" is not a cosmetic variant of the micro default: on the iteration-16 grid, micro
    # MCC and micro F1 pick DIFFERENT windows (128 vs 96) on a 0.0003 margin, so which micro
    # statistic leads the sort is itself a protocol decision (EXPERIMENT_LOG §17.1).
    lead = {"macro_mcc": ["calibration_macro_mcc"], "micro_f1": ["calibration_f1"]}
    rank_by = list(lead.get(objective, []))
    rank_by += [k for k in ("calibration_mcc", "calibration_balanced_accuracy", "calibration_f1",
                            "calibration_iou") if k not in rank_by]

    rows = []
    for scope, group in agg.groupby("scope"):
        eligible = group[group["selection_status"] == "fpr_constrained"]
        candidates = eligible if len(eligible) else group
        ranked = candidates.sort_values(
            by=rank_by + ["window_size", "target_fpr"],
            ascending=[False] * len(rank_by) + [True, True],
            na_position="last",
        )
        best = ranked.iloc[0]
        configuration = "shared_frozen" if scope == "shared" else "pipeline_conditioned"
        target_pipeline = "all" if scope == "shared" else scope
        row = {
            "configuration": configuration,
            "target_pipeline": target_pipeline,
            "window_size": int(best["window_size"]),
            "direction": best["direction"],
            "target_fpr": best["target_fpr"],
            "radius": int(best.get("radius", 0)),
            "min_area_frac": float(best.get("min_area_frac", 0.0)),
            # Defaults to False so a caller that never supplies the axis is unaffected.
            "local_contrast": bool(best.get("local_contrast", False)),
            "threshold_low": best["threshold_low"],
            "threshold_high": best["threshold_high"],
            "selection_status": best["selection_status"],
            "selection_objective": objective,
            "n_candidates_ranked": len(candidates),
            "n_images": int(best["n_images"]),
            "n_mcc_defined": int(best["n_mcc_defined"]),
            "calibration_macro_mcc": best["calibration_macro_mcc"],
            "calibration_macro_f1": best["calibration_macro_f1"],
            "calibration_mcc": best["calibration_mcc"],
            "calibration_balanced_accuracy": best["calibration_balanced_accuracy"],
            "calibration_f1": best["calibration_f1"],
            "calibration_iou": best["calibration_iou"],
            "calibration_fpr": best["calibration_fpr"],
            "calibration_positive_ratio": best["calibration_positive_ratio"],
        }
        if gate_on_strata:
            row["worst_stratum_lr_plus"] = best["worst_stratum_lr_plus"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_test(
    score_provider: ScoreProvider,
    test_ids: list[str],
    selected_configs: pd.DataFrame,
    pipelines: list[str],
    stratum_of: dict[str, str] | None = None,
    score_provider_low: ScoreProvider | None = None,
) -> pd.DataFrame:
    rows = []
    for _, config in selected_configs.iterrows():
        for target_pipeline in pipelines:
            for image_id in test_ids:
                window = int(config["window_size"])
                score_map, mask = score_provider(image_id, target_pipeline, window)
                score_map_low = None if score_provider_low is None else \
                    score_provider_low(image_id, target_pipeline, window)[0]
                pred = predict_mask(
                    score_map, config["direction"], config["threshold_low"], config["threshold_high"],
                    radius=int(config.get("radius", 0)), min_area_frac=float(config.get("min_area_frac", 0.0)),
                    score_map_low=score_map_low,
                )
                row = metrics_mod.compute_metrics(pred, mask)
                row.update(
                    {
                        "configuration": config["configuration"],
                        "config_target_pipeline": config["target_pipeline"],
                        "target_pipeline": target_pipeline,
                        "image_id": image_id,
                        "stratum": (stratum_of or {}).get(image_id, "ALL"),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_selection_diagnostic(
    score_provider: ScoreProvider,
    calibration_ids: list[str],
    pipelines: list[str],
    params: dict,
    seed: int,
    scopes: list[str] | None = None,
    stratum_of: dict[str, str] | None = None,
    score_provider_low: ScoreProvider | None = None,
    providers_by_contrast: dict | None = None,
) -> pd.DataFrame:
    """Estimate how optimistic the argmax-over-grid threshold selection is.

    best_configuration() picks the best of ~60 grid candidates on the calibration set, then
    reports that candidate's own score -- but an argmax over many candidates on a small sample is
    expected to look better on the data it was chosen from than on fresh data (winner's-curse /
    selection bias), independent of whether the detector itself is any good. To measure the size of
    that effect: split calibration_ids in half, repeat the selection on one half (inner_fit), then
    score the winning threshold on the other half (inner_check), which the selection never saw.
    optimism_gap = calibration_mcc_inner_fit - inner_check_mcc.

    The inner selection uses evaluate_calibration_grid_CV, matching what the notebook actually
    does. This is a fix, not a refactor. It used to call the plain in-sample
    evaluate_calibration_grid while the real selection (notebooks/forensics_final.py) has used the
    CV grid since iteration 15, so the reported gap summed two different things:

      (a) winner's curse -- the only quantity CLAUDE.md 5.7 asks for, and
      (b) in-sample vs out-of-sample fit of the threshold -- which the real procedure already
          removes with the CV, so it must not be charged to the selection a second time.

    Measured on run 20260817-204628: calibration_f1_inner_fit was 0.3355 in-sample against the
    real procedure's 0.2676 out-of-fold, i.e. ~0.068 of the reported +0.141..+0.172 was (b). That
    inflated gap was then used in EXPERIMENT_LOG 17.5 to argue the iteration-16 margin was "inside
    selection noise by a factor of 5" -- a claim the corrected number does not support (the
    conclusion survives on the test result and on the structural argument in 16.2, not on this).
    Same family as the 15.1 defect: a number whose label promises a different object than the one
    it computes.

    providers_by_contrast (iteration 20) is {local_contrast: (high_provider, low_provider)}. Each
    setting of that axis is a DIFFERENT MAP with its own thresholds, so the real selection runs the
    grid once per setting and argmaxes over the concatenation. Handing the diagnostic a single
    provider would repeat the 18.1 defect in a weaker form: it would measure the curse of a
    20-candidate argmax and label it with the gap of the 40-candidate one actually performed. When
    None the behaviour is exactly as before, single provider, unchanged for every other caller.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(calibration_ids)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    inner_fit_ids, inner_check_ids = shuffled[:half], shuffled[half:]

    provider_map = providers_by_contrast or {None: (score_provider, score_provider_low)}
    inner_grid_df = pd.concat(
        [
            evaluate_calibration_grid_cv(
                high, inner_fit_ids, pipelines, params, seed, scopes=scopes,
                stratum_of=stratum_of, score_provider_low=low,
            ).assign(**({} if contrast is None else {"local_contrast": contrast}))
            for contrast, (high, low) in provider_map.items()
        ],
        ignore_index=True,
    )
    inner_selected_df = best_configuration(inner_grid_df, params)

    # The check half is scored with the provider matching each winner's OWN axis setting: scoring
    # a top-hat candidate on the plain map would measure a detector nobody selected.
    check_frames = []
    for i in range(len(inner_selected_df)):
        selected_row = inner_selected_df.iloc[[i]]
        key = None if providers_by_contrast is None else bool(selected_row["local_contrast"].iloc[0])
        high, low = provider_map[key]
        check_frames.append(evaluate_test(
            high, inner_check_ids, selected_row, pipelines, stratum_of=stratum_of,
            score_provider_low=low,
        ))
    inner_check_df = pd.concat(check_frames, ignore_index=True)
    # All three aggregations, because the objective has changed across iterations and the gap that
    # matters is the gap of the number actually being argmaxed. Since iteration 17 that is micro
    # F1 -- which is also the statistic the trivial-baseline WARNING watches, so its optimism is
    # the one to read next to that warning (EXPERIMENT_LOG §17.2).
    inner_check_summary = summarize_heldout(inner_check_df).rename(
        columns={"calibration": "configuration", "mcc": "inner_check_mcc",
                 "macro_mcc": "inner_check_macro_mcc", "f1": "inner_check_f1"}
    )[["target_pipeline", "configuration", "inner_check_mcc", "inner_check_macro_mcc", "inner_check_f1"]]

    rows = []
    for _, sel in inner_selected_df.iterrows():
        eval_pipelines = pipelines if sel["configuration"] == "shared_frozen" else [sel["target_pipeline"]]
        for pipeline in eval_pipelines:
            match = inner_check_summary[
                (inner_check_summary["configuration"] == sel["configuration"])
                & (inner_check_summary["target_pipeline"] == pipeline)
            ]
            inner_check_mcc = float(match["inner_check_mcc"].iloc[0]) if len(match) else np.nan
            inner_check_macro = float(match["inner_check_macro_mcc"].iloc[0]) if len(match) else np.nan
            inner_check_f1 = float(match["inner_check_f1"].iloc[0]) if len(match) else np.nan
            rows.append(
                {
                    "configuration": sel["configuration"],
                    "target_pipeline": pipeline,
                    # Which side of the local_contrast axis the INNER argmax landed on. Without it
                    # there is no way to tell a diagnostic that searched the whole axis from one
                    # that only ever saw half of it.
                    "inner_selected_local_contrast": bool(sel.get("local_contrast", False)),
                    "n_inner_fit_images": len(inner_fit_ids),
                    "n_inner_check_images": len(inner_check_ids),
                    "calibration_mcc_inner_fit": sel["calibration_mcc"],
                    "inner_check_mcc": inner_check_mcc,
                    "optimism_gap": sel["calibration_mcc"] - inner_check_mcc,
                    # The macro pair is the one to read: it is the objective actually being
                    # argmaxed, so it is the gap that bounds this selection's winner's curse.
                    "calibration_macro_mcc_inner_fit": sel["calibration_macro_mcc"],
                    "inner_check_macro_mcc": inner_check_macro,
                    "optimism_gap_macro": sel["calibration_macro_mcc"] - inner_check_macro,
                    # Micro F1: the iteration-17 objective, and the statistic compared against the
                    # trivial all-positive baseline -- so this gap bounds how much of any
                    # "beats the trivial baseline" claim is selection optimism.
                    "calibration_f1_inner_fit": sel["calibration_f1"],
                    "inner_check_f1": inner_check_f1,
                    "optimism_gap_f1": sel["calibration_f1"] - inner_check_f1,
                }
            )
    return pd.DataFrame(rows)


def summarize_heldout(heldout_df: pd.DataFrame) -> pd.DataFrame:
    """Matched subset only: shared_frozen vs the pipeline-conditioned config whose
    config_target_pipeline equals the evaluated target_pipeline.

    Metrics are MICRO-averaged: tp/fp/fn/tn are pooled across the group's images and the
    metrics derived once from those counts -- the same aggregation best_configuration() uses
    to select, so selection and evaluation finally measure the same quantity. The previous
    macro version (mean of per-image metrics) was badly biased here: pandas' mean drops NaN
    silently, so per-image MCC was averaged over only the 67-249 of 350 images where the
    detector predicted any positive at all -- not a random subset, precisely the images it
    fired on. That reported balanced accuracy ~0.50 and precision ~0.16 for a detector whose
    pooled precision is 0.21-0.33 against a prevalence of 0.149.

    Macro columns are kept alongside (macro_* prefix) with n_mcc_defined, so the gap between
    the two is visible rather than a matter of which function you happened to call.
    """
    matched = heldout_df[
        (heldout_df["configuration"] == "shared_frozen")
        | (heldout_df["config_target_pipeline"] == heldout_df["target_pipeline"])
    ].copy()
    matched["calibration"] = matched["configuration"]

    rows = []
    for (target_pipeline, calibration_name), group in matched.groupby(["target_pipeline", "calibration"]):
        # prevalence / precision_lift / lr_plus come from metrics_from_counts.
        row = metrics_mod.metrics_from_counts(**{k: int(group[k].sum()) for k in ("tp", "fp", "fn", "tn")})
        row["target_pipeline"] = target_pipeline
        row["calibration"] = calibration_name
        row["n_images"] = len(group)
        row["n_mcc_defined"] = int(group["mcc"].notna().sum())
        row["macro_mcc"] = group["mcc"].mean()
        row["macro_f1"] = group["f1"].mean()
        # Objective-consistent macro: silent predictions counted as 0, matching the metric
        # best_configuration now ranks on. Kept as a SEPARATE column rather than redefining
        # macro_mcc, so the NaN-dropped value stays comparable with earlier runs and the gap
        # between the two conventions is visible rather than a matter of which one you read.
        row["macro_mcc_silent_as_zero"] = group["mcc"].fillna(0.0).mean()
        row["macro_f1_silent_as_zero"] = group["f1"].fillna(0.0).mean()
        # A silent image is one where the detector predicted no pixel at all (51/349 on the
        # iteration-14 config): the "does it find the tampered zone at all" counter.
        row["n_silent_images"] = int(((group["tp"] + group["fp"]) == 0).sum())
        rows.append(row)

    lead = ["target_pipeline", "calibration", "n_images", "n_mcc_defined", "n_silent_images",
            "prevalence", "precision", "precision_lift", "lr_plus", "recall", "mcc",
            "macro_mcc_silent_as_zero", "f1"]
    summary = pd.DataFrame(rows)
    return summary[lead + [c for c in summary.columns if c not in lead]]
