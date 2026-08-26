"""Confusion-matrix metrics (incl. MCC), degenerate-baseline sanity check,
and raw score-map discrimination AUC. No sklearn -- everything here is a
few lines of numpy/scipy.stats.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Derive every metric from a confusion matrix's raw counts. Shared by compute_metrics
    (single image) and by callers that first pool tp/fp/fn/tn across images -- pooling counts
    before computing MCC (a "micro" average) avoids the instability of averaging per-image MCC
    directly (a "macro" average), which is dominated by noise on images with few predicted
    positives."""
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    balanced_accuracy = (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else np.nan

    # Python ints (arbitrary precision): mcc_denom can exceed int64 for large/pooled counts,
    # which np.sqrt cannot take directly -- math.sqrt handles it via a plain float conversion.
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(mcc_denom) if mcc_denom > 0 else np.nan

    total = tp + fp + fn + tn
    positive_ratio = (tp + fp) / total if total else np.nan
    prevalence = (tp + fn) / total if total else np.nan

    # Two prevalence-aware readings of the same tail, both more interpretable than F1 here:
    #   precision_lift -- how much better than guessing at the base rate. 1.0 is chance. It is
    #     the number that shows this detector carries signal at all, but it moves with
    #     prevalence, so only compare it across groups that share one.
    #   lr_plus (positive likelihood ratio, TPR/FPR) -- prevalence-INVARIANT, so it is the safe
    #     one for comparing populations whose class balance differs (e.g. a class-capped sample
    #     against the real one). Computing lift on a capped sample and reading it as if it were
    #     the real operating point inverts conclusions: on the per-image-capped sample the
    #     apparent best direction is "high", on the true population it is "low".
    precision_lift = precision / prevalence if prevalence and not np.isnan(precision) else np.nan
    lr_plus = recall / fpr if fpr else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "iou": iou,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_ratio": positive_ratio,
        "prevalence": prevalence,
        "precision_lift": precision_lift,
        "lr_plus": lr_plus,
    }


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, ~ground_truth).sum())
    fn = int(np.logical_and(~prediction, ground_truth).sum())
    tn = int(np.logical_and(~prediction, ~ground_truth).sum())

    return metrics_from_counts(tp, fp, fn, tn)


def degenerate_all_positive_baseline(masks_by_pipeline: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """Sanity check: predicting every pixel tampered must not win. One
    micro-aggregated row per pipeline for the held-out test masks."""
    rows = []
    for pipeline, masks in masks_by_pipeline.items():
        gt = np.concatenate([m.ravel() for m in masks])
        pred = np.ones_like(gt, dtype=bool)
        row = compute_metrics(pred, gt)
        row["target_pipeline"] = pipeline
        row["baseline"] = "all_positive"
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = ["target_pipeline", "baseline"] + [c for c in df.columns if c not in ("target_pipeline", "baseline")]
    return df[ordered]


def rank_sum_auc(authentic_scores: np.ndarray, tampered_scores: np.ndarray) -> dict:
    """Mann-Whitney U / (n_auth*n_tamp), no sklearn.
    auc_high = P(tampered score > authentic score); auc_low = 1 - auc_high."""
    n_auth, n_tamp = len(authentic_scores), len(tampered_scores)
    if n_auth == 0 or n_tamp == 0:
        return {
            "auc_high": np.nan, "auc_low": np.nan, "auc_two_sided": np.nan,
            "best_continuous_direction": None, "best_continuous_auc": np.nan,
            "authentic_median": np.nan, "tampered_median": np.nan,
            "n_authentic_samples": n_auth, "n_tampered_samples": n_tamp,
        }
    combined = np.concatenate([authentic_scores, tampered_scores])
    ranks = rankdata(combined)
    rank_sum_tampered = ranks[n_auth:].sum()
    u_tampered = rank_sum_tampered - n_tamp * (n_tamp + 1) / 2
    auc_high = u_tampered / (n_auth * n_tamp)
    auc_low = 1.0 - auc_high
    best_direction = "high" if auc_high >= auc_low else "low"
    return {
        "auc_high": auc_high,
        "auc_low": auc_low,
        "auc_two_sided": max(auc_high, auc_low),
        "best_continuous_direction": best_direction,
        "best_continuous_auc": max(auc_high, auc_low),
        "authentic_median": float(np.median(authentic_scores)),
        "tampered_median": float(np.median(tampered_scores)),
        "n_authentic_samples": n_auth,
        "n_tampered_samples": n_tamp,
    }


def detector_discrimination_auc(samples: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """samples maps (pipeline, window_size) -> (authentic_scores, tampered_scores)."""
    rows = []
    for (pipeline, window_size), (authentic, tampered) in samples.items():
        row = rank_sum_auc(authentic, tampered)
        row["pipeline"] = pipeline
        row["window_size"] = window_size
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = ["pipeline", "window_size"] + [c for c in df.columns if c not in ("pipeline", "window_size")]
    return df[ordered]
