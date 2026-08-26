"""Classical residual-based tamper localization baseline.

Dense sliding-window residual-energy detector: a reduction, across a small bank
of classical high-pass/band-pass residuals put on a common frozen scale, of
log1p(local variance of that residual). Not a single fixed 3x3 residual (see
residual_bank()), and NOT a PRNU/camera-fingerprint estimator -- CASIA2 has no
multi-shot bursts per camera to fit one.

Deliberately frozen: in this experiment the compression pipeline is the
independent variable, so the detector must not be tuned per codec.

The per-filter scales are frozen on calibration (fit_filter_scales) and the
per-image normalization is applied AFTER the fusion (normalize_score_map on the
fused map). That order is load-bearing, not stylistic: iteration 18 measured
that doing it the other way round made `normalize=True/False` two different
FUSIONS rather than two units of one map, and turned the unnormalized map -- the
one the published codec ranking is measured on -- into the Sobel branch alone
(99.9% of pixels). See EXPERIMENT_LOG.md 18.3 and detector_score_map.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    convolve,
    gaussian_laplace,
    generic_gradient_magnitude,
    label,
    sobel,
    uniform_filter,
)

_LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
_LOG_SIGMA = 2.0  # ponytail: fixed second residual scale, not swept by CALIBRATION --
                  # window_size already governs the local-variance aggregation scale.


def extract_noise_residual(gray: np.ndarray) -> np.ndarray:
    return convolve(gray.astype(np.float64), _LAPLACIAN, mode="reflect")


def log_residual(gray: np.ndarray, sigma: float = _LOG_SIGMA) -> np.ndarray:
    return gaussian_laplace(gray.astype(np.float64), sigma=sigma, mode="reflect")


def gradient_residual(gray: np.ndarray) -> np.ndarray:
    return generic_gradient_magnitude(gray.astype(np.float64), sobel, mode="reflect")


def residual_bank(gray: np.ndarray) -> list[np.ndarray]:
    """Classical high-pass/band-pass residuals with different spatial-frequency/
    orientation sensitivity, in place of a single fixed 3x3 Laplacian (see
    EXPERIMENT_LOG.md). Not a PRNU estimator -- CASIA2 has no multi-shot bursts
    per camera to fit one."""
    return [extract_noise_residual(gray), log_residual(gray), gradient_residual(gray)]


def local_variance_map(residual: np.ndarray, window_size: int) -> np.ndarray:
    mean_square = uniform_filter(residual**2, size=window_size, mode="reflect")
    square_mean = uniform_filter(residual, size=window_size, mode="reflect") ** 2
    return np.maximum(mean_square - square_mean, 0.0)


_CONTRAST_FACTOR = 4  # background box = factor * window, capped at half the short side


def local_contrast(score_map: np.ndarray, window_size: int, factor: int = _CONTRAST_FACTOR) -> np.ndarray:
    """Top-hat: the score minus its own local background box-mean.

    The threshold downstream is a GLOBAL quantile, so it answers "is this pixel extreme in this
    IMAGE". A splice is an anomaly relative to its NEIGHBOURHOOD, which is a different question --
    and it is why the objective's top candidates are `direction="low"` configurations that flag
    sky and flat walls (EXPERIMENT_LOG.md §20.1). Subtracting a local background converts the
    first question into the second. It is the same quantity the failure analysis measures after
    the fact (notebook section 8.2), which the detector itself never computed.

    Note this is a purely image-derived operation: it uses no mask and no filename, which is what
    separates it from that analysis covariate. The static guard in tests/ enforces the boundary by
    searching for the covariate's identifier, so it is deliberately not spelled out here.

    The background box is ``min(factor * window_size, short_side // 2)``. That cap is measured,
    not cosmetic: at ``8 * window`` the box exceeds the short side on 96.3% of the test images at
    window=64 and on 100% at 96/128 (short side: min 180, median 256). Once the box covers the
    frame, ``uniform_filter`` tends to the global mean, the top-hat degenerates into an affine
    shift, and ``normalize_score_map`` removes it -- the axis would be DEAD exactly at the frozen
    window. Half the short side keeps it a genuine neighbourhood at every window in the grid.
    Image dimensions are available at inference, so the cap is not a forbidden covariate.

    MUST be applied BEFORE normalize_score_map: subtracting a background from a map already
    re-centred per image sets the two steps against each other.
    """
    size = max(1, min(factor * window_size, min(score_map.shape) // 2))
    return score_map - uniform_filter(score_map, size=size, mode="reflect")


def to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Unweighted channel mean, not luma weights: the residuals below are texture-energy
    filters, and no part of this detector claims a perceptual model."""
    gray = np.asarray(image_rgb, dtype=np.float64)
    return gray.mean(axis=2) if gray.ndim == 3 else gray


def filter_score_maps(image_rgb: np.ndarray, window_size: int) -> list[np.ndarray]:
    """One raw score map per residual filter: log1p of the local variance of that residual.

    Split out of detector_score_map so the fusion across filters can be inspected and varied
    without recomputing the filters (EXPERIMENT_LOG.md 18). The expensive part of the detector is
    exactly this -- three residuals plus three uniform_filter passes -- so a caller that wants two
    fusions of the same image computes this once and reduces it twice.
    """
    gray = to_gray(image_rgb)
    return [np.log1p(local_variance_map(residual, window_size)) for residual in residual_bank(gray)]


# max, not mean: a splice may show up as an anomaly in only one filter (e.g. an edge-orientation
# artifact the isotropic Laplacian misses but the gradient filter catches); mean would dilute that
# signal with filters that saw nothing.
#
# min is not a second free hyperparameter, it is what "max" means read from the other end. The two
# tails ask different questions of the bank: the HIGH tail asks "is any filter anomalous here",
# which one filter can answer, so max; the LOW tail asks "is the residual energy gone", which is
# only true if it is gone in EVERY filter, so min. np.maximum.reduce is one-tailed and actively
# suppresses low-tail evidence -- a single filter with a middling response lifts the max out of
# the tail -- which is the candidate mechanism for the project's own internal contradiction:
# EXPERIMENT_LOG.md 12.1 found the >15% stratum (which dominates the pixels) informative in the
# low tail, while 15.4 found direction="low" the worst of the four directions.
_FUSIONS = {"max": np.maximum.reduce, "min": np.minimum.reduce}

_MAD_TO_STD = 1.4826  # consistency constant so MAD approximates std under normality


def filter_scales_from_maps(filter_maps) -> list[tuple[float, float]]:
    """(median, MAD*1.4826) per filter map -- the fitting primitive.

    Shared by fit_filter_scales (pooled over the calibration set, which is what the pipeline
    actually freezes) and by callers that only have one image to hand (tests, one-off inspection).
    """
    scales = []
    for score_map in filter_maps:
        median = float(np.median(score_map))
        scales.append((median, float(np.median(np.abs(score_map - median)) * _MAD_TO_STD)))
    return scales


def apply_filter_scales(filter_maps, scales, eps: float = 1e-6) -> list[np.ndarray]:
    return [(m - loc) / max(scale, eps) for m, (loc, scale) in zip(filter_maps, scales)]


def fit_filter_scales(
    filter_provider, image_ids: list[str], pipelines: list[str], window_size: int,
    pixels_cap: int, seed: int,
) -> list[tuple[float, float]]:
    """Frozen per-filter (loc, scale) from pooled AUTHENTIC calibration pixels across ALL arms.

    This is what makes the fusion across filters well posed, and it is the fix of iteration 18.
    The per-image alternative -- normalize_score_map on each filter map before the reduce, the
    pre-18 behaviour -- makes the fusion depend on the image AND on the arm, because the codec
    changes each filter's MAD and therefore which filter wins. Measured in EXPERIMENT_LOG.md 18.3:
    the argmax composition of the normalized map moves between arms on 91-99% of images, while on
    the unnormalized map the Sobel branch wins 99.9% of pixels and the "three-filter bank" is a
    single filter. A constant fitted once removes that degree of freedom from both directions.

    Pooled across arms deliberately: a per-arm scale would make the arms incomparable, exactly as
    a per-arm threshold would (CLAUDE.md 5.1). One subsample of pixel POSITIONS per (image, arm),
    shared by every filter -- the filters are being put on a common scale, so they must be fitted
    on the same pixels.

    filter_provider(image_id, pipeline, window_size) -> (list_of_filter_maps, mask_bool).
    """
    rng = np.random.default_rng(seed)
    pools = None
    for image_id in image_ids:
        for pipeline in pipelines:
            filter_maps, mask = filter_provider(image_id, pipeline, window_size)
            if pools is None:
                pools = [[] for _ in filter_maps]
            authentic = np.flatnonzero(~np.asarray(mask).ravel())
            if len(authentic) > pixels_cap:
                authentic = rng.choice(authentic, size=pixels_cap, replace=False)
            for pool, score_map in zip(pools, filter_maps):
                pool.append(score_map.ravel()[authentic])
    return filter_scales_from_maps([np.concatenate(pool) for pool in pools])


def detector_score_map(image_rgb: np.ndarray, window_size: int, scales, fuse: str = "max") -> np.ndarray:
    """Fused detector score, in the frozen units defined by `scales`.

    Two normalizations are needed here and they do two different jobs. Until iteration 18 they
    were collapsed into one `normalize` flag, which did the first one per image and the second not
    at all:

      * ACROSS FILTERS -- which filter's units dominate the reduction. Must be FROZEN (fitted once
        on calibration, see fit_filter_scales), otherwise the fusion itself changes with the image
        and with the arm, and `normalize=True/False` are two different detectors rather than two
        units of one map (measured: EXPERIMENT_LOG.md 18.3, median per-image Spearman 0.9566).
      * ACROSS IMAGES -- a busy-texture photo and a flat-sky photo have different baselines, so a
        frozen absolute threshold needs a common scale. Must be PER IMAGE, and must be applied
        AFTER the reduction (normalize_score_map on the fused map), or it changes the fusion.

    So the two flavours the experiment needs are now genuinely one map in two units:
        raw        = detector_score_map(image, w, scales)          # frozen units, absolute
        normalized = normalize_score_map(raw)                      # per-image, for the threshold
    normalize_score_map is strictly increasing, so every rank statistic (auc_high,
    abs_separability) is IDENTICAL on the two -- which is what CLAUDE.md 5.5 always meant, and
    what the notebook now asserts as a guard. Only the scale-dependent ones (separation, the class
    means) differ, and those are exactly the ones that measure energy loss.

    `scales` has no default on purpose: there is no safe implicit choice. Identity would silently
    give the single-filter Sobel map of 18.3, and a per-image fit would silently restore the
    pre-18 fusion.
    """
    return _FUSIONS[fuse](apply_filter_scales(filter_score_maps(image_rgb, window_size), scales))


def normalize_score_map(score_map: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust per-image z-score (median/MAD) of a raw score map.

    A frozen absolute threshold only makes sense if the score is on comparable units
    across images; raw local-variance energy is not (a busy-texture photo and a flat-sky
    photo have very different baselines). MAD is used instead of mean/std because it
    stays robust with a tampered region covering a minority of the image.
    """
    median = np.median(score_map)
    mad = np.median(np.abs(score_map - median)) * _MAD_TO_STD
    return (score_map - median) / max(mad, eps)


def apply_threshold(
    score_map: np.ndarray, direction: str, threshold_low: float | None, threshold_high: float | None,
    score_map_low: np.ndarray | None = None,
) -> np.ndarray:
    """direction 'low': flag score <= threshold_low as tampered (spliced regions score lower).
    direction 'high': flag score >= threshold_high. direction 'two_sided': either tail.
    Thresholds are empirical quantiles of pooled calibration-set authentic scores at the target
    FPR (see calibration.py) -- not a fixed mean+k*std rule.

    score_map_low is the map the LOW tail is read from, for the per-tail fusion variant of
    iteration 18: `max` across filters resolves the high tail, `min` the low one (see _FUSIONS).
    None -- the default -- reads both tails off score_map and is bit-identical to the pre-18
    behaviour, so every caller that does not opt in is unaffected.

    Budget note for 'two_sided' with two maps: on one map the two tail events are disjoint by
    construction, so the realised FPR is exactly target_fpr. With two maps they are NOT disjoint
    (a pixel can be low on the min map and high on the max map -- large spread across filters), so
    the realised FPR is <= target. Conservative, never inflated, but it means the two variants can
    sit at different operating points and must be compared at matched positive_ratio or not at all
    (CLAUDE.md 5.1, the 11.2 trap). The notebook measures the realised rate and gates on it.
    """
    low_map = score_map if score_map_low is None else score_map_low
    if direction == "low":
        return low_map <= threshold_low
    if direction == "high":
        return score_map >= threshold_high
    if direction == "two_sided":
        return (low_map <= threshold_low) | (score_map >= threshold_high)
    raise ValueError(f"unknown direction: {direction}")


def coherence(prediction: np.ndarray) -> float:
    """Largest connected component as a fraction of flagged pixels.

    A real spliced region is contiguous; flagging the wrong tail scatters pixels along texture
    edges. Uses no ground truth, so it is usable at inference -- which is the whole point:
    EXPERIMENT_LOG.md 12.1 showed the informative tail flips with tampered-region size (small
    splices score HIGH, large ones LOW), and region size is not available when predicting.
    """
    total = int(prediction.sum())
    if total == 0:
        return 0.0
    labelled, n_components = label(prediction)
    if n_components == 0:
        return 0.0
    return int(np.bincount(labelled.ravel())[1:].max()) / total


def regularize_mask(prediction: np.ndarray, radius: int, min_area_frac: float) -> np.ndarray:
    """Spatial prior on the predicted mask: open, close, then drop small components.

    apply_threshold decides each pixel independently, so authentic texture edges come back as
    scattered speckle -- there is no contiguity prior anywhere upstream. A spliced region is one
    blob, so: opening removes speckle, closing fills pinholes, and the area filter deletes
    components too small to be the region being looked for.

    radius=0 and min_area_frac=0.0 is the identity, so this is safe to call unconditionally and
    'no regularization' stays a real candidate in the calibration grid.
    """
    if radius <= 0 and min_area_frac <= 0.0:
        return prediction
    regularized = prediction
    if radius > 0:
        # Square structuring element of side 2*radius+1. ponytail: a square, not a disk --
        # separable, and at these radii the difference is a handful of corner pixels.
        element = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        # ponytail: scipy's default border_value=0 treats outside-the-image as background, so the
        # erosion trims up to `radius` pixels off a region that touches the image edge. At radius
        # 2-3 on 384px+ images that is a <=3px frame, and it errs toward FEWER false positives --
        # which is the direction being optimized. Pad-and-crop if border splices ever matter.
        regularized = binary_opening(regularized, structure=element)
        regularized = binary_closing(regularized, structure=element)
    if min_area_frac > 0.0:
        labelled, n_components = label(regularized)
        if n_components:
            areas = np.bincount(labelled.ravel())
            keep = areas >= min_area_frac * regularized.size
            keep[0] = False  # label 0 is background, never a component to keep
            regularized = keep[labelled]
    return regularized


def predict_mask(
    score_map: np.ndarray,
    direction: str,
    threshold_low: float | None,
    threshold_high: float | None,
    radius: int = 0,
    min_area_frac: float = 0.0,
    score_map_low: np.ndarray | None = None,
) -> np.ndarray:
    """Threshold + spatial regularization: the single entry point every caller uses.

    The notebook and calibration.py both predict masks; routing them through one function is what
    keeps the frozen operating point from drifting between selection and evaluation.

    direction 'coherence' is the size-agnostic rule: threshold BOTH tails at the full target FPR
    and keep whichever mask is spatially coherent. Note the FPR budget -- 'coherence' spends the
    whole target FPR on the one tail it keeps, exactly like 'low'/'high', and unlike 'two_sided'
    which splits it in half across both tails. So it is compared against them at a matched budget,
    not handicapped.

    radius=0, min_area_frac=0.0 (the defaults) reduce to apply_threshold exactly.

    score_map_low: the low tail's map for the per-tail fusion variant; see apply_threshold, whose
    budget note applies here too. None keeps the pre-iteration-18 behaviour exactly.
    """
    if direction == "coherence":
        low_map = score_map if score_map_low is None else score_map_low
        flagged_low = low_map <= threshold_low
        flagged_high = score_map >= threshold_high
        prediction = flagged_low if coherence(flagged_low) >= coherence(flagged_high) else flagged_high
    else:
        prediction = apply_threshold(score_map, direction, threshold_low, threshold_high, score_map_low)
    return regularize_mask(prediction, radius, min_area_frac)
