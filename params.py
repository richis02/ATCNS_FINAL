"""Single source of truth for paths, seeds, and calibration grids.

All values are pinned to the confirmed reference run
(results_pipeline_calibration/experiment_metadata.json), not guessed.
"""
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASIA_ROOT = PROJECT_ROOT / "CASIA2"
CASIA_TP_DIR = CASIA_ROOT / "Tp"
CASIA_MASK_DIR = CASIA_ROOT / "CASIA 2 Groundtruth"
CACHE_DIR = PROJECT_ROOT / "cache"
JPEG_AI_CACHE_DIR = CACHE_DIR / "jpeg_ai_official"
# One directory per run: results/<RUN_ID>/. Runs no longer overwrite each other, and the
# executed notebook is written next to the outputs it produced.
# FORENSICS_RUN_ID lets the runner pin the id it already chose, so notebook and outputs agree;
# without it (interactive use, fusion.py, window_strata.py) each execution stamps its own.
RESULTS_ROOT = PROJECT_ROOT / "results"
RUN_ID = os.environ.get("FORENSICS_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
RESULTS_DIR = RESULTS_ROOT / RUN_ID
# Created here, not in each consumer: the notebook, fusion.py and window_strata.py all write
# straight into it with to_csv/savefig, which do not create parent directories.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

# The frozen classical run every existing number was measured on: 500 images -> 151 calibration /
# 349 test, iteration 20. PINNED as a literal, never resolved through results/latest -- that
# symlink is mutable and EXPERIMENT_LOG.md 20.4 published a wrong number by trusting it
# (CLAUDE.md 5.11). dataset_validation.frozen_test_ids() RAISES if this run is missing.
#
# KEPT AT n=500 ON PURPOSE (EXPERIMENT_LOG.md 34, reverting 33's repoint). This run is the
# sensitivity floor of the three-instrument comparison (classical / mvss_defacto / learned_poolB),
# already measured paired on these same 349 images across all three -- rebuilding it at n=800
# would make it non-comparable with the other two and destroy exactly the property the comparison
# needs. A separate, honest n=800 exploratory run of the classical lineage exists at
# results/20260819-145824 (DATASET["max_images"] below) but is NOT wired in here.
FROZEN_RUN_ID = "20260818-141123"

# Share of the train/validation pool held out for early stopping, hyperparameters and thresholds
# of the trained detectors. Assigned by HOST, so whole source images move together.
VALIDATION_FRACTION = 0.15

# The frozen Phase 1 dataset run: manifest + split every Phase 2/3/4 module resolves through.
# PINNED, like FROZEN_RUN_ID, and phase2_common.split_frame() RAISES if it is missing.
DATASET_RUN_ID = "20260818-210933"

# ---------------------------------------------------------------------------------------------
# Phase 2 (trained detectors). PRE-REGISTERED in EXPERIMENT_LOG.md 23 before the first training
# run, and declared here so the code and the log cannot drift apart.
#
# GOVERNING RULE: the detector is selected ONLY on the uncompressed reference arm, BEFORE any
# codec is involved, then frozen completely. No Phase 2 module imports forensics.compression.

# Trivial all-positive F1 per split -- prevalence is NOT constant across splits, so neither is the
# baseline. Measured in Phase 1; phase2_common asserts these against the manifest at run time
# rather than trusting the literals.
# CORRECTED before any Phase 2 number existed: ("A","train") was first written 0.2136, which is
# the figure for the WHOLE of pool A (train + validation, prevalence 0.11959). Pool A TRAIN alone
# has prevalence 0.11756 -> 0.2104. phase2_common.assert_declared_baselines() recomputes all five
# from the manifest and RAISES on a mismatch, which is how the slip was caught.
TRIVIAL_BASELINE_F1 = {
    ("A", "train"): 0.2104, ("A", "validation"): 0.2307,
    ("B", "train"): 0.2098, ("B", "validation"): 0.1944,
    ("test", "test"): 0.2580,
}

# The acceptance gates of EXPERIMENT_LOG.md 23.4, in figures. Fixed BEFORE any Phase 2 number
# exists: in front of a borderline candidate an adjective becomes a discretionary decision.
PHASE2_GATES = {
    "silent_rate_abs_max": 0.02,        # 23.4a, below the incumbent 3.44% on the test reference arm
    "silent_rate_rel_max": 0.5,         # 23.4a, vs baseline_refit on the SAME split
    "val_test_margin_gap_max": 0.05,    # 23.4b, on the margin over each split's own baseline
    "large_region_recall_ratio_min": 0.50,   # 23.4c
    "large_region_recall_abs_min": 0.1794,   # 23.4c, the incumbent's own >15% recall
    "delta_f1_relevance": 0.01,         # 23.6, above this the finding propagates to 19.1/Q2
}

LEARNING = {
    "encoder": "resnet34",       # torchvision IMAGENET1K_V1. ImageNet classification only: no
                                 # forensic dataset, so no contamination channel (21.1).
    "crop": 256,
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 120,
    "patience": 15,
    "focal_gamma": 2.0,
    "focal_weight": 0.5,         # DECLARED, not tuned: tuning these would grow the selection grid
    "dice_weight": 0.5,
    "pad_multiple": 32,          # inference pads to this and crops back -- never resizes (5.3)
    "jitter_brightness": 0.15,   # applied to the WHOLE image, never inside-vs-outside the mask
    "jitter_contrast": 0.15,
}

DATASET = {
    # FORENSICS_MAX_IMAGES exists so a smoke run can execute the whole notebook end to end on a
    # handful of images. The split permutes then truncates, so a smaller value selects a PREFIX
    # of the same images -- the JPEG AI cache stays valid and the reported run is unaffected.
    "max_images": int(os.environ.get("FORENSICS_MAX_IMAGES", 800)),
    "calibration_fraction": 0.3,
    "splicing_only": True,  # Tp_D_ only, excludes Tp_S_ copy-move
}

JPEG_AI_MIN_SIDE = 161  # spatial floor with tools_on/eICCI; pre-split exclusion, not resize/pad

PIPELINE_LABELS = ["jpeg", "osn", "jpeg_ai"]

# The uncompressed reference arm. "original" is NOT a codec: it is the CASIA2 file as
# distributed, run through the same detector with the same frozen parameters. Every
# preservation number is measured against it.
# Honesty: most CASIA2 Tp_D_ files are already JPEG, so "original" means "no ADDITIONAL
# compression", not "lossless". Its bpp is not rate-controlled and is reported as context
# only -- it takes no part in the matched-rate comparison.
REFERENCE_ARM = "original"
EVAL_ARMS = [REFERENCE_ARM] + PIPELINE_LABELS

# Rate matching is on bpp per SOURCE pixel (compression.compression_quality), the only
# denominator under which a resizing pipeline and a non-resizing one are comparable, and it is
# done PER IMAGE (compression._encode_jpeg_at_bpp bisects quality) rather than by fixing a
# quality knob. A fixed quality is a fixed quantiser scale, not a fixed rate: quality=13 landed
# at 0.42 bpp on the test set against jpeg_ai's 0.52 -- a 22% gap in the learned codec's favour,
# because jpeg_ai rate-controls per image and JPEG does not.
# RATE_TARGET_BPP is jpeg_ai's *measured* mean rate, not its nominal target_bpp=0.5: the codec
# overshoots slightly, and the comparison has to match what it actually spent.
# The earlier quality=15/10 pairing was matched on bpp per *delivered* pixel, which let osn
# spend 0.158 bpp of the source while reporting 0.408 -- a 3x rate advantage read as a matched
# operating point (retracted in EXPERIMENT_LOG.md).
RATE_TARGET_BPP = 0.5166
JPEG_PARAMS = {"target_bpp": RATE_TARGET_BPP, "subsampling": 2}  # optimize=True fixed in jpeg_compress
# max_dim=256: CASIA2 images here are 384-800px (median 384) so the old 1280 never resized a
# single one -- osn_compress's resize step was dead code for this dataset, making "osn" a no-op
# duplicate of "jpeg" at matched quality. 256 is below every image in the sample, so the
# pipelines are actually distinct. Trade-off: this is well below what real OSNs resize to
# (1080-2048px) -- "OSN-like" here means "downsample+recompress stress test", not a claim about
# a specific platform's actual threshold (see notebook's own disclaimer on this).
# osn_compress now restores the source resolution after recompressing, so osn is scored on the
# same pixel grid, the same unresampled mask and the same window scale as the other two, and it
# is held to the same per-source-pixel rate target.
OSN_PARAMS = {"max_dim": 256, "target_bpp": RATE_TARGET_BPP, "subsampling": 2}

JPEG_AI_ROOT = Path("/home/bundu/ACS/jpeg-ai-reference-software")
JPEG_AI_RUNNER = ["/home/bundu/miniconda3/envs/jpeg_ai_vm/bin/python"]
JPEG_AI_REVISION = "b0f832cde85bd873fbfa995668b7d7c8d0b0c2fe"
JPEG_AI_TARGET_BPP = 0.5
JPEG_AI_PROFILE = "high"
JPEG_AI_TOOLS = "on"
JPEG_AI_TIMEOUT_S = 1800

# PRE-REGISTERED operating point. Fixed BEFORE the experiment and never re-chosen by looking
# at the test set: the research question is which codec preserves the forensic signal, not
# which threshold flatters which codec. 0.10 is the FPR budget the earlier calibration runs
# converged on against a prevalence of 0.148 (EXPERIMENT_LOG.md), so it is a defensible prior,
# and it is spent identically by all four arms.
FROZEN_TARGET_FPR = 0.10

# Region-size strata, shared by the notebook, window_strata.py and fusion.py (each used to
# define its own copy). Prevalence varies ~32x across these strata (1.1% vs 35.7%), which is why
# lr_plus (TPR/FPR, prevalence-invariant) is the statistic compared ACROSS them.
REGION_BINS = [0, 0.02, 0.05, 0.15, 1.01]
REGION_LABELS = ["<2%", "2-5%", "5-15%", ">15%"]

# Deliberately small: 5 windows x 4 directions x 1 postprocess x 2 local_contrast x 1 FPR = 40
# candidates from iteration 20 (60 before it, and 60 is still the count of what the PROJECT has
# explored -- see the postprocess entry below for why that distinction matters), all scored
# on the CALIBRATION set only, one winner, frozen. The previous ~1800-candidate sweep existed to
# answer "does each codec want its own calibration?" -- a question this experiment no longer
# asks, and whose per-codec thresholds would make the codecs incomparable.
#
# Iteration 15 grew the grid from 9 to 36 to fix a detector that was barely above chance at its
# operating point (test MCC 0.006-0.011). Two axes were added, both classical, both fitted on
# calibration only:
#   directions += "coherence" -- per image, keep whichever tail forms the spatially coherent mask.
#     EXPERIMENT_LOG.md 12.1: the informative tail FLIPS with tampered-region size, so any single
#     global direction is anti-correlated on ~150/349 test images. Region size is not available at
#     inference; spatial coherence is.
#   postprocess -- (radius, min_area_frac) for detector.regularize_mask. Nothing upstream imposed
#     a contiguity prior, so authentic texture edges came back as scattered speckle.
# (0, 0.0) is kept in the list on purpose: "no regularization" has to be able to win. Iteration
# 14.1 showed grid size drives the winner's curse (1800 -> 9 cut the optimism gap from 0.06-0.17
# to 0.005-0.023), so if calibration_selection_diagnostic shows the gap climbing back past ~0.05
# this list gets trimmed rather than the number buried.
CALIBRATION = {
    # Iteration 16 added 96 and 128. Iteration 15's winner landed on 64, the largest value in
    # [16, 32, 64] -- an argmax at the edge of its own grid says the optimum may be outside it, and
    # 32 -> 64 is exactly where iteration 15's gain came from, so this is the cheapest open lead.
    # Geometric context: images here have a median SHORTER side of 256px (min 180), so window=128
    # spans half the short dimension and its "local" variance is nearly global. That is not
    # excluded a priori -- it is what selection_positive_ratio_bounds is there to catch if the map
    # degenerates -- but a win at 128 would need reading as "the detector prefers a near-global
    # contrast statistic", not as better localization.
    "windows": [16, 32, 64, 96, 128],
    "target_fpr_grid": [FROZEN_TARGET_FPR],
    "directions": ["high", "low", "two_sided", "coherence"],
    # Iteration 20 pruned this axis from 3 settings to 1, to pay for `local_contrast` without
    # growing the grid: 5 x 4 x 1 x 2 = 40 candidates, against 60 before.
    #
    # HONESTY ABOUT WHAT THIS PRUNING IS. It is DATA-INFORMED, not an a-priori constraint:
    # (0, 0.0) won every freeze the project ever did (EXPERIMENT_LOG.md §15, §16.2, §17, §18.6)
    # and on iteration 19's grid the three settings sat within 0.0016 micro F1 of each other. But
    # that conclusion comes from HAVING LOOKED at previous runs. The grid actually explored across
    # the project remains 60, and optimism_gap_f1 measured on these 40 has to be reported with
    # that note attached -- 40 was never the size of the problem, it is its size after a choice
    # made with hindsight.
    #
    # (0, 0.0) is therefore NOT "an axis removed", it is the FROZEN VALUE of the post-processing:
    # frozen_config states it explicitly so it reads as a decision and not as an omission, and
    # n_silent_images stays in every report even though the knob that moved it is gone -- it is
    # the statistic that describes failure mode A.
    "postprocess": [(0, 0.0)],  # (radius, min_area_frac) -- FROZEN, see above
    # Iteration 20's one new axis, binary. The threshold is a global quantile, so it flags what is
    # extreme in the IMAGE; a splice is an anomaly relative to its NEIGHBOURHOOD (§20.1, §20.3).
    # True applies detector.local_contrast to the fused map BEFORE the per-image normalization.
    # One axis and no more: with optimism_gap_f1 at +0.1436 there is no room for a second.
    "local_contrast": [False, True],
    # Selection objective. Three values are honoured by calibration.best_configuration:
    #   "macro_mcc" -- mean of PER-IMAGE MCC, silent (undefined) images counted as 0
    #   "micro_f1"  -- F1 from pixel-pooled tp/fp/fn/tn
    #   anything else -- micro MCC (the historical default)
    # Whichever is chosen, the other two stay reported next to it: the point is to DECLARE the
    # objective, not to hide the numbers it loses on.
    #
    # Iteration 17 changed this from "macro_mcc" to "micro_f1". Iterations 15-16 optimized the
    # macro because 13.2 showed micro and macro disagree and the pixel-weighted aggregate had
    # hidden a detector that was actively harmful on half the dataset. 16.2 then found, ON
    # CALIBRATION, that `w=96 coherence r=0` is the first configuration of the whole project to
    # beat the trivial all-positive baseline (micro F1 0.2676 vs 0.2394) with LR+ > 1 in every
    # stratum and 5 silent images instead of 80 -- and that it loses only on the macro. The
    # decision and its two declared costs (the sanity-check metric is now the objective, so
    # optimism_gap_f1 is reported; and the macro drops 0.0927 -> 0.0738) are in EXPERIMENT_LOG 17.
    # "micro_mcc" was rejected: it picks w=128 by a 0.0003 margin -- a near-global window with a
    # worst-stratum LR+ of 1.14 against 1.71.
    "selection_objective": "micro_f1",
    # Reject any candidate that is anti-correlated (lr_plus <= 1, i.e. worse than chance) in ANY
    # region-size stratum. Without this a candidate can win on macro-MCC while still being useless
    # on small splices, which is the failure mode this iteration exists to remove.
    "require_positive_lr_plus_per_stratum": True,
    "selection_max_macro_fpr": 0.15,
    "selection_positive_ratio_bounds": (0.001, 0.4),
    "calibration_pixels_per_class_per_map": 2048,
}

# The previously frozen operating point, kept so each iteration reports a like-for-like
# before/after on the SAME test pass instead of splicing numbers across two runs (different
# sample, different threshold pool, nothing controlled). Declared here rather than hardcoded in
# the notebook so what is being compared against is explicit.
# Iteration 18 keeps this at w=64/two_sided/r=0 rather than advancing it to iteration 17's
# w=96/coherence, and the reason is that it now does a better job than a plain before/after:
# iteration 18 froze exactly this geometry (with the per-tail fusion), so scoring this entry --
# which always uses the all_max fusion -- on the same test pass gives a CONTROLLED, fusion-only
# comparison: same window, same direction, same post-processing, same images, same mask. The only
# difference is the reduction across filters. Iteration 17's geometry is not lost either: the
# alternative-variant threshold set is w=96/coherence/all_max, which is exactly it.
# It also still IS iterations 15-16's frozen point, so the historical before/after survives.
PREVIOUS_FROZEN_CONFIG = {
    "window_size": 64,
    "direction": "two_sided",
    "radius": 0,
    "min_area_frac": 0.0,
    "label": "all_max_same_geometry",
}

# Label for the configuration THIS run freezes, used in the before/after and per-stratum outputs.
# One constant rather than a string literal repeated across the notebook's result cells: those
# copies went stale the moment PREVIOUS_FROZEN_CONFIG advanced.
RUN_LABEL = "iteration18"

PRESERVATION = {
    # Per-image preservation RATIOS are undefined when the reference arm scores ~0: the
    # detector is weak, F1_original is 0 on many images, and x/0 is not "infinite
    # preservation". Below this floor the ratio is NaN and the image is counted as excluded.
    "ratio_floor": 0.05,
}

# Bins over the REALISED bpp, for the rate-conditioned view. Everything is encoded at one
# target rate, so these bins only resolve the spread around it (images whose content cannot
# reach the target at any quality). Empty bins are dropped and reported, never back-filled.
RATE_BINS = [0.30, 0.40, 0.50, 0.60, 0.70]

BOOTSTRAP_REPETITIONS = 2000

# ---------------------------------------------------------------------------------------------
# Iteration 21: the DEEP comparison arm (deep_upper_bound.py). Inference only, published weights.
#
# CONTAMINATION AUDIT (EXPERIMENT_LOG.md 21.1). CAT-Net v2 and TruFor both declare CASIA v2 in
# their training data and are unusable on this test set at any price. IF-OSN and PSCC-Net do not
# declare theirs verifiably -> UNKNOWN, which is not "clean", so they are excluded. MVSS-Net
# publishes two checkpoints and they land on opposite sides of the audit, so BOTH are run:
#   mvss_defacto -- trained on DEFACTO-84k only, never saw CASIA. CLEAN, and the only one any
#                   conclusion is read from. Being out of domain, it gives a FLOOR, not a ceiling.
#   mvss_casia   -- trained on CASIA v2. CONTAMINATED, and labelled so in every row of every CSV.
#                   Ceiling only; no decision is ever taken on its numbers.
# The two are never averaged and never pooled.
MVSS_ROOT = PROJECT_ROOT / "third_party" / "mvssnet"   # models/mvssnet.py + models/resfcn.py, verbatim
# Downloaded with gdown from the Google Drive folder linked in the MVSS-Net README
# (drive.google.com/drive/folders/1CztGkd91xF1QqEXuc2n8rVDTBJ7X695U), which is why the directory
# keeps the Drive folder's own name: the provenance of a weight file is part of the experiment.
# deep_upper_bound.load_mvss RAISES on a missing file rather than guessing (CLAUDE.md 5.11), and
# it records size + sha256 of whatever it actually loaded in deep_threshold.json.
MVSS_WEIGHTS = {
    "mvss_defacto": PROJECT_ROOT / "weights" / "mvssnet_model" / "mvssnet_defacto.pt",
    "mvss_casia": PROJECT_ROOT / "weights" / "mvssnet_model" / "mvssnet_casia.pt",
}
MVSS_CONTAMINATED = {"mvss_defacto": False, "mvss_casia": True}
# The published inference resolution. Only the model INPUT is resized to it; the output score map
# comes back to the source grid and the mask is never resampled (CLAUDE.md 5.3).
MVSS_INPUT_SIZE = 512
