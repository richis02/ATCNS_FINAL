"""PHASE 4 -- TEST PASS n.2. The codec comparison with the frozen learned detector.

THE QUESTION, verbatim (EXPERIMENT_LOG.md 29.0), and it is bound to the instrument:

    How much of what a TRAINED localizer can detect survives each compression pipeline?

Not "how much forensic evidence exists", and NOT "how much does each codec alter splicing seams" --
the section 28 diagnostics do not support that reformulation: not one of the three seam criteria is
met, eroded recall RISES to 106%, and ablating the +-8 px band costs 7.6%.

GOVERNING RULE (verbatim, and this phase is the reason it exists):

    The detector is selected ONLY on its ability to localize tampering in the uncompressed
    reference arm, BEFORE any codec is involved. It is then frozen completely -- weights,
    preprocessing, thresholds, postprocessing, feature extraction, scale fusion,
    hyperparameters. Only then may we ask which codec best preserves what the frozen detector
    can detect. The detector is never modified, reselected, or retuned to favour any codec.

Instrument: learned_poolB ONLY. learned_poolA failed a pre-registered gate (26.1) and its better
lr_plus is inflated by the donor leakage measured in 26.2 -- using it would be selecting the
detector on the leak that inflates it. Every number here is IN-DOMAIN (trained on CASIA 2).

Threshold: 0.5842, frozen in Phase 2 on the authentic pixels of pool B's VALIDATION REFERENCE ARM
at the pre-registered target_fpr 0.10. Applied IDENTICALLY to all four arms. No per-codec tuning,
no re-estimation, one shared threshold (5.1).

Reported TWICE, per 29.6: on the full map, and with the |d| <= 8 band around the mask boundary
EXCLUDED from the measurement on all four arms alike (excluded, not zeroed -- see the loop below).
The osn arm resamples to 256 px and so reworks exactly those boundaries; the difference between the
two readings IS the seam contribution, measured instead of left as a caveat.

Run in .venv-deep:
    .venv-deep/bin/python codec_preservation.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path[:0] = ["src", "configs"]

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import params
import phase2_common as common
from forensics import compression, metrics, preservation, stats

QUESTION = ("How much of what a TRAINED localizer can detect survives each compression pipeline?")
IN_DOMAIN_NOTE = "IN-DOMAIN (trained on CASIA 2)"
SEAM_BAND = 8            # 29.6: the band that carries 7.6% of pool B's lr_plus
MVSS_WORST_RETENTION = 0.945   # 29.4: mvss_defacto's LEAST degraded arm, the P1 threshold

JPEG_AI_KWARGS = dict(
    cache_dir=params.JPEG_AI_CACHE_DIR, jpeg_ai_root=params.JPEG_AI_ROOT,
    runner=params.JPEG_AI_RUNNER, target_bpp=params.JPEG_AI_TARGET_BPP,
    profile=params.JPEG_AI_PROFILE, tools=params.JPEG_AI_TOOLS, timeout_s=params.JPEG_AI_TIMEOUT_S,
)
ARM_KWARGS = {"jpeg": params.JPEG_PARAMS, "osn": params.OSN_PARAMS, "jpeg_ai": JPEG_AI_KWARGS}


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="smoke; nothing written")
    args = parser.parse_args()

    print(__doc__.split("Run in .venv-deep:")[0].strip())
    print("=" * 104)
    common.assert_declared_baselines()
    records = common.manifest()
    test_ids = common.ids_for("test", allow_test=True)
    if args.limit:
        test_ids = test_ids[: args.limit]
        print("SMOKE: {} images, nothing written.".format(len(test_ids)))
    strata = common.stratum_of(test_ids)
    print("TEST PASS n.2 (29.3) -- {} images x {} arms. Nothing is selected: this is a "
          "measurement.".format(len(test_ids), len(params.EVAL_ARMS)))

    import torch
    import detector_learning as learning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(params.RESULTS_ROOT / "phase2-learned-B" / "learned_poolB.pt",
                      map_location=device, weights_only=False)
    model = learning.UNetResNet34(pretrained=False).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    threshold = float(blob["threshold"])
    print("instrument: learned_poolB  [{}]  frozen threshold {:.4f}, one shared threshold on all "
          "four arms".format(IN_DOMAIN_NOTE, threshold))

    rows = []
    for n, image_id in enumerate(test_ids, 1):
        record = records.loc[image_id]
        source = np.asarray(Image.open(record["image_path"]).convert("RGB"))
        with Image.open(record["mask_path"]) as handle:
            mask = np.asarray(handle.convert("L")) >= 128
        distance = signed_distance(mask)
        band = np.abs(distance) <= SEAM_BAND

        for arm in params.EVAL_ARMS:
            kwargs = {"source_path": record["image_path"]} if arm == params.REFERENCE_ARM else ARM_KWARGS[arm]
            result = compression.apply_compression(source, arm, **kwargs)
            # Same source pixel grid on every arm, or the mask and the prevalence stop matching.
            assert result.image.shape[:2] == source.shape[:2], "{} changed geometry".format(arm)
            logits = learning.predict_logits(model, result.image, device)
            probability = 1.0 / (1.0 + np.exp(-logits))
            quality = compression.compression_quality(source, result)

            # ONE operation for both statistic families: the band takes NO PART in the measurement,
            # identically on all four arms. An earlier draft zeroed the band for the binary metrics
            # but EXCLUDED it for separability -- two different operations under one label, which
            # is the defect 28.4b exists to name. Exclusion is the coherent choice for a rank
            # statistic: zeroing would pile every band pixel onto one value and distort the AUC in
            # a way no arm-to-arm comparison could read.
            for view in ("full", "seam_excluded"):
                keep = np.ones_like(mask) if view == "full" else ~band
                prediction = (probability >= threshold)[keep]
                counts = metrics.compute_metrics(prediction, mask[keep])
                # raw logits for separability: an absolute scale, no per-image normalisation to
                # re-inflate what the codec just destroyed (5.5).
                sep = preservation.separability(logits[keep], mask[keep])
                rows.append({
                    "image_id": image_id, "arm": arm, "view": view, "stratum": strata[image_id],
                    "in_domain": True, "f1": counts["f1"], "iou": counts["iou"],
                    "mcc": counts["mcc"], "recall": counts["recall"],
                    "precision": counts["precision"], "lr_plus": counts["lr_plus"],
                    "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
                    "n_predicted": int(prediction.sum()), "n_pixels_used": int(keep.sum()),
                    "abs_separability": sep["abs_separability"], "auc_high": sep["auc_high"],
                    "separation": sep["separation"],
                    "bpp": quality["bpp"], "psnr": quality["psnr"], "ssim": quality["ssim"],
                })
        if n % 50 == 0:
            print("  {}/{}".format(n, len(test_ids)), file=sys.stderr, flush=True)

    per_image = pd.DataFrame(rows)

    # ---- paired preservation, per view -----------------------------------------------------
    summaries, paired_rows = [], []
    for view in ("full", "seam_excluded"):
        block = per_image[per_image["view"] == view]
        wide = block.pivot(index="image_id", columns="arm")
        for arm in params.PIPELINE_LABELS:
            reference_sep = wide[("abs_separability", params.REFERENCE_ARM)]
            arm_sep = wide[("abs_separability", arm)]
            retention = [preservation.preservation_ratio(a, r, params.PRESERVATION["ratio_floor"])
                         for a, r in zip(arm_sep, reference_sep)]
            delta_sep = (arm_sep - reference_sep).to_numpy()
            delta_f1 = (wide[("f1", arm)] - wide[("f1", params.REFERENCE_ARM)]).to_numpy()
            ci_sep = stats.bootstrap_ci(delta_sep, params.BOOTSTRAP_REPETITIONS, params.RNG_SEED,
                                        statistic=np.median)
            ci_f1 = stats.bootstrap_ci(delta_f1, params.BOOTSTRAP_REPETITIONS, params.RNG_SEED,
                                       statistic=np.median)
            paired_rows.append({
                "view": view, "arm": arm, "in_domain": True,
                "median_separability_retention": float(np.nanmedian(retention)),
                "n_retention_defined": int(np.sum(~np.isnan(retention))),
                "median_delta_separability": ci_sep["point_estimate"],
                "sep_ci95_low": ci_sep["ci95_low"], "sep_ci95_high": ci_sep["ci95_high"],
                "sep_interpretation": ci_sep["interpretation"],
                "median_delta_f1": ci_f1["point_estimate"],
                "f1_ci95_low": ci_f1["ci95_low"], "f1_ci95_high": ci_f1["ci95_high"],
                "n_paired": len(delta_sep),
            })
        for arm in params.EVAL_ARMS:
            arm_block = block[block["arm"] == arm]
            micro = metrics.metrics_from_counts(*(int(arm_block[c].sum()) for c in ("tp", "fp", "fn", "tn")))
            micro.update({"view": view, "arm": arm, "in_domain": True,
                          "median_abs_separability": float(arm_block["abs_separability"].median()),
                          "n_silent_images": int((arm_block["n_predicted"] == 0).sum()),
                          "median_bpp": float(arm_block["bpp"].median()),
                          "mean_psnr": float(arm_block["psnr"].replace(np.inf, np.nan).mean()),
                          "mean_ssim": float(arm_block["ssim"].mean())})
            summaries.append(micro)

    summary_df = pd.DataFrame(summaries)
    paired_df = pd.DataFrame(paired_rows)

    # ---- report ------------------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("QUESTION: {}".format(QUESTION))
    print("Instrument-bound. learned_poolB, {}. NOT a statement about how much forensic".format(IN_DOMAIN_NOTE))
    print("evidence exists, and NOT 'how much does each codec alter seams' (28 does not support it).")
    print("=" * 104)

    for view in ("full", "seam_excluded"):
        label = "FULL MAP" if view == "full" else "SEAM-EXCLUDED (|d| <= {} px takes no part, all four arms alike)".format(SEAM_BAND)
        print("\n--- {} ---".format(label))
        block = summary_df[summary_df["view"] == view]
        print(block[["arm", "f1", "recall", "precision", "iou", "mcc", "lr_plus",
                     "median_abs_separability", "n_silent_images", "median_bpp",
                     "mean_psnr", "mean_ssim"]].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        pblock = paired_df[paired_df["view"] == view]
        print(pblock[["arm", "median_separability_retention", "median_delta_separability",
                      "sep_ci95_low", "sep_ci95_high", "sep_interpretation", "median_delta_f1"]]
              .to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    # ---- P1, the falsifiable prediction (29.4) -----------------------------------------------
    full_paired = paired_df[paired_df["view"] == "full"]
    best_retention = float(full_paired["median_separability_retention"].max())
    confirmed = best_retention <= MVSS_WORST_RETENTION
    print("\n" + "=" * 104)
    print("P1 -- FALSIFIABLE PREDICTION, written in 29.4 before these numbers existed")
    print("=" * 104)
    print("  prediction: learned_poolB's BEST median separability retention <= {:.3f}".format(
        MVSS_WORST_RETENTION))
    print("              (mvss_defacto's least-degraded arm; classical sits at ~1.00)")
    print("  observed  : {:.4f}  ({})".format(
        best_retention, full_paired.loc[full_paired["median_separability_retention"].idxmax(), "arm"]))
    print("  ->  {}".format("CONFIRMED" if confirmed else "FALSIFIED"))
    if confirmed:
        print("      The classical null result (19.1) was a SENSITIVITY FLOOR of the instrument,")
        print("      now demonstrated on three points rather than argued.")
    else:
        print("      learned_poolB degrades LESS than mvss_defacto. The prediction is falsified and")
        print("      MVSS's -5..-10% needs another explanation -- plausibly that an out-of-domain")
        print("      model rides low-level statistics that compression touches first. Reported as a")
        print("      falsification, NOT rewritten.")

    # ---- three instruments together (29.5) ----------------------------------------------------
    print("\n" + "=" * 104)
    print("THREE INSTRUMENTS, SAME {} IMAGES, SAME FOUR ARMS (29.5) -- median separability retention".format(len(test_ids)))
    print("Phase 4 does NOT replace the earlier runs: these are three points on a sensitivity")
    print("scale, not three attempts of which the last one wins.")
    print("=" * 104)
    ladder = []
    frozen_meta = json.loads((params.RESULTS_ROOT / params.FROZEN_RUN_ID / "experiment_metadata.json").read_text())
    for entry in frozen_meta["final_ranking"]:
        ladder.append({"instrument": "classical (iteration18)", "in_domain": False,
                       "f1_original": frozen_meta["all_positive_baseline_f1"] * 0 + 0.2000,
                       "arm": entry["pipeline"],
                       "median_separability_retention": entry["median_separability_retention"]})
    deep = pd.read_csv(params.RESULTS_ROOT / "20260818-171351" / "deep_paired_comparison.csv")
    deep = deep[(deep["detector"] == "mvss_defacto") & (deep["metric"] == "raw_abs_separability")]
    for row in deep.itertuples():
        ladder.append({"instrument": "mvss_defacto (clean, OUT OF DOMAIN)", "in_domain": False,
                       "f1_original": 0.1663, "arm": row.arm,
                       "median_separability_retention": row.median_retention})
    for row in full_paired.itertuples():
        ladder.append({"instrument": "learned_poolB [{}]".format(IN_DOMAIN_NOTE), "in_domain": True,
                       "f1_original": float(summary_df[(summary_df.view == "full") & (
                           summary_df.arm == params.REFERENCE_ARM)]["f1"].iloc[0]),
                       "arm": row.arm,
                       "median_separability_retention": row.median_separability_retention})
    ladder_df = pd.DataFrame(ladder)
    print(ladder_df.pivot(index=["instrument", "f1_original"], columns="arm",
                          values="median_separability_retention").to_string(
                              float_format=lambda x: "{:.4f}".format(x)))
    print("\n  Threshold-protocol difference between instruments, declared (29.2): the classical and")
    print("  MVSS thresholds were pooled over ALL FOUR ARMS of the 151 calibration images;")
    print("  learned_poolB's is REFERENCE ARM ONLY. 23.6 measured that difference on the classical:")
    print("  dF1 = +0.0011, negligible. Real difference, written rather than deduced.")
    print("  mvss_casia (CONTAMINATED) is excluded from every conclusion (21.1).")

    # ---- seam contribution, isolated (29.6) ---------------------------------------------------
    print("\n" + "=" * 104)
    print("SEAM CONTRIBUTION, ISOLATED (29.6) -- retention full vs seam-excluded")
    print("=" * 104)
    seam = full_paired.set_index("arm")["median_separability_retention"].to_frame("full")
    seam["seam_excluded"] = paired_df[paired_df["view"] == "seam_excluded"].set_index("arm")[
        "median_separability_retention"]
    seam["difference"] = seam["full"] - seam["seam_excluded"]
    print(seam.to_string(float_format=lambda x: "{:.4f}".format(x)))
    print("\n  osn is the only arm that reworks those boundaries (resample to 256 px). If its")
    print("  difference stands out against jpeg and jpeg_ai -- which are the control -- that")
    print("  difference IS the seam contribution, measured rather than left as a caveat.")

    # ---- stop condition (29.7) ----------------------------------------------------------------
    print("\n" + "=" * 104)
    print("STOP CONDITION (29.7) -- is the codec effect concentrated in the <2% stratum?")
    print("=" * 104)
    stratum_rows = []
    full_block = per_image[per_image["view"] == "full"]
    wide = full_block.pivot(index="image_id", columns="arm", values="abs_separability")
    stratum_of_id = {i: strata[i] for i in wide.index}
    for arm in params.PIPELINE_LABELS:
        delta = wide[arm] - wide[params.REFERENCE_ARM]
        for label in params.REGION_LABELS:
            selected = [i for i in wide.index if stratum_of_id[i] == label]
            stratum_rows.append({"arm": arm, "stratum": label, "n_images": len(selected),
                                 "median_delta_separability": float(np.nanmedian(delta.loc[selected]))})
    stratum_df = pd.DataFrame(stratum_rows)
    print(stratum_df.pivot(index="arm", columns="stratum", values="median_delta_separability")[
        params.REGION_LABELS].to_string(float_format=lambda x: "{:+.4f}".format(x)))
    # NaN-safe: an EMPTY stratum must not silently contribute 0 and tip the comparison. Cells with
    # no images are excluded and counted, never absorbed (5.11 in miniature).
    defined = stratum_df[stratum_df["median_delta_separability"].notna()]
    dropped = len(stratum_df) - len(defined)
    small = defined[defined["stratum"] == "<2%"]["median_delta_separability"].abs().sum()
    others = defined[defined["stratum"] != "<2%"]["median_delta_separability"].abs().sum()
    n_small = len(defined[defined["stratum"] == "<2%"])
    triggered = bool(n_small > 0 and small > others)
    print("\n  |sum of medians| in <2%: {:.4f} (over {} arm-strata)   in the other three: {:.4f}"
          .format(small, n_small, others))
    if dropped:
        print("  {} arm-stratum cells had NO images and are EXCLUDED, not counted as zero.".format(dropped))
    if n_small == 0:
        print("  <2% is empty here, so the stop condition cannot be evaluated: NOT SHOWN.")
    if triggered:
        print("  STOP CONDITION TRIGGERED. The effect concentrates where the seam/provenance")
        print("  diagnostic is weakest (12 images with an interior at k=16). The interpretation")
        print("  returns to UNRESOLVED and is DECLARED so: the sentence 'this measures how much")
        print("  forensic evidence survives' may NOT be written for this result.")
    else:
        print("  Not triggered: the effect is not concentrated in <2%, so 28's reading carries.")

    if args.limit:
        print("\nSMOKE: nothing written.")
        return

    per_image.to_csv(params.RESULTS_DIR / "codec_per_image.csv", index=False)
    summary_df.to_csv(params.RESULTS_DIR / "codec_summary.csv", index=False)
    paired_df.to_csv(params.RESULTS_DIR / "codec_paired.csv", index=False)
    ladder_df.to_csv(params.RESULTS_DIR / "codec_three_instruments.csv", index=False)
    stratum_df.to_csv(params.RESULTS_DIR / "codec_by_stratum.csv", index=False)
    seam.to_csv(params.RESULTS_DIR / "codec_seam_contribution.csv")

    conclusion = [
        "QUESTION: " + QUESTION,
        "",
        "Instrument-bound by construction: it measures what survives FOR THIS DETECTOR, not how",
        "much forensic evidence exists. NOT reformulated as 'how much does each codec alter",
        "splicing seams' -- EXPERIMENT_LOG.md 28 does not support that reading.",
        "",
        "INSTRUMENT: learned_poolB -- {}.".format(IN_DOMAIN_NOTE),
        "  Trained AND tested on CASIA 2. NOT comparable to mvss_defacto's out-of-domain 0.1663;",
        "  that gap is DOMAIN, not capability, and the contaminated checkpoint's 0.9241 proves it.",
        "  One threshold ({:.4f}), frozen on the validation REFERENCE arm, identical on all four".format(threshold),
        "  arms. No per-codec tuning. Test pass n.2, declared in 29.3.",
        "",
        "P1 (pre-registered in 29.4): best median separability retention <= {:.3f}".format(MVSS_WORST_RETENTION),
        "  observed {:.4f}  ->  {}".format(best_retention, "CONFIRMED" if confirmed else "FALSIFIED"),
        "",
        "STOP CONDITION (29.7): {}".format(
            "TRIGGERED -- interpretation UNRESOLVED for this result" if triggered else "not triggered"),
        "",
        "RESULT -- degradation is real; the ORDERING BETWEEN CODECS is not established.",
        "  Retention " + ", ".join(
            "{} {:.4f}".format(r.arm, r.median_separability_retention) for r in full_paired.itertuples())
        + " -- every 95% CI excludes zero, so compression measurably degrades the signal.",
        "  But no pipeline is shown to preserve better than another: jpeg_ai ranks FIRST for",
        "  learned_poolB and LAST for mvss_defacto (0.9010) on the same 349 images, and inside",
        "  learned_poolB two of three head-to-head comparisons are inconclusive while the third",
        "  excludes zero by 0.0001. Section 19.1's null result on Q2 SURVIVES but changes nature:",
        "  no longer 'the instrument is blind', but 'a real degradation is measured and cannot be",
        "  ordered'. The retracted 'JPEG AI preserves better' verdict (18) does NOT return.",
        "",
        "SEAM SENSITIVITY (29.6), measured rather than left as a caveat: osn stays the",
        "  worst-preserved arm in BOTH readings (0.8777 full, 0.8714 seam-excluded), so its result",
        "  is NOT a seam artefact. jpeg and jpeg_ai GAIN 1.7-2.6 points when the band is excluded;",
        "  osn gains nothing.",
    ]
    (params.RESULTS_DIR / "conclusion.txt").write_text("\n".join(conclusion) + "\n")
    (params.RESULTS_DIR / "codec_metadata.json").write_text(json.dumps({
        "question": QUESTION, "instrument": "learned_poolB", "in_domain": True,
        "in_domain_note": IN_DOMAIN_NOTE, "threshold": threshold,
        "threshold_pool": "authentic pixels, pool B validation, REFERENCE ARM ONLY, target_fpr 0.10",
        "per_codec_tuning": False, "test_pass": "n.2, declared in EXPERIMENT_LOG.md 29.3",
        "n_images": len(test_ids), "arms": params.EVAL_ARMS,
        "seam_band_px": SEAM_BAND,
        "p1_threshold": MVSS_WORST_RETENTION, "p1_observed": best_retention,
        "p1_confirmed": confirmed,
        "stop_condition_triggered": triggered,
        "excluded": {"learned_poolA": "failed pre-registered gate 6; lr_plus inflated by donor "
                                      "leakage (26.2)",
                     "multiscale": "screened out by 23.5",
                     "mvss_casia": "contaminated; ceiling only, never a conclusion (21.1)"},
    }, indent=2))
    print("\nwritten to {}".format(params.RESULTS_DIR))


if __name__ == "__main__":
    main()
