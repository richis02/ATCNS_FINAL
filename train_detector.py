"""PHASE 2: train the learned detector on TRAIN, select on VALIDATION, freeze. No test, no codec.

GOVERNING RULE (verbatim; binding on every Phase 2 module):

    The detector is selected ONLY on its ability to localize tampering in the uncompressed
    reference arm, BEFORE any codec is involved. It is then frozen completely -- weights,
    preprocessing, thresholds, postprocessing, feature extraction, scale fusion,
    hyperparameters. Only then may we ask which codec best preserves what the frozen detector
    can detect. The detector is never modified, reselected, or retuned to favour any codec.

Two runs, identical in every respect but data volume:
  pool A -- 920 train / 162 validation, host-disjoint from the frozen test set. PRIMARY.
  pool B -- 379 / 74, host AND donor-disjoint, NESTED in A's assignment. LEAKAGE ABLATION.
Their difference at the same test set, in Phase 3, is the donor-leakage measurement.

AUGMENTATION RULE, non-negotiable (23.8): every photometric augmentation is applied to the WHOLE
image, never differentially inside vs outside the mask. A per-region jitter would manufacture
exactly the synthetic splice signal this detector is supposed to find. No JPEG, blur or noise
augmentation at all -- those manufacture compression artifact families and would contaminate the
Phase 4 question before it is asked.

IN-DOMAIN: trained and tested on CASIA 2. Labelled so in every artefact this writes.

Run in .venv-deep:
    .venv-deep/bin/python train_detector.py --pool A [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path[:0] = ["src", "configs"]

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import params
import phase2_common as common
import detector_learning as learning
from forensics import dataset

CONFIG = params.LEARNING


class ReferenceCrops(Dataset):
    """Random crops of the reference arm. Loads the CASIA2 file; no codec is ever applied."""

    def __init__(self, image_ids, records, crop: int, seed: int, augment: bool = True):
        self.image_ids = list(image_ids)
        self.records = records
        self.crop = crop
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        rgb, mask = common.load_reference(image_id, self.records)
        # Per-worker, per-epoch-independent RNG. Deterministic in the seed, so a rerun of the same
        # (seed, pool) trains on the same stream.
        rng = np.random.default_rng((self.seed, index, torch.initial_seed() % (2**31)))

        # Reflect-pad IMAGE AND MASK IDENTICALLY when a side is short. Padding one and zero-filling
        # the other would label reflected tampered pixels authentic -- manufactured label noise.
        height, width = mask.shape
        pad_h, pad_w = max(0, self.crop - height), max(0, self.crop - width)
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            height, width = mask.shape

        if self.augment:
            top = int(rng.integers(0, height - self.crop + 1))
            left = int(rng.integers(0, width - self.crop + 1))
        else:
            top, left = (height - self.crop) // 2, (width - self.crop) // 2
        rgb = rgb[top:top + self.crop, left:left + self.crop]
        mask = mask[top:top + self.crop, left:left + self.crop]

        if self.augment:
            # Geometric: exact, lossless, applied to image and mask together.
            if rng.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                rgb, mask = rgb[::-1], mask[::-1]
            turns = int(rng.integers(0, 4))
            if turns:
                rgb, mask = np.rot90(rgb, turns), np.rot90(mask, turns)
            # Photometric: GLOBAL. One scalar brightness and one scalar contrast for the whole
            # crop, applied to every pixel regardless of the mask (23.8).
            brightness = 1.0 + float(rng.uniform(-CONFIG["jitter_brightness"], CONFIG["jitter_brightness"]))
            contrast = 1.0 + float(rng.uniform(-CONFIG["jitter_contrast"], CONFIG["jitter_contrast"]))
            adjusted = np.asarray(rgb, dtype=np.float32)
            adjusted = (adjusted - adjusted.mean()) * contrast + adjusted.mean() * brightness
            rgb = np.clip(adjusted, 0, 255).astype(np.uint8)

        rgb = np.ascontiguousarray(rgb)
        mask = np.ascontiguousarray(mask)
        x = learning.to_tensor(rgb)[0]
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y


def fit_threshold_and_score(model, val_ids, records, device, target_fpr=common.TARGET_FPR,
                            seed=params.RNG_SEED):
    """Threshold at the pre-registered FPR on VALIDATION authentic pixels, then score.

    One forward pass per image, reused for both: the threshold is a quantile of the authentic
    probabilities, and the prediction is that same map thresholded.
    """
    rng = np.random.default_rng(seed)
    cap = common.PIXELS_CAP
    probabilities, masks, pooled = {}, {}, []
    for image_id in val_ids:
        rgb, mask = common.load_reference(image_id, records)
        probability = 1.0 / (1.0 + np.exp(-learning.predict_logits(model, rgb, device)))
        probabilities[image_id], masks[image_id] = probability, mask
        authentic = probability[~mask]
        if len(authentic) > cap:
            authentic = authentic[rng.choice(len(authentic), size=cap, replace=False)]
        pooled.append(authentic)
    threshold = float(np.quantile(np.concatenate(pooled), 1 - target_fpr))
    per_image = common.score_images(
        lambda i: (probabilities[i] >= threshold, masks[i]), val_ids)
    return threshold, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="A", choices=["A", "B"])
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 40 images, nothing written")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="host-level nested prefix of the pool's train split, for a "
                             "data-volume sweep (e.g. 0.25/0.5/0.75/1.0). The host-prefix itself "
                             "is always chosen with params.RNG_SEED, independent of --seed, so "
                             "repeat runs at the same fraction train on the SAME images.")
    parser.add_argument("--seed", type=int, default=params.RNG_SEED,
                        help="model init / augmentation-stream seed only. Does not affect which "
                             "images the --train-fraction prefix selects.")
    args = parser.parse_args()

    print(__doc__.split("Run in .venv-deep:")[0].strip())
    print("=" * 94)
    common.assert_declared_baselines()
    records = common.manifest()
    train_ids = common.ids_for("train", args.pool)
    val_ids = common.ids_for("validation", args.pool)
    train_ids = common.host_prefix_ids(train_ids, args.train_fraction)
    common.assert_host_disjoint(train_ids, val_ids)
    train_hosts = {dataset.source_images(i)[0] for i in train_ids}
    train_donors = {dataset.source_images(i)[1] for i in train_ids}
    max_epochs, patience = CONFIG["max_epochs"], CONFIG["patience"]
    if args.smoke:
        train_ids, val_ids = train_ids[:40], val_ids[:20]
        max_epochs, patience = 2, 2
        print("SMOKE MODE: {} train / {} val, {} epochs. Numbers are NOT reported.".format(
            len(train_ids), len(val_ids), max_epochs))

    baseline_f1 = common.trivial_baseline_f1(val_ids)
    prevalence = float(records.loc[train_ids, "mask_area_px"].sum()
                       / (records.loc[train_ids, "image_width"] * records.loc[train_ids, "image_height"]).sum())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pool {}: {} train / {} validation | device {}".format(
        args.pool, len(train_ids), len(val_ids), device))
    if args.train_fraction < 1.0 or args.seed != params.RNG_SEED:
        print("train-fraction sweep: fraction={:.2f}  images={}  hosts={}  donors={}  "
              "init_seed={}".format(args.train_fraction, len(train_ids), len(train_hosts),
                                     len(train_donors), args.seed))
    print("mean tampered-pixel fraction on TRAIN = {:.5f} -- the class imbalance the combined "
          "focal({}) + dice loss handles, weights {} / {}, DECLARED and not tuned (23.8)".format(
              prevalence, CONFIG["focal_gamma"], CONFIG["focal_weight"], CONFIG["dice_weight"]))
    print("validation all-positive F1 = {:.4f} (this split's own baseline; the test set's 0.2580 "
          "is a different number and is not used here)".format(baseline_f1))
    print("IN-DOMAIN: trained and tested on CASIA 2. Label every number accordingly.")

    torch.manual_seed(args.seed)
    model = learning.UNetResNet34(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader(
        ReferenceCrops(train_ids, records, CONFIG["crop"], params.RNG_SEED),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, drop_last=True,
        persistent_workers=True,
    )

    history, per_image_by_epoch = [], {}
    best = {"f1": -1.0, "epoch": -1, "state": None, "threshold": None}
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = learning.combined_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()

        model.eval()
        threshold, per_image = fit_threshold_and_score(model, val_ids, records, device)
        summary = common.summarize(per_image)
        ok, reason = common.passes_selection_gates(per_image)
        # Kept for the optimism gap: the EPOCH axis is the grid this detector selects over, so the
        # host-halved procedure of 23.5 needs each epoch's per-image scores. 162 rows x <=120
        # epochs, so keeping them costs nothing and avoids a second pass over the validation set.
        per_image_by_epoch["epoch_{}".format(epoch)] = per_image
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)), "threshold": threshold,
            "val_f1": summary["f1"], "val_mcc": summary["mcc"], "val_recall": summary["recall"],
            "val_precision": summary["precision"], "val_silent": summary["n_silent_images"],
            "val_positive_ratio": summary["positive_ratio"],
            "selection_gates_pass": ok, "selection_gates_reason": reason,
        })
        marker = ""
        # Early stopping ON THE PRE-REGISTERED OBJECTIVE (23.1), not on the loss: the loss is not
        # the thing being selected on, and stopping on it would freeze a different criterion.
        if summary["f1"] > best["f1"]:
            best = {"f1": summary["f1"], "epoch": epoch, "threshold": threshold,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            marker = "  <- best"
        print("  epoch {:3d}  loss {:.4f}  val micro F1 {:.4f}  MCC {:.4f}  silent {:3d}  "
              "gates {}{}".format(epoch, np.mean(losses), summary["f1"], summary["mcc"],
                                  summary["n_silent_images"], "ok" if ok else "REJECT", marker))
        if epoch - best["epoch"] >= patience:
            print("  early stop: {} epochs without improving the objective".format(patience))
            break

    elapsed = time.time() - started
    print("\ntrained {:.1f} min, best epoch {} at micro F1 {:.4f}".format(
        elapsed / 60, best["epoch"], best["f1"]))

    model.load_state_dict(best["state"])
    model.eval()
    threshold, per_image = fit_threshold_and_score(model, val_ids, records, device)
    summary = common.summarize(per_image)
    ok, reason = common.passes_selection_gates(per_image)
    by_stratum = common.summarize_by_stratum(per_image)

    # The grid this detector selects over is the EPOCH axis, so the gap is measured over the epochs
    # with exactly the procedure 23.5 declares for every other detector: validation halved BY HOST,
    # argmax on one half, re-scored on the other. Not a surrogate -- an earlier draft used
    # "best minus median epoch", which is a different statistic wearing the same label.
    epochs_df = pd.DataFrame(history)
    gap = common.optimism_gap(per_image_by_epoch, val_ids)
    gap["note"] = ("epochs are the candidates; host-halved validation, argmax on one half, "
                   "re-scored on the other ({} epochs considered)".format(len(epochs_df)))
    corrected, screened = common.screening(summary["f1"], gap["optimism_gap_f1"], baseline_f1)

    print("\n" + "=" * 94)
    print("FROZEN -- pool {}, validation reference arm, {} images   [IN-DOMAIN: CASIA 2]".format(
        args.pool, len(val_ids)))
    print("=" * 94)
    for key in ("f1", "mcc", "precision", "recall", "iou", "lr_plus", "fpr", "positive_ratio",
                "n_silent_images", "silent_rate", "macro_mcc_silent_as_zero"):
        print("  {:<26} {}".format(key, round(summary[key], 4)))
    print("  {:<26} {:.4f}".format("all_positive_f1", baseline_f1))
    print("  {:<26} {}".format("beats_all_positive_f1", summary["f1"] > baseline_f1))
    print("  {:<26} {} ({})".format("selection_gates", "PASS" if ok else "REJECT", reason))
    print("  {:<26} {:+.4f}  ({} epochs, host-halved: {} won the inner fit, {:.4f} -> {:.4f})".format(
        "optimism_gap_f1", gap["optimism_gap_f1"], len(epochs_df), gap["inner_selected"],
        gap["inner_fit_f1"], gap["inner_check_f1"]))
    print("  {:<26} {:.4f} -> screening {}".format(
        "corrected_f1", corrected, "PASS" if screened else "FAIL (no test pass)"))
    print("\nper stratum:")
    print(by_stratum[["stratum", "n_images", "recall", "precision", "f1", "lr_plus",
                      "n_silent_images"]].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    if args.smoke:
        print("\nSMOKE MODE: nothing written.")
        return
    torch.save({"state_dict": best["state"], "pool": args.pool, "epoch": best["epoch"],
                "threshold": threshold, "config": CONFIG,
                "train_fraction": args.train_fraction, "train_seed": args.seed},
               params.RESULTS_DIR / "learned_pool{}.pt".format(args.pool))
    epochs_df.to_csv(params.RESULTS_DIR / "learned_training_log.csv", index=False)
    summary_row = dict(summary)
    summary_row.update({
        "detector": "learned_unet_resnet34", "pool": args.pool, "split": "validation",
        "in_domain": True, "domain_note": "IN-DOMAIN (trained on CASIA 2)",
        "threshold": threshold, "epoch": best["epoch"],
        "all_positive_f1": baseline_f1, "beats_all_positive_f1": bool(summary["f1"] > baseline_f1),
        "selection_gates_pass": ok, "selection_gates_reason": reason,
        "optimism_gap_f1": gap["optimism_gap_f1"], "corrected_f1": corrected,
        "pre_test_screening_pass": screened,
        "train_mean_tampered_fraction": prevalence,
        "test_images_touched": 0,
    })
    pd.DataFrame([summary_row]).to_csv(
        params.RESULTS_DIR / "learned_validation_summary.csv", index=False)
    by_stratum.insert(0, "in_domain", True)
    by_stratum.to_csv(params.RESULTS_DIR / "learned_by_stratum.csv", index=False)
    (params.RESULTS_DIR / "learned_frozen.json").write_text(json.dumps({
        "detector": "U-Net, torchvision resnet34 IMAGENET1K_V1 encoder, off the shelf",
        "in_domain": True, "domain_note": "IN-DOMAIN (trained on CASIA 2); never comparable to "
                                          "mvss_defacto's out-of-domain 0.1663",
        "channel_order": "RGB (trained through PIL/numpy, unlike the cv2/BGR MVSS arm)",
        "inference": "native resolution, reflect-pad to /{}, crop back; no resize, mask never "
                     "resampled".format(CONFIG["pad_multiple"]),
        "pool": args.pool, "epoch": best["epoch"], "threshold": threshold,
        "target_fpr": common.TARGET_FPR, "config": CONFIG,
        "train_mean_tampered_fraction": prevalence,
        "all_positive_f1_validation": baseline_f1,
        "optimism_gap": gap, "corrected_f1": corrected, "pre_test_screening_pass": screened,
        "test_images_touched": 0,
        "train_fraction": args.train_fraction, "train_seed": args.seed,
        "n_train_images": len(train_ids), "n_train_hosts": len(train_hosts),
        "n_train_donors": len(train_donors), "training_minutes": elapsed / 60,
    }, indent=2))
    print("\nwritten to {}".format(params.RESULTS_DIR))
    print("test images touched: 0")


if __name__ == "__main__":
    main()
