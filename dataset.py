"""CASIA2 Tp/GT pairing, dataset inventory, and image-level calibration/test split.

Pairing rule matches the confirmed reference run (exact case-insensitive
``<stem>_gt`` match), plus a serial-number fallback (the trailing 5-digit
token before the extension is the only field verified unique and reliable
across every Tp/GT pair in CASIA2 -- see plan section 2) to recover
filename-typo cases the reference run silently rejected.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _extract_serial(stem: str) -> str:
    core = stem
    for suffix in ("_gt", "_ed", "_mask"):
        if core.lower().endswith(suffix):
            core = core[: -len(suffix)]
            break
    parts = core.split("_")
    return parts[-1] if parts else ""


def list_tp_images(tp_dir: Path, splicing_only: bool = True) -> list[Path]:
    files = sorted(p for p in tp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if splicing_only:
        files = [p for p in files if p.stem.startswith("Tp_D_")]
    return files


def build_gt_index(mask_dir: Path) -> dict[str, dict[str, Path]]:
    stem_index: dict[str, Path] = {}
    serial_index: dict[str, Path] = {}
    for p in sorted(mask_dir.glob("*.png")):
        stem_index.setdefault(p.stem.lower(), p)
        serial = _extract_serial(p.stem)
        if serial:
            serial_index.setdefault(serial, p)
    return {"stem": stem_index, "serial": serial_index}


def find_mask(image_path: Path, gt_index: dict[str, dict[str, Path]]) -> tuple[Path | None, str]:
    exact_key = f"{image_path.stem.lower()}_gt"
    if exact_key in gt_index["stem"]:
        return gt_index["stem"][exact_key], "exact_<image_stem>_gt"
    serial = _extract_serial(image_path.stem)
    if serial in gt_index["serial"]:
        return gt_index["serial"][serial], "serial_number_fallback"
    return None, "missing exact <image_stem>_gt mask and no serial match"


def load_mask_and_check(mask_path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    """image_size is (width, height), PIL convention. Positive = white (>=128)."""
    with Image.open(mask_path) as im:
        mask_img = im.convert("L")
        mask_size = mask_img.size
        if mask_size != image_size:
            return None, f"source/mask size mismatch {image_size} vs {mask_size}"
        mask = np.asarray(mask_img) >= 128
    return mask, None


def build_dataset_index(
    casia_root: Path, splicing_only: bool = True, jpeg_ai_min_side: int = 161
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tp_dir = casia_root / "Tp"
    mask_dir = casia_root / "CASIA 2 Groundtruth"

    all_tp = sorted(p for p in tp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    tp_d = [p for p in all_tp if p.stem.startswith("Tp_D_")]
    tp_s = [p for p in all_tp if p.stem.startswith("Tp_S_")]
    unknown_names = [p for p in all_tp if p not in tp_d and p not in tp_s]

    all_masks = sorted(mask_dir.glob("*.png"))
    canonical_masks = [p for p in all_masks if p.stem.lower().endswith("_gt")]
    noncanonical_masks = [p for p in all_masks if p not in canonical_masks]

    gt_index = build_gt_index(mask_dir)
    candidates = tp_d if splicing_only else all_tp

    records, rejections = [], []
    for image_path in candidates:
        mask_path, match_rule = find_mask(image_path, gt_index)
        if mask_path is None:
            rejections.append(
                {"image_id": image_path.stem, "image_path": str(image_path), "mask_path": None, "reason": match_rule}
            )
            continue

        with Image.open(image_path) as im:
            image_size = im.size  # (width, height)

        mask, reject_reason = load_mask_and_check(mask_path, image_size)
        if mask is None:
            rejections.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "reason": reject_reason,
                }
            )
            continue

        width, height = image_size
        if min(width, height) < jpeg_ai_min_side:
            rejections.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "reason": f"below jpeg_ai_min_side={jpeg_ai_min_side}",
                }
            )
            continue

        records.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "mask_path": str(mask_path),
                "mask_width": width,
                "mask_height": height,
                "mask_filename": mask_path.name,
                "mask_match_rule": match_rule,
                "manipulation_type": "different_source_splicing" if splicing_only else "unspecified",
                "tampered_fraction": float(mask.mean()),
            }
        )

    records_df = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    rejections_df = pd.DataFrame(rejections)

    inventory_row = pd.DataFrame(
        [
            {
                "all_tampered_images": len(all_tp),
                "tp_d_splicing_images": len(tp_d),
                "tp_s_copy_move_excluded": len(tp_s),
                "unknown_names_excluded": len(unknown_names),
                "canonical_gt_masks": len(canonical_masks),
                "noncanonical_mask_files_ignored": len(noncanonical_masks),
                "pairing_rejections": len(rejections_df),
                "eligible_images": len(records_df),
            }
        ]
    )
    return records_df, rejections_df, inventory_row


def source_images(image_id: str) -> tuple[str, str]:
    """(host, donor) CASIA2 source images encoded in a Tp filename.

    Tp_D_CND_M_N_ani00018_sec00096_00138 -> ("ani00018", "sec00096"): the host supplies the
    background, the donor the spliced-in region. Falls back to the image_id itself for names
    that do not follow the convention, which makes such an image its own group (never merged
    with another image's group by accident).
    """
    parts = image_id.split("_")
    return (parts[5], parts[6]) if len(parts) >= 8 else (image_id, image_id)


@lru_cache(maxsize=4)
def _au_index(au_dir: Path) -> dict[str, Path]:
    """{"ani00018": Au/Au_ani_00018.jpg}. Cached: one directory scan per Au folder, not per image."""
    index: dict[str, Path] = {}
    for path in sorted(au_dir.iterdir()):
        parts = path.stem.split("_")
        if path.suffix.lower() in IMAGE_EXTS and len(parts) >= 3 and parts[0] == "Au":
            index.setdefault(parts[1] + parts[2], path)
    return index


def _background_ratio(code: str, composite: np.ndarray, mask: np.ndarray, au_dir: Path) -> float | None:
    """MAE outside the tampered mask divided by MAE inside it, or None if not comparable.

    A ratio, not an absolute: the Tp_ composite is recompressed with respect to the Au_ originals,
    so the outside-mask error is never 0 (measured median 0.0125 for the host, EXPERIMENT_LOG.md
    §19.3). Dividing by the inside-mask error normalises that away.
    """
    path = _au_index(au_dir).get(code)
    if path is None:
        return None
    with Image.open(path) as im:
        source = np.asarray(im.convert("RGB"), dtype=float)
    if source.shape != composite.shape:  # the donor is usually a differently sized image
        return None
    error = np.abs(source - composite).mean(axis=2)
    inside = float(error[mask].mean())
    return float(error[~mask].mean() / inside) if inside > 1e-6 else None


def verify_host_donor(
    image_id: str, image_path: Path, mask_path: Path, au_dir: Path, accept_ratio: float = 0.2
) -> str:
    """Does this filename follow the documented convention -- parts[5] host, parts[6] donor?

    ANALYSIS COVARIATE ONLY. It opens the ground-truth mask, so it can never be a detector
    feature; it exists to *label* the analysis in notebook section 8.2, not to predict anything.

    The host supplies the background, so its pixels outside the mask must match the composite.
    Decision on _background_ratio, host := the candidate minimising it, accepted only below
    accept_ratio. Measured over the 349 test images (EXPERIMENT_LOG.md §19.3): 329 confirmed,
    15 degenerate, 1 inverted, 4 undecidable.
    """
    host_code, donor_code = source_images(image_id)
    if host_code == donor_code:
        # 15/349 Tp_D_ ("different source") files name the same source twice. Nothing to decide
        # between two identical candidates -- reported as its own bucket, not folded into
        # "confirmed" (which would inflate the confirmation rate with empty cases).
        return "degenerate"

    with Image.open(image_path) as im:
        composite = np.asarray(im.convert("RGB"), dtype=float)
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L")) >= 128
    if mask.shape != composite.shape[:2] or not mask.any() or mask.all():
        return "undecidable"

    ratios = [_background_ratio(code, composite, mask, au_dir) for code in (host_code, donor_code)]
    best = min(range(2), key=lambda i: np.inf if ratios[i] is None else ratios[i])
    if ratios[best] is None or ratios[best] >= accept_ratio:
        return "undecidable"
    return "confirmed" if best == 0 else "inverted"


def split_calibration_test(
    records_df: pd.DataFrame, max_images: int, calib_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Image-level split, grouped so no CASIA2 host image straddles calibration and test.

    Windows from the same source image are correlated (CLAUDE.md 4), and the thresholds are
    fit specifically on AUTHENTIC pixels (calibration.pool_authentic_scores samples
    score_map[~mask]) -- which are the host's pixels. Splitting on composite image_id alone
    left 58.6% of test images sharing a source with a calibration image, so the "held-out"
    set was partly seen.

    ponytail: grouped on the host only, not on connected components of the host+donor sharing
    graph. Strict component grouping is not usable here -- one component holds 62% of the
    sample, so it would force a wildly unbalanced split. Residual donor sharing is a declared
    limitation, measured by donor_leakage_fraction() rather than hidden.

    The permutation/truncation is unchanged, so a given (seed, max_images) selects the same
    images as before; only their calibration/test assignment changes. That keeps the JPEG-AI
    encode cache (keyed on source bytes) valid.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records_df))
    truncated = records_df.iloc[order[:max_images]].reset_index(drop=True)

    n_calibration = round(calib_fraction * len(truncated))
    n_calibration = max(1, min(n_calibration, len(truncated) - 1))

    hosts = truncated["image_id"].map(lambda i: source_images(i)[0])
    shuffled_hosts = list(dict.fromkeys(hosts.iloc[rng.permutation(len(truncated))]))

    calibration_hosts: set[str] = set()
    n_assigned = 0
    for host in shuffled_hosts:
        if n_assigned >= n_calibration:
            break
        calibration_hosts.add(host)
        n_assigned += int((hosts == host).sum())

    is_calibration = hosts.isin(calibration_hosts)
    calibration_df = truncated[is_calibration].copy()
    test_df = truncated[~is_calibration].copy()
    calibration_df["split"] = "calibration"
    test_df["split"] = "test"

    assert set(calibration_df["image_id"]).isdisjoint(set(test_df["image_id"])), "calibration/test leakage"
    assert calibration_hosts.isdisjoint(
        set(test_df["image_id"].map(lambda i: source_images(i)[0]))
    ), "host image straddles the calibration/test split"
    return calibration_df, test_df


def donor_leakage_fraction(calibration_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Fraction of test images whose donor also donates to a calibration image.

    The residual leakage channel the host-level split cannot close (see split_calibration_test).
    Reported, not silently carried."""
    calibration_donors = {source_images(i)[1] for i in calibration_df["image_id"]}
    test_donors = test_df["image_id"].map(lambda i: source_images(i)[1])
    return float(test_donors.isin(calibration_donors).mean()) if len(test_df) else 0.0
