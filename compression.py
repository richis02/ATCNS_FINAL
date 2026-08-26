"""JPEG / OSN-like / official JPEG-AI compression pipelines.

JPEG and OSN are plain Pillow round-trips. JPEG-AI shells out to the real
ITU-T T.840 / ISO/IEC 6048 reference codec running in its own conda env
(jpeg_ai_vm) -- that subprocess is free to depend on torch/CUDA internally,
since it is never imported into this process. No proxy is used: if the
codec/runner is missing, validate_official_jpeg_ai() fails loudly.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class CompressionResult:
    image: np.ndarray
    byte_size: int
    pipeline: str


def _pil_from_array(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")


def _encode_jpeg(im: Image.Image, quality: int, subsampling: int) -> io.BytesIO:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=True)
    return buf


def _encode_jpeg_at_bpp(im: Image.Image, source_pixels: int, target_bpp: float, subsampling: int) -> tuple[io.BytesIO, int]:
    """Pick the JPEG quality whose rate is closest to target_bpp for THIS image.

    A fixed quality is a fixed quantiser scale, not a fixed rate: at quality 13 the same
    setting produced 0.42 bpp on the test set against JPEG AI's 0.52, a 22% rate gap in the
    learned codec's favour, because JPEG AI does its own per-image rate control and JPEG does
    not. Matching the knob is not matching the rate, and the whole comparison rests on the
    rate being matched -- so bisect quality per image instead. ~7 Pillow encodes per image.
    """
    low, high, best = 1, 95, None
    while low <= high:
        quality = (low + high) // 2
        buf = _encode_jpeg(im, quality, subsampling)
        bpp = (buf.tell() * 8) / source_pixels
        if best is None or abs(bpp - target_bpp) < best[2]:
            best = (buf, quality, abs(bpp - target_bpp))
        if bpp < target_bpp:
            low = quality + 1
        else:
            high = quality - 1
    return best[0], best[1]


def identity_compress(image_rgb: np.ndarray, source_path: Path | str | None = None) -> CompressionResult:
    """The reference arm: no ADDITIONAL compression. Pixels pass through untouched.

    Not "the uncompressed image": most CASIA2 Tp_D_ files ship as JPEG already, so this arm is
    the file *as distributed*, which is what an analyst would actually receive before any of the
    three pipelines is applied. byte_size is that file's size on disk, so bpp is comparable in
    units -- but it is not rate-controlled and must not be read as a matched operating point.
    Without a source_path (synthetic arrays, tests) byte_size is 0 and compression_quality
    reports bpp 0.0; the notebook records NaN for the reference arm in that case.
    """
    byte_size = Path(source_path).stat().st_size if source_path is not None else 0
    return CompressionResult(image=np.asarray(image_rgb), byte_size=byte_size, pipeline="original")


def jpeg_compress(
    image_rgb: np.ndarray, quality: int = 80, subsampling: int = 2, target_bpp: float | None = None
) -> CompressionResult:
    im = _pil_from_array(image_rgb)
    source_pixels = im.width * im.height
    buf = (
        _encode_jpeg(im, quality, subsampling)
        if target_bpp is None
        else _encode_jpeg_at_bpp(im, source_pixels, target_bpp, subsampling)[0]
    )
    byte_size = buf.tell()
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"))
    return CompressionResult(image=out, byte_size=byte_size, pipeline="jpeg")


def osn_compress(
    image_rgb: np.ndarray, max_dim: int = 1280, quality: int = 65, subsampling: int = 2,
    target_bpp: float | None = None,
) -> CompressionResult:
    """Downsample -> JPEG -> upsample back to the source resolution.

    The restore step is what an analyst actually works with (you get the platform's image
    back and inspect it against the original geometry), and it is what makes this pipeline
    comparable to jpeg/jpeg_ai at all: without it osn is scored on a different pixel grid,
    with a NEAREST-resampled mask (prevalence 0.1385 vs 0.1491) and a different effective
    window scale, so its metrics answer a different question. byte_size stays the size of
    the *transmitted* bitstream -- the upsample is a decoder-side operation and adds no bits.
    """
    im = _pil_from_array(image_rgb)
    source_size = im.size
    source_pixels = im.width * im.height
    if max(im.size) > max_dim:
        scale = max_dim / max(im.size)
        new_size = (round(im.width * scale), round(im.height * scale))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
    # target_bpp is charged against the SOURCE pixel count even though the encode happens on
    # the downsampled grid -- that is the rate the pipeline actually spends per source pixel.
    buf = (
        _encode_jpeg(im, quality, subsampling)
        if target_bpp is None
        else _encode_jpeg_at_bpp(im, source_pixels, target_bpp, subsampling)[0]
    )
    byte_size = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    if decoded.size != source_size:
        decoded = decoded.resize(source_size, Image.Resampling.LANCZOS)
    return CompressionResult(image=np.asarray(decoded), byte_size=byte_size, pipeline="osn")


def validate_official_jpeg_ai(jpeg_ai_root: Path, runner: list[str], expected_revision: str | None = None) -> str:
    if not jpeg_ai_root.is_dir():
        raise RuntimeError(f"JPEG AI reference software not found at {jpeg_ai_root}. No proxy will be used.")
    runner_bin = Path(runner[0])
    if not runner_bin.is_file():
        raise RuntimeError(f"JPEG AI runner interpreter not found at {runner_bin}.")
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=jpeg_ai_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not read JPEG AI repo revision: {result.stderr}")
    revision = result.stdout.strip()
    if expected_revision is not None and revision != expected_revision:
        raise RuntimeError(f"JPEG AI revision mismatch: found {revision}, expected {expected_revision}.")
    return revision


def _image_cache_key(image_bytes: bytes, target_bpp: float, profile: str, tools: str) -> str:
    payload = image_bytes + f"|{target_bpp}|{profile}|{tools}".encode()
    return hashlib.sha256(payload).hexdigest()


def jpeg_ai_compress(
    image_rgb: np.ndarray,
    cache_dir: Path,
    jpeg_ai_root: Path,
    runner: list[str],
    target_bpp: float = 0.5,
    profile: str = "high",
    tools: str = "on",
    timeout_s: int = 1800,
) -> CompressionResult:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_im = _pil_from_array(image_rgb)
    src_bytes_buf = io.BytesIO()
    src_im.save(src_bytes_buf, format="PNG")
    key = _image_cache_key(src_bytes_buf.getvalue(), target_bpp, profile, tools)

    out_png = (cache_dir / f"{key}.png").resolve()
    out_bin = (cache_dir / f"{key}.bin").resolve()
    out_meta = (cache_dir / f"{key}.json").resolve()

    if out_png.exists() and out_meta.exists():
        meta = json.loads(out_meta.read_text())
        out = np.asarray(Image.open(out_png).convert("RGB"))
        return CompressionResult(image=out, byte_size=meta["byte_size"], pipeline="jpeg_ai")

    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        src_path = tmp_dir / "input.png"
        src_im.save(src_path)

        # Positional args (input_path, bin_path); -r/--rec_path makes the encoder emit the
        # reconstructed PNG directly, no separate decoder call needed (verified end to end).
        # --set_target_bpp takes an int, target_bpp*100 (e.g. 0.5 bpp -> 50).
        encode_cmd = runner + [
            "-m", "src.reco.coders.encoder",
            str(src_path), str(out_bin),
            "--cfg", f"cfg/profiles/{profile}.json", f"cfg/tools_{tools}.json",
            "--set_target_bpp", str(int(round(target_bpp * 100))),
            "-r", str(out_png),
        ]
        subprocess.run(encode_cmd, cwd=jpeg_ai_root, check=True, capture_output=True, timeout=timeout_s)

    byte_size = out_bin.stat().st_size
    out_meta.write_text(json.dumps({"byte_size": byte_size, "target_bpp": target_bpp}))
    out = np.asarray(Image.open(out_png).convert("RGB"))
    return CompressionResult(image=out, byte_size=byte_size, pipeline="jpeg_ai")


def apply_compression(image_rgb: np.ndarray, pipeline: str, **kwargs) -> CompressionResult:
    dispatch = {
        "original": identity_compress,
        "jpeg": jpeg_compress,
        "osn": osn_compress,
        "jpeg_ai": jpeg_ai_compress,
    }
    if pipeline not in dispatch:
        raise ValueError(f"unknown pipeline: {pipeline}")
    return dispatch[pipeline](image_rgb, **kwargs)


def compression_quality(original_rgb: np.ndarray, result: CompressionResult) -> dict:
    original = np.asarray(original_rgb, dtype=np.uint8)
    compressed = np.asarray(result.image, dtype=np.uint8)
    # bpp is per SOURCE pixel, always. Dividing by the delivered pixel count instead makes a
    # pipeline that downsamples look like it spends the same rate as one that does not: osn at
    # max_dim=256 reported 0.408 bpp on its own 256px grid while actually spending 0.158 bpp of
    # the source image -- a 3x rate gap masquerading as a matched operating point.
    source_pixels = original.shape[0] * original.shape[1]
    if original.shape != compressed.shape:
        ref_im = Image.fromarray(original).resize((compressed.shape[1], compressed.shape[0]), Image.Resampling.LANCZOS)
        original = np.asarray(ref_im, dtype=np.uint8)
    # The reference arm is bit-identical to its own input, so MSE is 0 and PSNR is +inf by
    # definition. Guarded rather than left to skimage's divide-by-zero warning; plots and
    # summaries drop the reference arm from the PSNR axis instead of charting an infinity.
    identical = np.array_equal(original, compressed)
    psnr = np.inf if identical else peak_signal_noise_ratio(original, compressed, data_range=255)
    ssim = 1.0 if identical else structural_similarity(original, compressed, channel_axis=2, data_range=255)
    bpp = (result.byte_size * 8) / source_pixels
    return {"psnr": psnr, "ssim": ssim, "bpp": bpp, "byte_size": result.byte_size}


def one_image_integrity_check(sample_image_rgb: np.ndarray, jpeg_ai_kwargs: dict) -> dict:
    """Must succeed before the full experiment starts. Raises AssertionError on failure."""
    result = jpeg_ai_compress(sample_image_rgb, **jpeg_ai_kwargs)
    quality = compression_quality(sample_image_rgb, result)
    assert result.byte_size > 0, "JPEG AI produced an empty bitstream"
    assert not np.array_equal(result.image, sample_image_rgb), "JPEG AI output is identical to input (proxy/no-op?)"
    assert quality["psnr"] > 10, f"JPEG AI PSNR implausibly low: {quality['psnr']}"
    return quality
