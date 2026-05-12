"""
OpenCV pre-processing for Vision DOCUMENT_TEXT_DETECTION.

* Light-aware path: overexposed / high-key photos (sun, glare) get CLAHE on
  luminance before the signboard HSV "pop" — without changing pixel dimensions
  (geometry for Vision bboxes is unchanged).
* Optional adaptive binarization for low-contrast dark ink on light boards; skipped
  when the frame is too bright (washed) to avoid erasing Indic/OCR strokes.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def bgr_ensure_min_long_edge(bgr: np.ndarray, min_edge: int = 1024) -> np.ndarray:
    """
    If the long edge of ``bgr`` is below ``min_edge`` px, upsample with cubic
    interpolation so Vision and geometry see enough resolution for Indic and
    Latin script strokes. Does not downscale large photos.
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return bgr
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    m = int(min_edge) if int(min_edge) > 0 else 1024
    long_ = max(h, w)
    if long_ >= m:
        return bgr
    s = m / float(long_)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _env_light_aware() -> bool:
    v = (os.getenv("OCR_DISABLE_LIGHT_AWARE") or "").strip().lower()
    return v not in ("1", "true", "yes", "on")


def bgr_hsv_v_mean(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return 128.0
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def bgr_is_high_key(bgr: np.ndarray, v_mean_min: float = 188.0) -> bool:
    """Overexposed / backlit: mean V in OpenCV HSV; sunlit boards are often very bright."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return False
    return bgr_hsv_v_mean(bgr) >= v_mean_min


def bgr_strong_sky_glare(bgr: np.ndarray, v_floor: int = 248, frac_min: float = 0.06) -> bool:
    """Many near-white pixels — sky reflection or specular; needs tone recovery."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return False
    hsv = cv2.cvtColor(
        np.ascontiguousarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV
    )
    v = hsv[:, :, 2]
    n = v.size
    if n == 0:
        return False
    return (float(np.sum(v >= v_floor)) / float(n)) >= frac_min


def retone_high_key_bgr(
    bgr: np.ndarray,
    clip: float = 2.2,
    tile: int = 8,
) -> np.ndarray:
    """
    Local contrast on L* only (LAB) — recovers text in sunlit or washed flex without
    re-encoding a wrong global white balance. Same shape/dtype.
    """
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    if bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b2 = cv2.split(lab)
    tw = int(tile) if int(tile) > 0 else 8
    clahe = cv2.createCLAHE(
        clipLimit=float(clip), tileGridSize=(tw, tw)
    )
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b2]), cv2.COLOR_LAB2BGR)


def prepare_bgr_for_document_ocr(
    bgr: np.ndarray,
) -> np.ndarray:
    """
    Pre-Vision chain: (optional) high-key retone, then HSV signboard boost.

    ``boost_indic_contrast`` is imported from `image_prep` to avoid a circular
    import at `image_prep` load time.
    """
    if not _env_light_aware() or bgr is None or bgr.size == 0:
        from image_prep import boost_indic_contrast  # local import

        return boost_indic_contrast(bgr, light_aware=False)

    from image_prep import boost_indic_contrast  # local import

    out = np.ascontiguousarray(bgr, dtype=np.uint8)
    if bgr_is_high_key(out) or bgr_strong_sky_glare(out):
        out = retone_high_key_bgr(out)
    return boost_indic_contrast(out, light_aware=True)


def should_prebinarize_for_vision_ocr() -> bool:
    v = (os.getenv("VISION_OCR_PREBINARIZE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def prebinarize_bgr_for_vision_ocr(
    bgr: np.ndarray,
    block_size: int = 31,
    c: int = 7,
) -> np.ndarray:
    """
    Grayscale + adaptive threshold, then 3-channel BGR for image/png encode and Vision.
    Tuned for dark ink on light backgrounds; does not change geometry (same W×H).
    """
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    block_size = int(block_size) | 1
    if block_size < 3:
        block_size = 3
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(c),
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def prebinarize_bgr_smart(
    bgr: np.ndarray,
    block_size: int = 31,
    c: int = 7,
) -> np.ndarray:
    """
    Like :func:`prebinarize_bgr_for_vision_ocr`, but for washed / high-mean luma
    (sun, glare) returns CLAHE retone in color instead of a harsh binary that can
    erode Indic stroke edges.
    """
    if not _env_light_aware():
        return prebinarize_bgr_for_vision_ocr(
            bgr, block_size=block_size, c=c
        )
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    if bgr.size == 0 or bgr.ndim != 3:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray)) / 255.0
    if m > 0.72:
        return retone_high_key_bgr(bgr, clip=1.8, tile=8)
    return prebinarize_bgr_for_vision_ocr(bgr, block_size=block_size, c=c)


def horizontal_projection_profile(gray: np.ndarray) -> np.ndarray:
    """
    Sum of ink-like pixels (dark) per row — useful for headline / script heuristics.
    Expects 8-bit grayscale. Values are 0..255*width.
    """
    g = np.asarray(gray, dtype=np.uint8)
    inv = 255 - g
    return inv.sum(axis=1).astype(np.float64)
