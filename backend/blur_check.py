"""
Reject overly blurry rasters before OCR (Laplacian variance on grayscale).
PDF uploads are not checked (render path differs from OpenCV).
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _blur_check_disabled() -> bool:
    v = (os.environ.get("OCR_DISABLE_BLUR_CHECK") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def laplacian_sharpness_score_bgr(
    bgr: np.ndarray, max_long_edge: int = 800
) -> float:
    """
    Higher = sharper (more edge energy). ``bgr`` is BGR uint8, any size.
    Normalizes by downscaling the long edge to at most ``max_long_edge`` (area-only).
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return 0.0
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = int(gray.shape[0]), int(gray.shape[1])
    m = max(h, w)
    if m > int(max_long_edge) > 0:
        s = float(max_long_edge) / float(m)
        gray = cv2.resize(
            gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def min_laplacian_for_accept() -> float:
    return float(os.environ.get("OCR_BLUR_LAPLACIAN_VAR_MIN", "100"))


def is_too_blurry_bgr(
    bgr: np.ndarray, min_score: float | None = None
) -> tuple[bool, float, float]:
    if _blur_check_disabled():
        return False, 0.0, 0.0
    m = min_laplacian_for_accept() if min_score is None else float(min_score)
    s = laplacian_sharpness_score_bgr(bgr)
    return s < m, s, m
