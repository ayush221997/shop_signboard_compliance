"""
Cloud Vision only accepts a fixed set of raster encodings (JPEG, PNG, GIF, WebP, BMP, TIFF, …).
It does *not* accept AVIF, HEIC/HEIF, or JPEG XL — the API returns
'Unsupported type: image/avif. Use PNG, JPEG, ...'. We decode those in-process and
re-encode as PNG before calling annotate_image.
"""

from __future__ import annotations

import logging
from io import BytesIO

from fastapi import HTTPException

_log = logging.getLogger(__name__)

# HEIC/HEIF/AVIF openers (registers with Pillow)
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception as exc:  # optional dependency issues
    _log.debug("pillow_heif: %s", exc)

# Extra AVIF codecs (registers with Pillow when present)
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


def _ftyp_suggests_modern_container(b: bytes) -> bool:
    if len(b) < 20 or b[4:8] != b"ftyp":
        return False
    s = b[8:20].lower()
    return any(
        mark in s
        for mark in (b"avif", b"avis", b"heic", b"heif", b"heix", b"mif1", b"msf1")
    )


def _declared_is_modern_image(ct: str) -> bool:
    t = (ct or "").lower()
    return any(
        x in t
        for x in (
            "avif",
            "heic",
            "heif",
            "jxl",
            "jxs",
            "hsj2",
            "heif-",
            "heic-",
        )
    )


def should_transcode_to_png_for_vision(
    data: bytes, declared_mime: str, filename: str | None
) -> bool:
    """If True, decode to PNG; PDF is never transcoded here."""
    dm = (declared_mime or "").split(";", 1)[0].strip().lower()
    if dm == "application/pdf" or (dm.startswith("image/") and "pdf" in dm):
        return False
    if _declared_is_modern_image(dm):
        return True
    name = (filename or "").lower()
    if any(name.endswith(suf) for suf in (".avif", ".heic", ".heif", ".hif", ".jxl", ".jxs")):
        return True
    if _ftyp_suggests_modern_container(data):
        return True
    return False


def transcode_to_png(data: bytes) -> bytes:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    from PIL import Image, ImageFile  # type: ignore

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(BytesIO(data))
    except Exception as e:
        raise HTTPException(
            status_code=415,
            detail=(
                "This image type is not supported by the OCR service yet. "
                f"Could not decode: {e!s}. Try exporting to PNG, JPEG, or WebP."
            ),
        ) from e

    # White background for transparency; Vision is happiest with non-palette RGB
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        layer = im.convert("RGBA")
        bg.paste(layer, mask=layer.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    out = BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def boost_indic_contrast(image, light_aware: bool = True):
    """
    Boost HSV for typical signboard foregrounds (green / red / yellow) on blue or dark
    backgrounds to improve low-contrast OCR (e.g. green flex on blue).

    When ``light_aware`` is True, already-bright images skip the strong V boost to
    avoid clipping highlights (sun / glare) that harm OCR for Indic text.

    Expects a BGR ``uint8`` array (as from ``cv2.imdecode``). Returns a BGR ``uint8``
    array of the same shape. On failure, returns the input unchanged.
    """
    try:
        import numpy as np
        import cv2
    except Exception:  # pragma: no cover
        return image
    if image is None or getattr(image, "size", 0) == 0 or image.dtype != np.uint8:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    bgr = image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    gvm = float(np.mean(v_ch)) if v_ch.size else 128.0
    if light_aware and gvm >= 200.0:
        return bgr
    # OpenCV H ∈ [0, 179], S and V ∈ [0, 255] — only vivid (“bright sign”) colors
    lo_s, lo_v = 50, 55
    m_red1 = cv2.inRange(
        hsv, np.array([0, lo_s, lo_v], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)
    )
    m_red2 = cv2.inRange(
        hsv, np.array([170, lo_s, lo_v], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)
    )
    m_red = cv2.bitwise_or(m_red1, m_red2)
    m_yellow = cv2.inRange(
        hsv, np.array([15, lo_s, lo_v], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)
    )
    m_green = cv2.inRange(
        hsv, np.array([40, lo_s, lo_v], dtype=np.uint8), np.array([100, 255, 255], dtype=np.uint8)
    )
    mask = cv2.bitwise_or(m_red, m_yellow)
    mask = cv2.bitwise_or(mask, m_green)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    m1 = (mask > 0)
    if not np.any(m1):
        return bgr
    hsv2 = hsv.copy()
    h, s, v = cv2.split(hsv2)
    v = v.astype(np.float32)
    s = s.astype(np.float32)
    if light_aware and gvm >= 175.0:
        # Sunlit / mid-bright: gentler V/S gain to avoid driving ink into 255
        v_scale, v_add, s_scale, s_add = 1.1, 14.0, 1.05, 6.0
        blend = 0.4
    else:
        v_scale, v_add, s_scale, s_add = 1.32, 35.0, 1.1, 10.0
        blend = 0.65
    v[m1] = np.clip(v[m1] * v_scale + v_add, 0, 255)
    s[m1] = np.clip(s[m1] * s_scale + s_add, 0, 255)
    hsv2 = cv2.merge([h, s.astype(np.uint8), v.astype(np.uint8)])
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    m3 = m1.astype(np.float32)[:, :, np.newaxis]
    a = blend
    out = bgr.astype(np.float32) * (1.0 - m3 * a) + bgr2.astype(np.float32) * (m3 * a)
    return np.clip(out, 0, 255).astype(np.uint8)
