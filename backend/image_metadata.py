"""
File and embedded image metadata (dimensions, format, common EXIF) for OCR results.
All values are JSON-serialisable for `raw_ocr_json` and API responses.
"""

from __future__ import annotations

import contextlib
from io import BytesIO
from typing import Any, Optional

# Register HEIC/AVIF and similar so Pillow can open the same rasters as `image_prep` / Vision.
try:
    import image_prep  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# ── Excluded / large EXIF tags (noise or binary) ────────────────────────────
_SKIP_EXIF_NAMES = frozenset(
    {
        "UserComment",
        "MakerNote",
        "Interoperability",
        "ComponentsConfiguration",
    }
)
_MAX_EXIF = 32  # number of tags to keep (after filtering)
_MAX_STR = 200


def _json_safe_exif_value(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if len(t) > _MAX_STR:
            t = t[:_MAX_STR] + "…"
        return t
    if isinstance(v, (list, tuple)):
        out = []
        for x in v[:12]:
            out.append(_json_safe_exif_value(x))
        if len(v) > 12:
            out.append(f"…+{len(v) - 12} more")
        return out
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(v)} bytes>"
    with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
        if hasattr(v, "numerator") and hasattr(v, "denominator") and v.denominator:
            return round(float(v), 5)
    return str(v)[:_MAX_STR]


def extract_pillow_metadata(
    data: bytes,
) -> dict[str, Any]:
    """
    Return format, size, mode, and a trimmed EXIF name→value map.
    Fails soft: on error, returns { "error": "..." } only.
    """
    out: dict[str, Any] = {}
    try:
        from PIL import ExifTags, Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(BytesIO(data)) as im:
            out["format"] = (im.format or "").upper() or None
            w, h = im.size
            out["width_px"] = w
            out["height_px"] = h
            out["color_mode"] = im.mode
            exif: dict[str, Any] = {}
            try:
                e = im.getexif() if im else None
                if e:
                    n = 0
                    for tag_id, raw_val in e.items():
                        if n >= _MAX_EXIF:
                            break
                        name = str(ExifTags.TAGS.get(tag_id, tag_id))
                        if str(name) in _SKIP_EXIF_NAMES:
                            continue
                        v = _json_safe_exif_value(raw_val)
                        exif[str(name)] = v
                        n += 1
            except Exception:
                exif = {}
            if exif:
                out["exif"] = exif
    except Exception as e:  # noqa: BLE001
        out = {"error": f"Could not read image with Pillow: {e!s}"}
    return out


def build_image_metadata(
    *,
    data: bytes,
    filename: Optional[str],
    declared_mime: str,
    is_pdf: bool,
    vision_width: int = 0,
    vision_height: int = 0,
) -> dict[str, Any]:
    """
    Aggregate file properties + optional EXIF and Vision page dimensions.
    """
    name = (filename or "").strip() or "upload"
    mime = (declared_mime or "").split(";", 1)[0].strip().lower() or "application/octet-stream"
    out: dict[str, Any] = {
        "file": {
            "name": name,
            "size_bytes": len(data),
            "content_type": mime,
        },
    }
    if is_pdf:
        out["pdf"] = {
            "is_pdf": True,
        }
        out["raster"] = {
            "note": "PDF: pixel dimensions and EXIF are not extracted here.",
        }
    else:
        pmeta = extract_pillow_metadata(data)
        out["raster"] = pmeta
    if vision_width and vision_height:
        out["vision"] = {
            "width": int(vision_width),
            "height": int(vision_height),
        }
    return out
