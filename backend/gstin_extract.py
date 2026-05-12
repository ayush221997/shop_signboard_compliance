"""
Extract Indian GSTINs from OCR text, validate format + check digit, map to text blocks.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# 2 (state) + 10 (PAN) + 1 (entity) + Z + 1 (check) — 14th is Z for normal registrants
_GSTIN_RE = re.compile(
    r"(?<![0-9A-Za-z])(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z])(?![0-9A-Za-z])",
    re.IGNORECASE,
)

_CODE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MOD = 36


def _check_digit_for_prefix(first_14: str) -> str:
    """15th character from first 14 (GSTN mod-36, right-to-left weighting 2,1,2,1,…)."""
    s = 0
    factor = 2
    for i in range(13, -1, -1):
        ch = first_14[i].upper()
        cp = _CODE36.find(ch)
        if cp < 0:
            return "?"
        d = factor * cp
        factor = 1 if factor == 2 else 2
        d = (d // _MOD) + (d % _MOD)
        s += d
    check = (_MOD - (s % _MOD)) % _MOD
    return _CODE36[check]


def is_valid_gstin(value: str) -> bool:
    v = re.sub(r"[\s\-\.]", "", value or "").upper()
    if len(v) != 15:
        return False
    if not re.fullmatch(
        r"\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]", v, re.IGNORECASE
    ):
        return False
    return v[14] == _check_digit_for_prefix(v[:14])


def _normalize_ocr(s: str) -> str:
    return re.sub(r"[\s\u200b\-\.:/]", "", s or "")


def find_gstins(full_text: str) -> list[str]:
    """Deduplicate matches on raw and whitespace-stripped text."""
    out: list[str] = []
    seen: set[str] = set()
    for blob in (full_text, _normalize_ocr(full_text)):
        for m in _GSTIN_RE.finditer(blob):
            raw = m.group(1).upper()
            if len(raw) == 15 and raw not in seen:
                seen.add(raw)
                out.append(raw)
    return out


def _block_contains(block_text: str, gstin: str) -> bool:
    n = _normalize_ocr(block_text)
    g = re.sub(r"[\s\-\.]", "", gstin)
    u = block_text.upper()
    return g in n.upper() or gstin in u or g in n


def enrich_gstins(full_text: str, blocks: list[Any]) -> list[dict[str, Any]]:
    found = find_gstins(full_text)
    result: list[dict[str, Any]] = []
    for g in found:
        idx: Optional[int] = None
        for i, b in enumerate(blocks):
            t = getattr(b, "text", str(b) if b is not None else "") or ""
            if t and _block_contains(t, g):
                idx = i
                break
        result.append(
            {
                "value": g,
                "format_valid": bool(
                    re.fullmatch(
                        r"\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]Z[0-9A-Z]", g, re.IGNORECASE
                    )
                ),
                "checksum_valid": is_valid_gstin(g),
                "block_index": idx,
            }
        )
    return result
