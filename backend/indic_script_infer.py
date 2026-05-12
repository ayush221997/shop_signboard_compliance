"""
When Google Vision returns no language (und) for a line, but glyphs are clearly
in a Brahmic or Latin range, map that line to a BCP-47 base code for threshed ink and rules.
"""

from __future__ import annotations

# Unicode block → BCP-47. Devanagari uses internal key "deva" (hi vs mr not inferable from chars).
_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "deva"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
]

_MIN_FRACTION = 0.25
# Allow single-glyph lines (e.g. one big Kannada word) when that glyph is unambiguous
_MIN_GLYPHS = 1


def _count_by_script(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in text:
        if c.isspace():
            continue
        o = ord(c)
        for lo, hi, code in _RANGES:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
        else:
            if o <= 0x7F and c.isalpha():
                counts["en"] = counts.get("en", 0) + 1
    return counts


def infer_indic_dominant_bcp47(text: str) -> str | None:
    """
    If the line is clearly dominated by one script, return a BCP-47 base (kn, te, en, ...).
    Returns None for devanagari (ambiguous) or if unclear.
    """
    t = (text or "").strip()
    if not t:
        return None
    cts = _count_by_script(t)
    if not cts:
        return None
    if "deva" in cts and len([k for k in cts if k != "deva"]) == 0:
        return None  # only devanagari — cannot choose hi/mr
    # Ignore deva when other scripts are present: mixed deva+en shop name
    c2 = {k: v for k, v in cts.items() if k != "deva"}
    if not c2:
        return None
    total = sum(c2.values())
    if total < _MIN_GLYPHS:
        return None
    best_v = max(c2.values())
    winners = [k for k, v in c2.items() if v == best_v]
    if len(winners) > 1:
        return None
    best_k = winners[0]
    if best_v / total < _MIN_FRACTION:
        return None
    out = {
        "kn": "kn", "te": "te", "ta": "ta", "ml": "ml", "or": "or", "bn": "bn", "pa": "pa", "gu": "gu", "en": "en"
    }
    return out.get(best_k)


def text_is_mostly_kannada_script(text: str) -> bool:
    """Kannada script clearly dominates the letter content (same as infer… == 'kn' when unambiguous)."""
    return infer_indic_dominant_bcp47(text) == "kn"
