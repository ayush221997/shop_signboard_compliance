"""
BCP-47 language hints for Google Cloud Vision `ImageContext.languageHints`.

Hints steer the first-pass script models toward regional scripts; pair with
DOCUMENT_TEXT_DETECTION. Max ~10 per request — we cap at 8 and dedupe.
"""

from __future__ import annotations

import os
import re
from typing import Optional

# Regional primary + common Latin / Hindi English shop text.
# "en" is almost always needed for phone numbers, brands, and Latin.
_STATE_VISION_LANG_HINTS: dict[str, list[str]] = {
    "KA": ["kn", "en"],
    "TN": ["ta", "en"],
    "TG": ["te", "en"],
    "AP": ["te", "en"],
    "MH": ["mr", "en", "hi"],
    "GJ": ["gu", "en", "hi"],
    "WB": ["bn", "en"],
    "OR": ["or", "en"],
    "AS": ["as", "en", "bn"],
    "PB": ["pa", "en", "hi"],
    "DL": ["hi", "en", "ur", "pa"],
    "HP": ["hi", "en"],
    "KL": ["ml", "en"],
    "GA": ["en", "hi", "mr"],
    "UT": ["hi", "en"],
    "UP": ["hi", "en"],
    "MP": ["hi", "en", "mr"],
    "BR": ["hi", "en"],
    "HR": ["hi", "en"],
    "RJ": ["hi", "en"],
    "SK": ["ne", "en", "hi"],
    "AR": ["hi", "en"],
}

_DEFAULT = ["en", "hi", "es"]
_MAX = 8


def list_vision_language_hints(state_code: Optional[str]) -> list[str]:
    s = (state_code or "").strip().upper() or None
    if s and s.startswith("IN-"):
        s = s[3:]
    out: list[str] = list(_STATE_VISION_LANG_HINTS.get(s) or _DEFAULT)
    extra = (os.getenv("VISION_LANGUAGE_HINTS_EXTRA") or "").strip()
    if extra:
        for part in re.split(r"[,\s]+", extra, flags=re.IGNORECASE):
            p = part.strip().lower()
            if 2 <= len(p) <= 3 and p.isalpha() and p not in out:
                out.append(p)
    # Dedupe, preserve order, 2–3 letter BCP-47 base tags only
    seen: set[str] = set()
    final: list[str] = []
    for x in out:
        t = re.match(r"^([a-z]{2,3})", (x or "").lower())
        if not t:
            continue
        c = t.group(1)
        if c in seen or c in ("und",):
            continue
        seen.add(c)
        final.append(c)
        if len(final) >= _MAX:
            break
    return final
