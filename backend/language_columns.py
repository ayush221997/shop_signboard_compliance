"""
Map signboard geometry (threshed ink) into flat DB columns for SQL queries and compliance reporting.
"""

from __future__ import annotations

from typing import Any, Optional

from compliance_rules import normalize_lang


def threshed_language_audit_columns(
    geo: Any,
    location_state_code: Optional[str],
) -> dict[str, Any]:
    """
    Return kwargs for `AuditRecord` (or any model with the same field names):
    - location_state_code: 2-letter state/UT when set
    - threshed_dominant_lang: agent dominant
    - threshed_l{1,2,3}_code + _ratio: top three by ink share (ratio 0–1, same as API)
    """
    st = (location_state_code or "").strip().upper() or None
    if st and st.startswith("IN-"):
        st = st[3:]

    nulled = {
        "location_state_code": st,
        "threshed_dominant_lang": None,
        "threshed_l1_code": None,
        "threshed_l1_ratio": None,
        "threshed_l2_code": None,
        "threshed_l2_ratio": None,
        "threshed_l3_code": None,
        "threshed_l3_ratio": None,
    }
    if geo is None:
        return nulled

    d = geo.model_dump() if hasattr(geo, "model_dump") else (geo if isinstance(geo, dict) else {})
    if not isinstance(d, dict):
        nulled["location_state_code"] = st
        return nulled

    nulled["location_state_code"] = st
    dom = (d.get("dominant_language") or "").strip()
    nulled["threshed_dominant_lang"] = normalize_lang(dom) if dom else None

    ratios: dict[str, float] = d.get("language_ratios") or {}
    if not isinstance(ratios, dict) or not ratios:
        return nulled

    by: list[tuple[str, float]] = []
    for raw_k, v in ratios.items():
        b = normalize_lang(str(raw_k))
        if b:
            by.append((b, float(v)))
    by.sort(key=lambda x: -x[1])
    for i, (code, r) in enumerate(by[:3], start=1):
        nulled[f"threshed_l{i}_code"] = code
        nulled[f"threshed_l{i}_ratio"] = r

    return nulled
