"""
Heuristic evaluation of public shop signboard policy against OCR + threshed geometry.
Materials, fines, police NOC, façade %, and exact font pt sizes are not computable from pixels alone
— those are listed under not_verifiable.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from compliance_rules import (
    evaluate_compliance,
    get_rules_for_state,
    normalize_lang,
)
from governance_data import GOVERNANCE_RULE_TABLE
from indic_script_infer import text_is_mostly_kannada_script

# Pydantic TextBlock in main; use duck typing here
def _base_lang(b: Any) -> str:
    if not b.languages:
        return "und"
    return normalize_lang(b.languages[0].code or "und")


def _block_lang_best_ocr(b: Any) -> str:
    """Highest-confidence language hint on a block (matches Vision block ordering in main)."""
    if not b.languages:
        return "und"
    best = max(b.languages, key=lambda h: float(h.confidence or 0.0))
    return normalize_lang(best.code or "und")


def _block_dict_for_sort(b: Any) -> tuple[float, str]:
    n = b.norm
    y = n.y_pct if n is not None else 999.0
    t = b.text or ""
    return (y, t)


def _sorted_text_blocks(blocks: list[Any]) -> list[Any]:
    """Top-to-bottom, non-empty first."""
    xs = [b for b in blocks if (b.text or "").strip()]
    return sorted(xs, key=lambda b: b.norm.y_pct if b.norm is not None else 1e9)


def _sorted_blocks_reading_order(blocks: list[Any]) -> list[Any]:
    """Approximate reading order: top-to-bottom, then left-to-right."""
    xs = [b for b in blocks if (b.text or "").strip() and b.norm is not None]
    if not xs:
        return _sorted_text_blocks(blocks)
    return sorted(xs, key=lambda b: (float(b.norm.y_pct), float(b.norm.x_pct)))


def _check_sequence_first(blocks: list[Any], want: str) -> tuple[bool, str]:
    """Check that the first block in reading order is tagged with `want`."""
    wantn = normalize_lang(want)
    xs = _sorted_blocks_reading_order(blocks)
    if not xs:
        return False, f"No text regions for sequence check (want {wantn})."
    first = xs[0]
    got = normalize_lang(_base_lang(first))
    if got == wantn:
        return True, f"First text block in reading order is tagged {wantn}."
    return False, f"First text block is {got} (expected {wantn} first)."


def _ink_ratios(geo: Optional[dict[str, Any]]) -> dict[str, float]:
    r = (geo or {}).get("language_ratios") or {}
    if not isinstance(r, dict):
        return {}
    by: dict[str, float] = {}
    for k, v in r.items():
        b = normalize_lang(str(k))
        if b:
            by[b] = by.get(b, 0.0) + float(v)
    return by


def _y_center(b: Any) -> Optional[float]:
    if b.norm is None:
        return None
    return float(b.norm.y_pct) + float(b.norm.h_pct or 0) / 2.0


def _word_best_lang_vision(word: Any) -> str:
    prop = getattr(word, "property", None)
    if not prop or not list(prop.detected_languages):
        return "und"
    best = max(
        list(prop.detected_languages), key=lambda l: float(l.confidence or 0.0)
    )
    return normalize_lang(getattr(best, "language_code", None) or "und")


def ka_ocr_word_upper_layout_hint(vision_response: Any, img_w: int, img_h: int) -> Optional[dict[str, Any]]:
    """
    Word-level bounding boxes: if Vision/scripts identify Kannada, use centroid Y in 0–100
    image space. Returns a definitive hint for KA upper-half, or None to use block/ink path.
    """
    if not vision_response or not getattr(vision_response, "full_text_annotation", None):
        return None
    pages = list(vision_response.full_text_annotation.pages)
    if not pages or img_w <= 0 or img_h <= 0:
        return None
    kn_upper = 0
    kn_lower = 0
    for page in pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    wt = "".join(s.text for s in word.symbols)
                    if not (wt or "").strip():
                        continue
                    wlang = _word_best_lang_vision(word)
                    is_kn = wlang == "kn" or text_is_mostly_kannada_script(wt)
                    if not is_kn:
                        continue
                    bb = word.bounding_box
                    if not bb or not list(bb.vertices):
                        continue
                    ys = [v.y for v in bb.vertices]
                    y0, y1 = min(ys), max(ys)
                    yc = (y0 + (y1 - y0) * 0.5) * 100.0 / float(img_h)
                    if yc < 50.0:
                        kn_upper += 1
                    else:
                        kn_lower += 1
    if kn_upper > 0:
        return {
            "ok": True,
            "definitive": True,
            "detail": (
                "At least one Kannada word (OCR or script) has its vertical centroid in the "
                "upper 50% of the sign (Y<50%), from word-level bounding boxes."
            ),
        }
    if kn_lower > 0 and kn_upper == 0:
        return {
            "ok": False,
            "definitive": True,
            "detail": (
                "Kannada was identified in word-level boxes, but all centroids lie in the lower 50% "
                "of the sign (Y≥50%)."
            ),
        }
    return None


def _check_tamil_top(blocks: list[Any]) -> tuple[bool, str]:
    tbs = _sorted_text_blocks(blocks)
    if not tbs:
        return False, "No text regions for top-of-board check."
    top = tbs[0]
    if normalize_lang(_base_lang(top)) == "ta":
        return True, "Topmost text block is tagged Tamil."
    return False, "Topmost text block is not tagged as Tamil; verify layout."


def _check_lang_top(blocks: list[Any], want: str) -> tuple[bool, str]:
    wantn = normalize_lang(want)
    tbs = _sorted_text_blocks(blocks)
    if not tbs:
        return False, f"No text regions for {wantn}-at-top check."
    top = tbs[0]
    if normalize_lang(_base_lang(top)) == wantn:
        return True, f"Topmost block is tagged {wantn}."
    return (
        False,
        f"Topmost block is {_base_lang(top)} (expected {wantn} at top).",
    )


def _check_ink_geq_others(geo: dict[str, Any] | None, want: str) -> tuple[bool, str]:
    by = _ink_ratios(geo)
    w = normalize_lang(want)
    if w not in by and by:
        return (
            False,
            f"Threshed ink for '{w}' not found among detected scripts.",
        )
    if not by:
        return False, "No threshed language ink ratios."
    mv = 0.0
    for k, v in by.items():
        if k != w:
            mv = max(mv, v)
    if by.get(w, 0) + 1e-6 >= mv:
        return (
            True,
            f"Threshed ink for '{w}' (≈{by.get(w, 0) * 100:.1f}%) is at least as large as other lines.",
        )
    return (
        False,
        f"Threshed ink for '{w}' (≈{by.get(w, 0) * 100:.1f}%) is less than some other script.",
    )


def _check_ka_upper_half_kn(
    blocks: list[Any],
    geo: Optional[dict[str, Any]],
    layout_hints: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    """
    Kannada must sit in the upper half of the sign. Uses the vertical **centroid** of each
    text block in normalised (0–100) image space: y_centre = y_pct + h_pct/2.

    - If ``layout_hints`` contains a definitive word-level OCR result (Kannada boxes from
      Vision in image space), that wins over block/ink heuristics.
    - Prefer blocks where OCR tags Kannada (first hint or best-confidence hint).
    - If no Kannada-tagged line exists but threshed ink still shows Kannada share, use the
      topmost text line by bounding box (OCR layout only).
    """
    wk = (layout_hints or {}).get("ka_ocr_word_upper")
    if isinstance(wk, dict) and wk.get("ok") is True and (wk.get("detail") or "").strip():
        return True, str(wk["detail"])
    if (
        isinstance(wk, dict)
        and wk.get("ok") is False
        and wk.get("definitive")
        and (wk.get("detail") or "").strip()
    ):
        return False, str(wk["detail"])

    tbs = [b for b in blocks if (b.text or "").strip() and _y_center(b) is not None]
    if not tbs:
        return False, "No text regions with positions for upper-half check."

    def _is_kn(b: Any) -> bool:
        tx = (getattr(b, "text", None) or "").strip()
        return (
            normalize_lang(_base_lang(b)) == "kn"
            or _block_lang_best_ocr(b) == "kn"
            or text_is_mostly_kannada_script(tx)
        )

    kblocks = [b for b in tbs if _is_kn(b)]
    if kblocks:
        if any(_y_center(b) is not None and _y_center(b) < 50.0 for b in kblocks):
            return (
                True,
                "At least one Kannada-tagged line (by OCR) has its vertical centroid in the upper 50% of the sign (Y<50%).",
            )
        return (
            False,
            "Kannada-tagged lines (OCR) have centroids in the lower half; may not meet the upper-half policy.",
        )

    # Ink shows Kannada but per-block language failed: use top line by layout (centroid in upper 50%)
    kn_ink = _ink_ratios(geo).get("kn", 0.0) if geo else 0.0
    if kn_ink < 0.03:
        return (
            False,
            "No Kannada OCR label on any line and threshed Kannada ink is very low; cannot place the regional line in the upper half.",
        )
    tbs.sort(key=lambda b: b.norm.y_pct if b.norm is not None else 1e9)
    top = tbs[0]
    yc = _y_center(top)
    if yc is not None and yc < 50.0:
        return (
            True,
            "Kannada was not labeled per line, but threshed ink suggests Kannada; the top text line (by bounding-box centroid) lies in the upper 50% of the sign (Y<50%).",
        )
    return (
        False,
        "No Kannada OCR label; threshed ink suggests Kannada but the top line centroid is not in the upper 50% (Y≥50%).",
    )


def _check_tn_ratio(geo: dict[str, Any] | None) -> tuple[bool, str]:
    r = (geo or {}).get("language_ratios") or {}
    if not isinstance(r, dict) or not r:
        return False, "No threshed ratios for 5:3:2 check."
    by: dict[str, float] = {}
    for k, v in r.items():
        b = normalize_lang(str(k))
        if b:
            by[b] = by.get(b, 0) + float(v)
    s = sum(by.values()) or 1.0
    ta = (by.get("ta", 0) / s) if s else 0.0
    en = (by.get("en", 0) / s) if s else 0.0
    other = 1.0 - ta - en
    if other < 0:
        other = 0.0
    # Target 0.5 / 0.3 / 0.2; relaxed tolerance
    t = 0.12
    ok_5 = abs(ta - 0.5) <= t
    ok_3 = abs(en - 0.3) <= t
    ok_2 = abs(other - 0.2) <= 0.15
    if ok_5 and ok_3 and ok_2:
        return (
            True,
            f"Threshed ta:en:other ≈ {ta*100:.0f}% : {en*100:.0f}% : {other*100:.0f}% (target 5:3:2, relaxed).",
        )
    return (
        False,
        f"Threshed share ta:en:other ≈ {ta*100:.0f}% : {en*100:.0f}% : {other*100:.0f}%; not close to 5:3:2 (illustrative check).",
    )


def _as_en_present(geo: dict[str, Any] | None) -> tuple[bool, str]:
    by = _ink_ratios(geo)
    if "as" in by and "en" in by and by.get("as", 0) > 0.05 and by.get("en", 0) > 0.05:
        return True, "Assamese and English both have measurable ink in threshed regions."
    return (
        False,
        "Assamese-English 'alongside' pair not both prominent in threshed ink (OCR/geometry only).",
    )


def _te_strong(geo: dict[str, Any] | None) -> tuple[bool, str]:
    by = _ink_ratios(geo)
    t = by.get("te", 0)
    if t >= 0.2:
        return (
            True,
            f"Telugu threshed ink ≈{t*100:.1f}% (visibility proxy).",
        )
    if t > 0:
        return (False, f"Telugu ink ≈{t*100:.1f}% (policy may need stronger prominence).")
    return False, "No Telugu threshed ink found."


def _gu_dominant(geo: dict[str, Any] | None) -> tuple[bool, str]:
    by = _ink_ratios(geo)
    if not by or "gu" not in by:
        return False, "Gujarati (gu) not detected in threshed script mix."
    g = float(by["gu"])
    m = max((float(v) for k, v in by.items() if k != "gu"), default=0.0)
    if g + 1e-6 >= m:
        return True, f"Gujarati ink share (≈{g * 100:.1f}%) is at least as large as the next script."
    return False, f"Gujarati ink (≈{g * 100:.1f}%) is not the dominant share vs other visible scripts."


def evaluate_governance(
    state_code: str,
    geo: dict[str, Any] | None,
    blocks: list[Any],
    layout_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    sc = (state_code or "").strip().upper()
    if sc.startswith("IN-"):
        sc = sc[3:]
    if sc not in GOVERNANCE_RULE_TABLE:
        return None
    spec = copy.deepcopy(GOVERNANCE_RULE_TABLE[sc])
    out: dict[str, Any] = {
        "state_code": sc,
        "jurisdiction": spec.get("jurisdiction", sc),
        "citations": spec.get("citations", []),
        "automated": [],
        "not_verifiable": [],
        "ink_rule_eval": None,
    }

    # Threshed-area check when STATE_COMPLIANCE_RULES has min % for this state
    # Use `geo is not None` (not `if geo`): an empty {} is still valid input; `not {}` would skip
    # ink_rule_eval and break ink_ratio / governance merge with base compliance.
    cr = get_rules_for_state(sc)
    if cr and geo is not None:
        ev = evaluate_compliance(geo, cr)
        out["ink_rule_eval"] = {
            "status": ev.get("status"),
            "message": ev.get("message"),
            "details": ev.get("details"),
        }

    nvc = out["not_verifiable"]
    nvc.append(
        {
            "area": "materials_and_structure",
            "text": spec.get("material_summary")
            or spec.get("max_signage_sqm_note")
            or "Ban/flex/illumination/road projection rules require physical inspection.",
        }
    )
    nvc.append(
        {
            "area": "fines_enforcement",
            "text": spec.get("enforcement_summary")
            or "Penalties and license actions are not determined from the image alone.",
        }
    )

    tags: list[str] = list(spec.get("automated_checks") or [])
    for tag in tags:
        ok, detail = (False, "unsupported")
        if tag == "ink_ratio" and cr:
            e = out.get("ink_rule_eval") or {}
            s = (e.get("status") or "").lower()
            out["automated"].append(
                {
                    "id": "threshed_mandatory_script_share",
                    "ok": s in ("pass", "uncertain"),
                    "detail": e.get("message", ""),
                }
            )
        elif tag == "kannada_upper_half" and sc == "KA":
            ok, detail = _check_ka_upper_half_kn(blocks, geo, layout_hints)
            out["automated"].append(
                {"id": "kannada_upper_half", "ok": ok, "detail": detail}
            )
        elif tag == "tamil_top_block" and sc == "TN":
            ok, detail = _check_tamil_top(blocks)
            out["automated"].append(
                {"id": "tamil_top_block", "ok": ok, "detail": detail}
            )
        elif tag == "tn_ratio_5_3_2" and sc == "TN":
            ok, detail = _check_tn_ratio(geo)
            out["automated"].append(
                {"id": "tn_5_3_2_ink", "ok": ok, "detail": detail}
            )
        elif tag == "bengali_top_block" and sc == "WB":
            ok, detail = _check_lang_top(blocks, "bn")
            out["automated"].append(
                {"id": "bengali_top", "ok": ok, "detail": detail}
            )
        elif tag == "bengali_ink_geq_others" and sc == "WB":
            ok, detail = _check_ink_geq_others(geo, "bn")
            out["automated"].append(
                {"id": "bengali_ink", "ok": ok, "detail": detail}
            )
        elif tag == "punjabi_top_block" and sc == "PB":
            ok, detail = _check_lang_top(blocks, "pa")
            out["automated"].append(
                {"id": "punjabi_top", "ok": ok, "detail": detail}
            )
        elif tag == "punjabi_ink_geq_others" and sc == "PB":
            ok, detail = _check_ink_geq_others(geo, "pa")
            out["automated"].append(
                {"id": "punjabi_ink", "ok": ok, "detail": detail}
            )
        elif tag == "odia_top_or_prominent" and sc == "OR":
            t1, d1 = _check_lang_top(blocks, "or")
            t2, d2 = _check_ink_geq_others(geo, "or")
            ok = t1 or t2
            out["automated"].append(
                {
                    "id": "odia_prominence",
                    "ok": ok,
                    "detail": f"Top: {d1} | Ink: {d2}",
                }
            )
        elif tag == "te_visibility_combined" and sc in ("TG", "AP"):
            t1, d1 = _check_lang_top(blocks, "te")
            t2, d2 = _te_strong(geo)
            # Prominence = Telugu topmost line OR strong threshed share (OCR+geometry proxy).
            ok = t1 or t2
            out["automated"].append(
                {
                    "id": "telugu_visibility",
                    "ok": ok,
                    "detail": f"Top: {d1} | {d2}",
                }
            )
        elif tag == "marathi_top_block" and sc == "MH":
            ok, detail = _check_lang_top(blocks, "mr")
            out["automated"].append(
                {"id": "marathi_top", "ok": ok, "detail": detail}
            )
        elif tag == "marathi_ink_geq_others" and sc == "MH":
            ok, detail = _check_ink_geq_others(geo, "mr")
            out["automated"].append(
                {"id": "marathi_ink", "ok": ok, "detail": detail}
            )
        elif tag == "hindi_first_sequence" and sc == "DL":
            ok, detail = _check_sequence_first(blocks, "hi")
            out["automated"].append(
                {"id": "hindi_first_sequence", "ok": ok, "detail": detail}
            )
        elif tag == "gujarati_ink_dominant" and sc == "GJ":
            ok, detail = _gu_dominant(geo)
            out["automated"].append(
                {"id": "gujarati_dominant", "ok": ok, "detail": detail}
            )
        elif tag == "assamese_plus_english_present" and sc == "AS":
            ok, detail = _as_en_present(geo)
            out["automated"].append(
                {"id": "as_en_ink", "ok": ok, "detail": detail}
            )
        # DL / HP: no auto tags, already added not_verifiable

    # De-duplicate NVC (merge policy lines once)
    seen: set[str] = set()
    nv2: list[dict[str, str]] = []
    for n in nvc:
        t = f"{n.get('area', '')}::{(n.get('text') or '')[:80]}"
        if t not in seen:
            seen.add(t)
            nv2.append(n)
    out["not_verifiable"] = nv2

    # Overall: fail if any automated check exists and one clearly fails; ink fail counts
    fails = 0
    oks = 0
    for a in out["automated"]:
        if a.get("ok") is True:
            oks += 1
        else:
            fails += 1
    iev = out.get("ink_rule_eval")
    if iev and (iev.get("status") or "").lower() == "fail":
        fails += 1

    if not out["automated"] and not (iev and (iev.get("status") or "").lower() in ("pass", "fail", "uncertain")):
        out["overall"] = "uncertain"
    elif fails == 0 and out["automated"]:
        out["overall"] = "pass"
    elif fails and oks:
        out["overall"] = "partial"
    elif fails and not oks and out["automated"]:
        out["overall"] = "fail"
    else:
        ou = (iev or {}).get("status") or "uncertain"
        out["overall"] = (ou or "uncertain").lower() if isinstance(ou, str) else "uncertain"

    out["headline"] = _headline_for(out)
    return out


def _headline_for(gv: dict[str, Any]) -> str:
    ov = gv.get("overall", "uncertain")
    j = gv.get("jurisdiction", "")
    nfailed = len([a for a in gv.get("automated", []) if a.get("ok") is False])
    return f"Signboard policy ({j}): automated checks → {ov}. Failed sub-checks: {nfailed}."


def _merge_compliance_status(base_s: str, g_overall: str) -> str:
    """Pick the stricter of base threshed-rule status and governance overall."""
    a = (base_s or "pending").lower()
    b = (g_overall or "uncertain").lower()
    rank = {"fail": 0, "partial": 1, "uncertain": 2, "pending": 3, "pass": 4}
    return min((rank.get(a, 2), a), (rank.get(b, 2), b), key=lambda x: x[0])[1]


def merge_governance_compliance(
    base_message: str,
    base_status: str,
    base_ruleset: Optional[str],
    state_code: Optional[str],
    geo: Any,
    blocks: list[Any],
    layout_hints: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    """Return (message, status, governance dict). `base_ruleset` reserved for future use."""
    _ = base_ruleset
    if not state_code:
        return base_message, base_status, None
    geo_d = None
    if geo is not None:
        if hasattr(geo, "model_dump"):
            geo_d = geo.model_dump()
        elif isinstance(geo, dict):
            geo_d = geo
    if not isinstance(geo_d, dict):
        geo_d = None
    gv = evaluate_governance(state_code, geo_d, blocks, layout_hints)
    if not gv:
        return base_message, base_status, None
    message = f"{base_message} · {gv.get('headline', '')}".strip(" ·")
    st = _merge_compliance_status(base_status, str(gv.get("overall", "uncertain")))
    return message, st, gv
