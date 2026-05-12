"""
COMPLai — universal per-state threshed-ink language rules and evaluation.

Rules are keyed by 2-letter state/UT code (e.g. ``KA``, ``MH``, ``DL``). The engine does
**not** depend on Google Vision language hints; OCR script detection is model-driven.

``verified_data`` is a plain dict shaped like :class:`signboard_agent.SignboardGeometryAgent` output, e.g.:
  { "language_ratios": {"kn": 0.62, "en": 0.38}, "primary_ratio": 0.62, ... }
``location_rules`` is a normalized row from :class:`UniversalComplianceEngine` (primary, min %, flags).
"""

from __future__ import annotations

import copy
import re
from typing import Any, ClassVar, Optional

# Aliases: alternate codes clients may send → canonical key in STATE_RULES
_STATE_CODE_ALIASES: dict[str, str] = {
    "IN-KA": "KA",
    "IN-MH": "MH",
    "TS": "TG",  # Telangana (older code)
    "UK": "UT",  # Uttarakhand (older vehicle code)
}


def _r(
    name: str,
    primary_iso: str,
    min_primary_area_pct: float,
    font_ratio_required: bool = False,
    min_text_share_pct: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "primary_iso_code": primary_iso,
        "min_primary_area_pct": min_primary_area_pct,
        "font_ratio_required": font_ratio_required,
        "min_text_share_pct": min_text_share_pct,
    }


# Extra BCP-47 base scripts often seen on signboards in addition to ``primary`` + English.
# E.g. Maharashtra: Marathi (primary) + Hindi; Punjab: Punjabi + Hindi; J&K: Urdu + Hindi.
_STATE_SCRIPT_EXTRAS: dict[str, tuple[str, ...]] = {
    "CH": ("hi",),  # Punjabi + Hindi/English
    "GA": ("hi", "mr"),  # Goa: common alongside Konkani; do not add kn (wrong-jurisdiction sign)
    "GJ": ("hi",),
    "JK": ("hi",),
    "MH": ("hi",),
    "PB": ("hi",),
}

_SCRIPT_LABEL: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "lus": "Mizo",
    "mni": "Meitei",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "hmn": "Hmong",  # Vision sometimes misfires on Indic; still surfaced for allowlist rules
}


def get_required_scripts_for_state_code(state_code: str) -> list[str]:
    """
    Returned list is the jurisdiction **allowlist** for threshed ink:

    - the **dominant** script must be in this set; and
    - **every** script with ink ≥ :data:`JURISDICTION_MIN_INK_FRACTION` (see
      :func:`evaluate_compliance`) must be in this set — so 0% primary minimum
    (e.g. Goa) cannot “pass” on English alone if another script (e.g. Kannada) has
    material ink in the mix.
    """
    c = UniversalComplianceEngine.normalize_state_code(state_code)
    if not c or c not in UniversalComplianceEngine.STATE_RULES:
        return ["en"]
    row = UniversalComplianceEngine.STATE_RULES[c]
    p = normalize_lang(str(row["primary_iso_code"]))
    allow: set[str] = set()
    if p:
        allow.add(p)
    allow.add("en")
    for x in _STATE_SCRIPT_EXTRAS.get(c, ()):
        allow.add(normalize_lang(x))
    return sorted(allow)


def _label_for_script(code: str) -> str:
    c = normalize_lang(code)
    return _SCRIPT_LABEL.get(c, c.upper() if c else "unknown")


def _jurisdiction_script_mismatch_message(
    state_code: str,
    detected: str,
    location_name: str,
    required_sorted: list[str],
) -> str:
    c = UniversalComplianceEngine.normalize_state_code(state_code)
    d = normalize_lang(detected)
    if c == "DL" and d == "kn":
        return (
            "Mismatch Detected: Kannada script is not recognized for Delhi municipal "
            "compliance. Hindi/English required."
        )
    det_l = _label_for_script(d)
    req_l = ", ".join(_label_for_script(x) for x in required_sorted) if required_sorted else "n/a"
    return (
        f"Mismatch Detected: {det_l} script is not recognized for {location_name} "
        f"municipal compliance. Allowlist: {req_l}."
    )


# Threshed-ink share (0–1) above which a script is checked against the jurisdiction allowlist.
# Without this, states with 0% primary floor (e.g. Goa) still “pass” when the dominant
# script (often English) is allowed but another script (e.g. Kannada) has material ink.
JURISDICTION_MIN_INK_FRACTION: float = 0.015

# Some scripts are common misfires from OCR language tagging (e.g. Indic/LATN confusion).
# Treat them as "material" only at a higher floor so they don't dominate compliance outcomes.
_JURISDICTION_NOISE_FLOOR: dict[str, float] = {
    "hmn": 0.10,  # Hmong: often spurious on Indian signboards; require 10% to treat as meaningful
}


def _jurisdiction_unexpected_ink_in_mix_message(
    location_name: str,
    disallowed: list[tuple[str, float]],
    req_sorted: list[str],
) -> str:
    a, v = disallowed[0]
    det_l = _label_for_script(a)
    more = f" (plus other disallowed script ink)" if len(disallowed) > 1 else ""
    req_l = ", ".join(_label_for_script(x) for x in req_sorted) if req_sorted else "n/a"
    return (
        f"Mismatch Detected: measurable threshed ink in {det_l} (≈{100.0 * v:.1f}%{more}) is not "
        f"in the allowlist for {location_name}. Jurisdiction allows: {req_l}."
    )


class UniversalComplianceEngine:
    """
    Registry of per-state/UT signboard threshed-ink rules for Indian jurisdictions.

    Each entry defines a mandatory *primary* BCP-47 base script, a minimum **share of
    threshed text ink** (0–1) for that script where policy prescribes a ratio, and
    whether **font/ordering** policy exists (e.g. Maharashtra) — the latter is surfaced
    in compliance ``details`` for governance; the flat ink ratio test does not prove
    font-size ordering.
    """

    STATE_RULES: ClassVar[dict[str, dict[str, Any]]] = {
        "AN": _r("Andaman and Nicobar Islands", "en", 0.0),
        "AP": _r("Andhra Pradesh", "te", 0.5),
        "AR": _r("Arunachal Pradesh", "en", 0.0),
        "AS": _r("Assam", "as", 0.0),
        "BR": _r("Bihar", "hi", 0.4),
        "CH": _r("Chandigarh", "pa", 0.0),
        "CG": _r("Chhattisgarh", "hi", 0.4),
        # Delhi municipal overlay can include text-vs-graphic readability policy
        "DL": _r("Delhi", "hi", 0.0, min_text_share_pct=0.55),
        "GA": _r("Goa", "kok", 0.0),  # Konkani (BCP-47: kok); often mixed — 0% floor
        "GJ": _r("Gujarat", "gu", 0.5),
        "HR": _r("Haryana", "hi", 0.4),
        "HP": _r("Himachal Pradesh", "hi", 0.0),
        "JK": _r("Jammu and Kashmir", "ur", 0.0),
        "JH": _r("Jharkhand", "hi", 0.4),
        "KA": _r("Karnataka", "kn", 0.6),
        "KL": _r("Kerala", "ml", 0.5),
        "LA": _r("Ladakh", "hi", 0.0),
        "LD": _r("Lakshadweep", "ml", 0.0),
        "MP": _r("Madhya Pradesh", "hi", 0.4),
        "MH": _r("Maharashtra", "mr", 0.5, font_ratio_required=True),
        "MN": _r("Manipur", "mni", 0.0),
        "ML": _r("Meghalaya", "en", 0.0),
        "MZ": _r("Mizoram", "lus", 0.0),
        "NL": _r("Nagaland", "en", 0.0),
        "OR": _r("Odisha", "or", 0.5),
        "PY": _r("Puducherry", "ta", 0.0),
        "PB": _r("Punjab", "pa", 0.0),
        "RJ": _r("Rajasthan", "hi", 0.4),
        "SK": _r("Sikkim", "ne", 0.0),
        "TN": _r("Tamil Nadu", "ta", 0.0),  # flat % often N/A; governance uses 5:3:2
        "TG": _r("Telangana", "te", 0.5),
        "TR": _r("Tripura", "bn", 0.0),
        "UP": _r("Uttar Pradesh", "hi", 0.4),
        "UT": _r("Uttarakhand", "hi", 0.4),
        "WB": _r("West Bengal", "bn", 0.0),
    }

    @classmethod
    def normalize_state_code(cls, state_code: str) -> str:
        s = (state_code or "").strip().upper()
        if s.startswith("IN-"):
            s = s[3:]
        s = _STATE_CODE_ALIASES.get(s, s)
        return s

    @classmethod
    def to_evaluation_rules(cls, state_code: str) -> Optional[dict[str, Any]]:
        """Shape expected by :func:`evaluate_compliance` (``primary``, ``min_local_lang_pct`` in 0–100, …)."""
        c = cls.normalize_state_code(state_code)
        if not c or c not in cls.STATE_RULES:
            return None
        raw = cls.STATE_RULES[c]
        m = float(raw["min_primary_area_pct"])
        return {
            "name": raw["name"],
            "state_code": c,
            "primary": raw["primary_iso_code"],
            "required_scripts": get_required_scripts_for_state_code(c),
            "min_local_lang_pct": 100.0 * m,
            "min_primary_area_pct": m,
            "font_ratio_required": bool(raw.get("font_ratio_required", False)),
            "min_text_share_pct": raw.get("min_text_share_pct"),
        }

    @classmethod
    def list_state_rows(cls) -> list[dict[str, str]]:
        return [
            {"code": k, "name": str(v["name"])}
            for k, v in sorted(cls.STATE_RULES.items(), key=lambda x: x[0])
        ]


def normalize_lang(code: str) -> str:
    c = (code or "").lower().strip()
    if not c or c in ("und", "unknown"):
        return c
    m = re.match(r"^([a-z]{2,3})(?:[-_].*)?$", c)
    if m:
        return m.group(1)
    return c.split("-")[0].lower() if "-" in c else c


def _ratio_to_pct(r: float) -> float:
    r = float(r)
    if r > 1.0 + 0.01:
        return r
    return 100.0 * r


def list_state_rules() -> list[dict[str, str]]:
    return UniversalComplianceEngine.list_state_rows()


def get_rules_for_state(state_code: str) -> Optional[dict[str, Any]]:
    return UniversalComplianceEngine.to_evaluation_rules(state_code)


# Backward compatibility: same keys as :attr:`UniversalComplianceEngine.STATE_RULES`
STATE_COMPLIANCE_RULES: dict[str, dict[str, Any]] = {
    code: UniversalComplianceEngine.to_evaluation_rules(code)  # type: ignore[misc]
    for code in UniversalComplianceEngine.STATE_RULES
}

SUPPORTED_STATE_CODES: frozenset[str] = frozenset(UniversalComplianceEngine.STATE_RULES.keys())


# ── Category overlays (statutory checks) ─────────────────────────────────────

_CATEGORY_LOGIC: dict[str, dict[str, Any]] = {
    # Minimal machine-checkable subset of your rulebook: keywords + mandatory regexes.
    "pharmacy": {
        "keywords": ["pharmacy", "chemist", "druggist", "aushadhi", "auṣadhi", "औषधि", "ಔಷಧಿ"],
        # common DL patterns: "DL No", "Drug Lic", "20B/21B", etc.
        "mandatory_regex": r"(?:DL|Drug\s*Lic(?:en[cs]e)?)[\s#:.-]*No\.?\s*[A-Z0-9/-]{4,}",
        "alt_regex": r"\b(20B|21B|20\s*B|21\s*B)\b",
        "id": "pharmacy_dl_number",
        "label": "Drug license number (DL No.)",
    },
    "clinic": {
        "keywords": ["clinic", "mbbs", "md", "bams", "hospital", "dr.", "doctor"],
        "mandatory_regex": r"(?:Reg(?:istration)?\s*No\.?|MCI|KMC|NMC)[\s#:.-]*[A-Z0-9/-]{3,}",
        "id": "clinic_registration_number",
        "label": "Registration number (Reg No.)",
    },
    "jewelry": {
        "keywords": ["hallmark", "huid", "bis", "916", "gold", "jewellery", "jewelry"],
        "mandatory_regex": r"\bHUID\b|\bBIS\b|\b916\b",
        "id": "jewelry_hallmark",
        "label": "BIS hallmark / purity (HUID/BIS/916)",
    },
    "liquor": {
        "keywords": ["bar", "liquor", "wine", "whisky", "whiskey", "beer"],
        "mandatory_regex": r"(?:Excise|License|Lic\.)[\s#:.-]*No\.?\s*[A-Z0-9/-]{3,}",
        "id": "liquor_excise_license",
        "label": "Excise license number",
    },
    "bank": {
        "keywords": ["bank", "atm", "branch", "ifsc"],
        "mandatory_regex": r"\bIFSC\b[\s#:.-]*[A-Z]{4}0[A-Z0-9]{6}\b",
        "id": "bank_ifsc",
        "label": "IFSC code",
    },
    "general_retail": {
        "keywords": ["store", "shop", "mart", "traders", "enterprises", "agency"],
        "mandatory_regex": r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
        "id": "gstin_number",
        "label": "GSTIN number",
    },
}


def _norm_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if not c:
        return ""
    # allow a few synonyms
    if c in ("medical", "doctor", "hospital"):
        return "clinic"
    if c in ("chemist", "drugstore", "medical store"):
        return "pharmacy"
    if c in ("jewel", "bullion"):
        return "jewelry"
    if c in ("bar", "bars", "liquor shop", "wine shop"):
        return "liquor"
    if c in ("bank/atm", "atm"):
        return "bank"
    if c in ("general retail", "retail", "general"):
        return "general_retail"
    return c


def evaluate_category_overlays(full_text: str, category: str, state_code: str) -> dict[str, Any] | None:
    """
    Category overlays from the rulebook (regex/keyword checks on OCR text).
    Returns a compact dict to be merged into compliance.governance.
    """
    cat = _norm_category(category)

    t = (full_text or "")
    t_low = t.lower()
    st = (state_code or "").strip().upper()
    checks: list[dict[str, Any]] = []

    def _mk_check(
        *,
        cid: str,
        label: str,
        rx: str,
        alt: str = "",
        kws: list[str] | None = None,
    ) -> dict[str, Any]:
        kw_hit = any(k.lower() in t_low for k in (kws or [])) if kws else None
        ok = False
        hit = ""
        if rx:
            m = re.search(rx, t, flags=re.IGNORECASE)
            if m:
                ok = True
                hit = m.group(0)[:80]
        if (not ok) and alt:
            m2 = re.search(alt, t, flags=re.IGNORECASE)
            if m2:
                ok = True
                hit = m2.group(0)[:80]
        return {
            "id": cid,
            "label": label,
            "ok": bool(ok),
            "keyword_present": kw_hit,
            "match": hit or None,
        }

    # 1) Category-native mandatory checks
    active_cats: list[str] = []
    if cat:
        active_cats.append(cat)
    else:
        # Auto-infer likely category from OCR text so mandatory IDs can still be flagged.
        for c0, r0 in _CATEGORY_LOGIC.items():
            kws0 = [k for k in (r0.get("keywords") or []) if isinstance(k, str) and k.strip()]
            if kws0 and any(k.lower() in t_low for k in kws0):
                active_cats.append(c0)
    # de-dupe
    active_cats = list(dict.fromkeys(active_cats))
    for c0 in active_cats:
        rule = _CATEGORY_LOGIC.get(c0)
        if not rule:
            continue
        kws = [k for k in (rule.get("keywords") or []) if isinstance(k, str) and k.strip()]
        rx = str(rule.get("mandatory_regex") or "").strip()
        alt = str(rule.get("alt_regex") or "").strip()
        checks.append(
            _mk_check(
                cid=str(rule.get("id") or f"{c0}_mandatory"),
                label=str(rule.get("label") or "Mandatory field"),
                rx=rx,
                alt=alt,
                kws=kws,
            )
        )

    # 2) GSTIN overlays from rulebook (state/category-dependent)
    # Delhi rule text explicitly mentions GSTIN legibility; Gujarat retail commonly mandates GSTIN display.
    # For machine checks, we require presence only (not legibility distance).
    gstin_rx = r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b"
    gstin_needed = (
        st in ("DL", "GJ")
        or cat in ("general_retail", "jewelry", "liquor", "bank")
        or any(c0 in ("general_retail", "jewelry", "liquor", "bank") for c0 in active_cats)
    )
    if gstin_needed:
        checks.append(
            _mk_check(
                cid="gstin_number",
                label="GSTIN number",
                rx=gstin_rx,
                kws=["gstin", "gst", "tax invoice"],
            )
        )

    if not checks:
        return None
    overall = "pass" if all(bool(c.get("ok")) for c in checks) else "fail"
    return {
        "category": cat or None,
        "inferred_categories": active_cats or None,
        "state_code": st or None,
        "checks": checks,
        "overall": overall,
    }


def evaluate_compliance(
    verified_data: dict[str, Any],
    location_rules: Optional[dict[str, Any]],
    blocks: Optional[list[dict[str, Any]]] = None,
    verified_text_json: Optional[dict[str, Any]] = None,
    detected_graphics: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Compare threshed language distribution against the selected state's rules
    (from :class:`UniversalComplianceEngine`).

    If ``blocks`` and ``verified_text_json`` are provided, per-block threshed weights
    and user-verified languages override ``verified_data`` [``language_ratios``] for the
    primary-language test.
    """
    if not location_rules:
        return {
            "status": "pending",
            "message": "No jurisdiction rules; supply state_code to evaluate signboard language mix.",
            "ruleset_id": None,
            "details": {"reason": "no_rules"},
        }

    primary = normalize_lang(str(location_rules.get("primary", "")))
    if not primary:
        return {
            "status": "uncertain",
            "message": "Rule set is missing a primary language code.",
            "ruleset_id": location_rules.get("name") or "unknown",
            "details": {},
        }

    if not isinstance(verified_data, dict) or not verified_data:
        return {
            "status": "uncertain",
            "message": "No signboard geometry (threshed ink) available for this file — cannot score compliance.",
            "ruleset_id": str(location_rules.get("name", "")) or None,
            "details": {"reason": "no_verified_data"},
        }

    vdata: dict[str, Any] = copy.deepcopy(verified_data)
    vt = dict(verified_text_json) if verified_text_json else {}
    if blocks and isinstance(blocks, list) and verified_text_json is not None:
        bl2 = [b for b in blocks if isinstance(b, dict)]
        if bl2:
            rb0 = rebuild_effective_threshed_language_ratios_from_blocks(
                bl2, dict(verified_text_json)
            )
            if rb0:
                vdata["language_ratios"] = rb0
                pa = normalize_lang(str(vdata.get("primary_language") or ""))
                if pa:
                    vdata["primary_ratio"] = float(rb0.get(pa, 0.0))
                vdata["dominant_language"] = (
                    max(rb0, key=rb0.get) if rb0 else vdata.get("dominant_language", "und")
                )
                vdata["monolingual_single_lang"] = _msl_after_remap(rb0)

    min_pct = float(location_rules.get("min_local_lang_pct", 0.0) or 0.0)
    ratios: dict[str, float] = vdata.get("language_ratios") or {}
    if not isinstance(ratios, dict) or not ratios:
        return {
            "status": "uncertain",
            "message": "Language ink ratios are empty; try a non-PDF, non-empty signboard image.",
            "ruleset_id": str(location_rules.get("name", "")) or None,
            "details": {"language_ratios": {}},
        }

    by_base: dict[str, float] = {}
    for raw_k, v in ratios.items():
        b = normalize_lang(str(raw_k))
        if b:
            by_base[b] = by_base.get(b, 0.0) + float(v)

    name = str(location_rules.get("name", "")) or "region"
    font_flag = bool(location_rules.get("font_ratio_required", False))
    min_text_share_pct = location_rules.get("min_text_share_pct")

    d0 = normalize_lang(str(vdata.get("dominant_language", "")))
    dominant_adj = d0
    if d0 in ("", "und") and by_base:
        non_und = {k: v for k, v in by_base.items() if k not in ("", "und")}
        if non_und:
            dominant_adj = max(non_und, key=non_und.get)

    # Jurisdiction / dominant script: dominant threshed ink must match the state allowlist
    # (e.g. DL: hi/en only — a Kannada-dominant board must not pass when min % is 0 % for Hindi)
    st_code = str(location_rules.get("state_code") or "").strip().upper()
    req_raw = location_rules.get("required_scripts")
    if not isinstance(req_raw, list) or not req_raw:
        req_set = {x for x in (primary, "en") if x}
    else:
        req_set = {normalize_lang(str(x)) for x in req_raw if str(x).strip()}
    if not req_set and primary:
        req_set = {primary, "en"}
    det_dom = normalize_lang(dominant_adj) if dominant_adj else ""
    if det_dom and det_dom not in ("", "und") and req_set and det_dom not in req_set:
        d_js: dict[str, Any] = {
            "jurisdiction_script_mismatch": True,
            "detected_dominant_script": det_dom,
            "required_scripts": sorted(req_set),
            "language_ratios": by_base,
            "dominant": dominant_adj,
        }
        if font_flag:
            d_js["font_ratio_required"] = True
        return {
            "status": "fail",
            "message": _jurisdiction_script_mismatch_message(
                st_code, det_dom, name, sorted(req_set)
            ),
            "ruleset_id": location_rules.get("name") or "region",
            "details": d_js,
        }

    mono_exempt = False
    msl = vdata.get("monolingual_single_lang")
    if msl is not None:
        msl = normalize_lang(str(msl))
    if min_pct > 0.0 and msl and msl == primary:
        mono_exempt = True
    if min_pct > 0.0 and not mono_exempt and primary in by_base:
        other_named = sum(
            v for k, v in by_base.items() if k not in (primary, "und", "")
        )
        if other_named < 0.01 and by_base.get(primary, 0.0) >= 0.99:
            mono_exempt = True
    if mono_exempt and min_pct > 0.0:
        dmono = {
            "required_lang": primary,
            "min_local_lang_pct": min_pct,
            "min_primary_area_pct": (location_rules.get("min_primary_area_pct")),
            "measured_local_lang_pct": 100.0,
            "language_ratios": by_base,
            "monolingual_exemption": True,
            "dominant": dominant_adj,
        }
        if font_flag:
            dmono["font_ratio_required"] = True
        return {
            "status": "pass",
            "message": (
                f"PASS · {name}: monolingual board in '{primary}' (≈100% of threshed ink in that script; "
                f"minimum {min_pct:.0f}% share is satisfied without a multi-language split)."
            ),
            "ruleset_id": location_rules.get("name") or "region",
            "details": dmono,
        }

    local_from_ratios = _ratio_to_pct(by_base.get(primary, 0.0))
    p_lang = normalize_lang(str(vdata.get("primary_language", "")))
    p_ratio = vdata.get("primary_ratio")
    if p_lang == primary and p_ratio is not None:
        local_pct = max(
            local_from_ratios,
            _ratio_to_pct(float(p_ratio)),
        )
    else:
        local_pct = local_from_ratios

    observed_primary = normalize_lang(str(vdata.get("primary_language", "")))
    dominant = dominant_adj

    passed = (min_pct == 0.0) or (local_pct + 0.5 >= min_pct)

    details: dict[str, Any] = {
        "required_lang": primary,
        "min_local_lang_pct": min_pct,
        "min_primary_area_pct": location_rules.get("min_primary_area_pct"),
        "measured_local_lang_pct": round(local_pct, 2),
        "language_ratios": by_base,
        "agent_primary": observed_primary,
        "dominant": dominant,
    }
    if font_flag:
        details["font_ratio_required"] = True
        details["font_ratio_note"] = (
            "Jurisdictions such as Maharashtra may require relative font sizes/ordering; "
            "threshed ink share alone does not verify font height."
        )

    # Optional municipal overlay: verified text-vs-graphic ratio using user-marked brand logos.
    try:
        if min_text_share_pct is not None:
            min_text_share = float(min_text_share_pct)
            # Text share proxy from OCR blocks
            text_share = 0.0
            if blocks and isinstance(blocks, list):
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    w = _per_block_threshed_weight(b)
                    if w > 0:
                        text_share += float(w)
            # Graphic share from detected graphics only when user verifies "brand_logo"
            graphic_share = 0.0
            if detected_graphics and isinstance(detected_graphics, list):
                for i, g in enumerate(detected_graphics):
                    if not isinstance(g, dict):
                        continue
                    v = vt.get(f"graphic_{i}")
                    is_logo = False
                    if isinstance(v, dict):
                        t0 = str(v.get("type") or "").strip().lower()
                        is_logo = (t0 == "brand_logo") or bool(v.get("brand_logo"))
                    if not is_logo:
                        continue
                    gs = float(g.get("threshed_ink_share") or 0.0)
                    if gs > 0:
                        graphic_share += gs
            tot_tg = text_share + graphic_share
            text_ratio = (text_share / tot_tg) if tot_tg > 0 else 1.0
            details["text_graphic_ratio"] = {
                "text_share": round(text_share, 5),
                "graphic_share": round(graphic_share, 5),
                "text_ratio": round(text_ratio, 5),
                "min_text_share_required": round(min_text_share, 3),
            }
            if text_ratio + 1e-9 < min_text_share:
                return {
                    "status": "fail",
                    "message": (
                        f"FAIL · {name}: text-vs-graphic share is {text_ratio*100:.1f}% text vs "
                        f"{(1.0-text_ratio)*100:.1f}% graphics; requires at least {min_text_share*100:.0f}% text."
                    ),
                    "ruleset_id": location_rules.get("name") or "region",
                    "details": details,
                }
    except Exception:
        # Never block primary language evaluation on ratio computation issues.
        pass

    if not passed:
        return {
            "status": "fail",
            "message": (
                f"FAIL · {name}: threshed ink for '{primary}' is {local_pct:.1f}%; "
                f"required ≥ {min_pct:.0f}%. (Dominant: {dominant or 'n/a'}.)"
            ),
            "ruleset_id": location_rules.get("name") or "region",
            "details": details,
        }

    # Every script with material threshed ink must be in the allowlist (not just dominant).
    # This check runs only if the primary-share test passed (or min_pct==0),
    # so the headline failure reason is consistent (ratio failures remain ratio failures).
    disallowed: list[tuple[str, float]] = []
    for k, v in by_base.items():
        if not k or k in ("", "und"):
            continue
        floor = max(JURISDICTION_MIN_INK_FRACTION, float(_JURISDICTION_NOISE_FLOOR.get(k, 0.0)))
        if float(v) < floor:
            continue
        if k not in req_set:
            disallowed.append((k, float(v)))
    if disallowed:
        disallowed.sort(key=lambda t: t[1], reverse=True)
        d_mix: dict[str, Any] = {
            "jurisdiction_unexpected_script_ink": True,
            "disallowed_ink": [{"script": a, "ratio_ink": round(b, 4)} for a, b in disallowed],
            "required_scripts": sorted(req_set),
            "language_ratios": by_base,
            "dominant": dominant_adj,
        }
        if font_flag:
            d_mix["font_ratio_required"] = True
        return {
            "status": "fail",
            "message": _jurisdiction_unexpected_ink_in_mix_message(
                name, disallowed, sorted(req_set)
            ),
            "ruleset_id": location_rules.get("name") or "region",
            "details": d_mix,
        }

    return {
        "status": "pass",
        "message": (
            f"PASS · {name}: '{primary}' covers {local_pct:.1f}% of detected script ink (minimum {min_pct:.0f}%)."
        ),
        "ruleset_id": location_rules.get("name") or "region",
        "details": details,
    }


def _parse_verified_value(v: Any) -> tuple[Optional[str], Optional[str]]:
    if v is None:
        return None, None
    if isinstance(v, str):
        return v, None
    if isinstance(v, dict):
        st: Optional[str] = None
        if "text" in v:
            tx = v.get("text")
            st = None if tx is None else str(tx)
        lang = v.get("verified_language") or v.get("language") or v.get("lang")
        lang_s = str(lang).strip() if lang else None
        return (st, normalize_lang(lang_s) if lang_s else None)
    return None, None


def _block_ocr_base_lang(blk: dict[str, Any]) -> str:
    lh = blk.get("languages")
    if isinstance(lh, list) and lh:
        first = lh[0]
        c = (first.get("code") or first.get("Code") or "") if isinstance(first, dict) else ""
        if c and str(c).strip():
            b = normalize_lang(str(c).strip())
            if b:
                return b
    return "und"


def _per_block_threshed_weight(blk: dict[str, Any]) -> float:
    w = blk.get("threshed_ink_share")
    if w is not None:
        return max(0.0, float(w))
    n = blk.get("norm")
    t = 0.0
    if isinstance(n, dict):
        t = float(n.get("threshed_area_pct") or 0) / 100.0
    if t > 0:
        return max(0.0, t)
    return 0.0


def rebuild_effective_threshed_language_ratios_from_blocks(
    blocks: list[dict[str, Any]],
    verified_text_json: dict[str, Any] | None,
) -> dict[str, float] | None:
    if not blocks:
        return None
    vt = dict(verified_text_json) if verified_text_json else {}
    by: dict[str, float] = {}
    dblocks = [b for b in blocks if isinstance(b, dict)]
    use_share = bool(dblocks) and all(
        b.get("threshed_ink_share") is not None for b in dblocks
    )
    if use_share:
        for i, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                continue
            w = _per_block_threshed_weight(raw)
            _tx, to_lang = _parse_verified_value(vt.get(str(i)))
            ocr_lang = _block_ocr_base_lang(raw)
            lang = (normalize_lang(to_lang) if to_lang else ocr_lang) or "und"
            by[lang] = by.get(lang, 0.0) + w
    else:
        mlist: list[float] = []
        for i, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                mlist.append(0.0)
                continue
            n = raw.get("norm")
            t = 0.0
            if isinstance(n, dict):
                t = float(n.get("threshed_area_pct") or 0) / 100.0
            if t <= 0.0 and raw.get("threshed_ink_share") is not None:
                t = max(0.0, float(raw["threshed_ink_share"]))
            mlist.append(t)
        s_ink = sum(mlist) or 1.0
        for i, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                continue
            w = mlist[i] / s_ink
            st, to_lang = _parse_verified_value(vt.get(str(i)))
            ocr_lang = _block_ocr_base_lang(raw)
            lang = (normalize_lang(to_lang) if to_lang else None) or ocr_lang
            if not lang:
                lang = "und"
            by[lang] = by.get(lang, 0.0) + w
    tot = sum(max(0.0, v) for v in by.values()) or 1.0
    by2: dict[str, float] = {
        k: max(0.0, v) / tot
        for k, v in by.items()
        if k and max(0.0, v) > 0
    }
    ssum = sum(by2.values()) or 1.0
    return {k: round(v / ssum, 5) for k, v in by2.items()}


def recompute_geometry_after_verified_overrides(
    geo: dict[str, Any] | None,
    blocks: list[dict[str, Any]],
    verified_text_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not geo or not isinstance(geo, dict):
        return geo
    if not verified_text_json or not blocks:
        return copy.deepcopy(geo)
    g = copy.deepcopy(geo)
    bl = [dict(b) for b in blocks if isinstance(b, dict)]
    if not bl:
        return g
    rb = rebuild_effective_threshed_language_ratios_from_blocks(
        bl,
        dict(verified_text_json),
    )
    if not rb:
        return g
    g["language_ratios"] = rb
    p_lang0 = str(g.get("primary_language", "") or "")
    primary_agent = normalize_lang(p_lang0) or p_lang0
    g["primary_ratio"] = float(rb.get(primary_agent, 0.0)) if primary_agent else 0.0
    g["dominant_language"] = max(rb, key=rb.get) if rb else g.get("dominant_language", "und")
    g["monolingual_single_lang"] = _msl_after_remap(rb)
    return g


def _msl_after_remap(by2: dict[str, float]) -> str | None:
    non_und = {k: v for k, v in by2.items() if k and k != "und"}
    if not non_und:
        return None
    top, tv = max(non_und.items(), key=lambda x: x[1])
    other = sum(v for k, v in non_und.items() if k != top)
    if other < 0.01 and tv >= 0.99:
        return top
    return None


def block_dicts_with_verified_text(
    raw_blocks: list[dict[str, Any]],
    verified_text_json: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not raw_blocks or not verified_text_json:
        return [dict(b) for b in raw_blocks] if raw_blocks else []
    out: list[dict[str, Any]] = []
    for i, b in enumerate(raw_blocks):
        b2: dict[str, Any] = dict(b) if isinstance(b, dict) else {}
        key = str(i)
        if key in verified_text_json:
            t, lang = _parse_verified_value(verified_text_json[key])
            if t is not None and str(t).strip() != "":
                b2["text"] = str(t)
            if lang:
                b2["languages"] = [{"code": lang, "confidence": 1.0}]
        out.append(b2)
    return out
