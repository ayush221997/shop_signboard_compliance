"""
State / city shop signboard rules (simplified from public order sources).
Citations (snippet IDs) are for traceability; enforcement text is not auto-verified.
"""

from __future__ import annotations

from typing import Any, Optional

# Keys = 2-letter state/UT code used in the app (e.g. IN-KA -> KA)
# automated_checks: tags understood by governance_evaluate.py
GOVERNANCE_RULE_TABLE: dict[str, dict[str, Any]] = {
    "KA": {
        "jurisdiction": "Karnataka (Bengaluru example)",
        "mandatory_languages": ["kn"],
        "min_script_area_pct": 60.0,
        "positioning": "mandatory language must appear in the upper half of the board (heuristic: block center Y < 50% of sign height) [1]",
        "font_ratio_notes": "60% of signboard area in Kannada [1]",
        "material_summary": "PVC/flex < 100 µm may be restricted; materials not verified from image [2]",
        "enforcement_summary": "Fines and trade-license linkage per local rules [3][4]",
        "citations": ["[1]", "[2]", "[3]", "[4]"],
        "automated_checks": ["ink_ratio", "kannada_upper_half"],
    },
    "TN": {
        "jurisdiction": "Tamil Nadu (Chennai example)",
        "mandatory_languages": ["ta"],
        "ratio_tamil_english_other": (5, 3, 2),  # target share of (ta+en+rest) for ta,en,other
        "min_script_area_pct": None,  # prefer 5:3:2 rule over a flat 60% minimum
        "positioning": "Tamil should be at the top of the board [5]",
        "max_signage_sqm_note": "Establishment area rules e.g. <9 sq.m — not from image [1]",
        "citations": ["[5]", "[1]"],
        "automated_checks": ["tn_ratio_5_3_2", "tamil_top_block"],
    },
    "WB": {
        "jurisdiction": "West Bengal (Kolkata example)",
        "mandatory_languages": ["bn"],
        "min_script_area_pct": None,
        "positioning": "Bengali at top of sign/hoarding [6][7]",
        "size_rule": "Bengali font/area at least as large as any other language (ink ratio) [8]",
        "material_summary": "Structural rules — not from image [9][7]",
        "citations": ["[6]", "[7]", "[8]", "[9]", "[10]"],
        "automated_checks": ["bengali_top_block", "bengali_ink_geq_others"],
    },
    "PB": {
        "jurisdiction": "Punjab",
        "mandatory_languages": ["pa"],
        "min_script_area_pct": None,
        "positioning": "Gurmukhi (Punjabi) at the top [12][13]",
        "size_rule": "Punjabi display larger than other languages (ink) [11]",
        "citations": ["[11]", "[12]", "[13]", "[14]"],
        "automated_checks": ["punjabi_top_block", "punjabi_ink_geq_others"],
    },
    "DL": {
        "jurisdiction": "Delhi",
        "mandatory_languages": ["hi", "en", "pa", "ur"],
        "sequence_note": "Display order Hindi → English → Punjabi → Urdu [15][17]",
        "size_note": "Uniform size across four languages (not measurable from threshed ink) [15][18]",
        "display_rules": "Static digital min. duration, spacing — not from static image [19]",
        "citations": ["[15]", "[16]", "[17]", "[18]", "[19]", "[20]"],
        "automated_checks": ["hindi_first_sequence"],
    },
    "OR": {
        "jurisdiction": "Odisha",
        "mandatory_languages": ["or"],
        "min_script_area_pct": 60.0,
        "positioning": "Odia prominent, clearly displayed (position heuristic) [citations in policy]",
        "citations": [],
        "automated_checks": ["ink_ratio", "odia_top_or_prominent"],
    },
    "TG": {
        "jurisdiction": "Telangana (Hyderabad example)",
        "mandatory_languages": ["te"],
        "min_script_area_pct": None,
        "visibility_note": "Mandatory Telugu visibility (language presence + size proxy via ink) [21][22]",
        "citations": ["[21]", "[22]", "[2]", "[23]"],
        "automated_checks": ["te_visibility_combined"],
    },
    "AP": {
        "jurisdiction": "Andhra Pradesh",
        "mandatory_languages": ["te"],
        "min_script_area_pct": None,
        "visibility_note": "Prominent Telugu [21][22]; flex rules [24]",
        "citations": ["[21]", "[22]", "[24]"],
        "automated_checks": ["te_visibility_combined"],
    },
    "MH": {
        "jurisdiction": "Maharashtra (Mumbai example)",
        "mandatory_languages": ["mr"],
        "min_script_area_pct": None,
        "positioning": "Marathi in Devanagari at beginning / top of board",
        "size_rule": "Marathi not smaller than any other language (ink ratio)",
        "noc_note": "Traffic/illuminated hoarding permits — not from image [7]",
        "citations": ["[7]"],
        "automated_checks": ["marathi_top_block", "marathi_ink_geq_others"],
    },
    "GJ": {
        "jurisdiction": "Gujarat (Ahmedabad/Surat example)",
        "mandatory_languages": ["gu"],
        "min_script_area_pct": None,
        "visibility_note": "Gujarati with priority visibility [25]; façade 25% etc. not from image [26]",
        "citations": ["[25]", "[26]"],
        "automated_checks": ["gujarati_ink_dominant"],
    },
    "AS": {
        "jurisdiction": "Assam (example cities)",
        "mandatory_languages": ["as"],
        "min_script_area_pct": None,
        "pairing_note": "Assamese alongside English — both should appear in OCR [27]",
        "citations": ["[27]"],
        "automated_checks": ["assamese_plus_english_present"],
    },
    "HP": {
        "jurisdiction": "Himachal Pradesh",
        "mandatory_languages": [],
        "material_only": "PVC flex restrictions [2] — not verifiable from OCR image",
        "citations": ["[2]"],
        "automated_checks": [],
    },
}


def get_governance_state_codes() -> list[str]:
    return sorted(GOVERNANCE_RULE_TABLE.keys())
