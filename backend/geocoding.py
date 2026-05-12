"""Reverse geocoding for display + India state code inference (Nominatim)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from compliance_rules import SUPPORTED_STATE_CODES

logger = logging.getLogger(__name__)

# Indian state/UT name (as in OSM) → 2-letter code used in this app
INDIA_STATE_NAME_TO_CODE: dict[str, str] = {
    "karnataka": "KA",
    "maharashtra": "MH",
    "tamil nadu": "TN",
    "telangana": "TG",
    "andhra pradesh": "AP",
    "kerala": "KL",
    "gujarat": "GJ",
    "west bengal": "WB",
    "delhi": "DL",
    "national capital territory of delhi": "DL",
    "uttar pradesh": "UP",
    "madhya pradesh": "MP",
    "rajasthan": "RJ",
    "punjab": "PB",
    "haryana": "HR",
    "bihar": "BR",
    "odisha": "OR",
    "assam": "AS",
    "goa": "GA",
    "jammu and kashmir": "JK",
    "ladakh": "LA",
    "uttarakhand": "UT",
    "himachal pradesh": "HP",
    "jharkhand": "JH",
    "chhattisgarh": "CG",
    "meghalaya": "ML",
    "manipur": "MN",
    "mizoram": "MZ",
    "nagaland": "NL",
    "arunachal pradesh": "AR",
    "sikkim": "SK",
    "tripura": "TR",
}


def _norm_state(s: str) -> str:
    return s.strip().lower() if s else ""


def infer_india_state_code_from_address(address: dict[str, Any]) -> Optional[str]:
    st = _norm_state(
        (address or {}).get("state")
        or (address or {}).get("state_district")
        or ""
    )
    if not st:
        return None
    if st in INDIA_STATE_NAME_TO_CODE:
        return INDIA_STATE_NAME_TO_CODE[st]
    for name, code in INDIA_STATE_NAME_TO_CODE.items():
        if name in st or st in name:
            return code
    return None


def city_from_address(address: dict[str, Any]) -> str:
    a = address or {}
    for k in (
        "city",
        "town",
        "village",
        "municipality",
        "suburb",
        "county",
    ):
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    st = a.get("state")
    if isinstance(st, str) and st.strip():
        return st.strip()
    return "Unknown"


async def reverse_geocode(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Returns { city, state_name, state_code, country_code, raw }.
    `state_code` is our 2-letter key when country is IN and the state is mapped.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
        "addressdetails": 1,
    }
    headers = {
        "User-Agent": os.getenv(
            "NOMINATIM_USER_AGENT",
            "OCR-Compliance-App/1.0 (local dev; support@example.com)",
        ),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data: dict = r.json()

    addr = data.get("address") or {}
    city = city_from_address(addr)
    state_name = (addr.get("state") or "").strip() or None
    cc = (addr.get("country_code") or "").upper() or None

    state_code: Optional[str] = None
    if cc == "IN" or (data.get("display_name") or "").find("India") >= 0:
        state_code = infer_india_state_code_from_address(addr)
        if state_code and state_code not in SUPPORTED_STATE_CODES:
            # Known code but we have no rules — still return for UI
            pass

    return {
        "city": city,
        "state_name": state_name,
        "state_code": state_code,
        "country_code": cc,
        "raw": data,
    }
