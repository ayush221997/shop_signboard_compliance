"""
FastAPI OCR service — Google Vision: text, language detection, logo detection,
per-block areas, and structured fields (e.g. GSTIN). State-based compliance
rules are applied later; see ComplianceInfo on the response.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv(Path(__file__).resolve().parent / ".env")

from google.api_core.exceptions import GoogleAPICallError, PermissionDenied, Unauthenticated
from google.cloud import vision
from compliance_rules import (
    block_dicts_with_verified_text,
    evaluate_compliance,
    get_rules_for_state,
    list_state_rules,
    recompute_geometry_after_verified_overrides,
    evaluate_category_overlays,
)
from governance_evaluate import ka_ocr_word_upper_layout_hint, merge_governance_compliance
from database import get_db, init_db
from language_columns import threshed_language_audit_columns
from db_models import AuditRecord, ComplianceStatus
from geocoding import reverse_geocode
from gemini_postprocess import (
    get_last_gemini_error,
    gemini_detect_script_confidences,
    should_translate_block,
    translate_blocks_to_english,
    translate_texts_on_demand,
)
from gstin_extract import enrich_gstins
from blur_check import is_too_blurry_bgr
from image_metadata import build_image_metadata
from image_prep import should_transcode_to_png_for_vision, transcode_to_png
from signboard_agent import SILENT_INK_PCT_FLOOR, SILENT_INK_PCT_STRONG, SignboardGeometryAgent
from storage import ensure_local_dir, save_image_bytes
from vision_preprocess import (
    bgr_ensure_min_long_edge,
    prepare_bgr_for_document_ocr,
    prebinarize_bgr_smart,
    should_prebinarize_for_vision_ocr,
)
from vision_verification import extract_verified_words_from_block


# ── Response models ─────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class NormArea(BaseModel):
    """Position and size as percentages of the full image (0–100)."""
    x_pct: float   # left edge % from image left
    y_pct: float   # top edge % from image top
    w_pct: float   # width as % of image width
    h_pct: float   # height as % of image height
    area_pct: float  # w_pct × h_pct / 100 — share of signboard (bounding box)
    threshed_area_pct: Optional[float] = None  # ink pixels on rectified board / total board (0–100)


class LanguageHint(BaseModel):
    code: str          # BCP-47 language code e.g. "en", "fr", "zh"
    confidence: float  # 0.0–1.0


class VerifiedWord(BaseModel):
    """OCR word with Vision confidence; is_doubtful if symbol-level min confidence < 0.85."""
    text: str
    confidence: float = 0.0
    is_doubtful: bool = False


class TextBlock(BaseModel):
    text: str
    bounds: Optional[BoundingBox] = None
    norm: Optional[NormArea] = None       # position relative to the signboard
    languages: list[LanguageHint] = []
    words: list[VerifiedWord] = []  # per-word confidences; is_doubtful if word/symbol < 0.85
    translated_text: Optional[str] = None  # English, when block language is not en
    # Block-level review (e.g. OpenCV sees ink but Vision returns no text)
    is_doubtful: bool = False
    warning_type: Optional[str] = None
    warning_message: Optional[str] = None
    # Share of total threshed ink across returned blocks (0–1); used when merging manual language
    threshed_ink_share: Optional[float] = None


class LogoResult(BaseModel):
    name: str
    confidence: float
    bounds: Optional[BoundingBox] = None
    norm: Optional[NormArea] = None       # position relative to the signboard


class GstinFinding(BaseModel):
    value: str
    format_valid: bool = False
    checksum_valid: bool = False
    block_index: Optional[int] = None  # text region index, if found inside one block


class ComplianceInfo(BaseModel):
    """
    Signboard / shopfront compliance (language ratios, min sizes, etc.) is evaluated
    only after you supply a state (or city) ruleset. OCR returns structured data first.
    `governance` is optional heuristics against published shop signboard policy for the state.
    """
    status: str = "pending"  # pending | pass | fail | partial | uncertain (governance can add partial)
    message: str = (
        "PENDING - STATE REQUIRED. Choose a state/UT to run COMPLai threshed-ink rules."
    )
    ruleset_id: Optional[str] = None
    governance: Optional[dict[str, Any]] = None


class SignboardGeometrySummary(BaseModel):
    """Threshed-pixel (OpenCV) language area metrics from SignboardGeometryAgent."""
    language_ratios: dict[str, float] = {}
    language_details: dict = {}  # lang -> { ink_pixels, board_ratio, text_ratio, block_count }
    compliance_status: str = "UNCERTAIN"
    confidence_score: float = 0.0
    dominant_language: str = ""
    primary_language: str = ""
    primary_ratio: float = 0.0
    # Filled when ≈100% of threshed ink is a single non-und script; drives monolingual pass in rules
    monolingual_single_lang: Optional[str] = None
    # Threshed ink as % of canonical 1024×1024 board, Vision block order (same as first page’s blocks)
    per_block_threshed_board_pct: Optional[list[float]] = None
    # Residual contour graphics not covered by OCR text boxes
    detected_graphics: Optional[list[dict[str, Any]]] = None


class DetectedGraphic(BaseModel):
    x: int
    y: int
    w: int
    h: int
    type: str = "unknown_graphic"
    threshed_ink_share: Optional[float] = None
    bbox_area_share: Optional[float] = None


class OcrResult(BaseModel):
    text: str
    image_width: int = 0
    image_height: int = 0
    languages: list[LanguageHint] = []
    blocks: list[TextBlock] = []
    logos: list[LogoResult] = []
    gstin: list[GstinFinding] = []
    compliance: ComplianceInfo = ComplianceInfo()
    # Full signboard in English: English blocks as-is, others translated when Gemini is configured
    translated_text: str = ""
    signboard_geometry: Optional[SignboardGeometrySummary] = None
    detected_graphics: list[DetectedGraphic] = []
    is_pdf_ocr: bool = False
    audit_id: Optional[str] = None
    resolved_location: Optional[dict[str, Any]] = None
    verified_text_json: Optional[dict[str, Any]] = None
    # File + embedded metadata (EXIF, dimensions) — not used for rules; audit trail / UI
    image_metadata: Optional[dict[str, Any]] = None
    # Optional store category for statutory overlays (pharmacy/clinic/etc.)
    category: Optional[str] = None


class AuditUpdate(BaseModel):
    user_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    state_code: Optional[str] = None
    city: Optional[str] = None
    category: Optional[str] = None
    verified_text_json: Optional[dict[str, Any]] = None


class TranslateRequest(BaseModel):
    """On-demand English translation of OCR segments (uses Gemini if API key is set)."""
    texts: list[str]
    languages: Optional[list[str]] = None  # optional per-block BCP-47, same length as texts


class TranslateResponse(BaseModel):
    translations: list[Optional[str]]
    detail: Optional[str] = None  # e.g. missing API key when all results are null


class SuggestCornersResult(BaseModel):
    """
    Auto-detected quadrilateral for perspective correction, in **original image pixel** space.
    Order: top-left, top-right, bottom-right, bottom-left (maps to `sortCorners` on the client).
    """

    corners: list[list[float]]  # length 4, each [x, y]
    method: str  # "text_hull" | "full_image"
    image_width: int
    image_height: int


def _compliance_info_from_eval(ev: dict) -> ComplianceInfo:
    st = (ev.get("status") or "pending").lower()
    return ComplianceInfo(
        status=st,
        message=str(ev.get("message", "")),
        ruleset_id=ev.get("ruleset_id") if ev.get("ruleset_id") is not None else None,
    )


def _compliance_to_db_code(c: ComplianceInfo) -> str:
    s = (c.status or "").lower()
    if s == "pass":
        return ComplianceStatus.PASS.value
    if s == "fail":
        return ComplianceStatus.FAIL.value
    if s in ("uncertain", "unknown", "partial"):
        return ComplianceStatus.UNCERTAIN.value
    return ComplianceStatus.PENDING.value

# ── Vision helpers ───────────────────────────────────────────────────────────

def _poly_to_bounds(bounding_poly) -> Optional[BoundingBox]:
    verts = list(bounding_poly.vertices)
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return BoundingBox(
        x=min(xs), y=min(ys),
        w=max(xs) - min(xs),
        h=max(ys) - min(ys),
    )


def _to_norm(bounds: Optional[BoundingBox], img_w: int, img_h: int) -> Optional[NormArea]:
    """Convert pixel bounds to percentages of the full image (signboard)."""
    if bounds is None or img_w <= 0 or img_h <= 0:
        return None
    x_pct = round(bounds.x * 100 / img_w, 1)
    y_pct = round(bounds.y * 100 / img_h, 1)
    w_pct = round(bounds.w * 100 / img_w, 1)
    h_pct = round(bounds.h * 100 / img_h, 1)
    area_pct = round(w_pct * h_pct / 100, 2)
    return NormArea(x_pct=x_pct, y_pct=y_pct, w_pct=w_pct, h_pct=h_pct, area_pct=area_pct)


def _get_languages(prop) -> list[LanguageHint]:
    return [
        LanguageHint(code=dl.language_code, confidence=round(dl.confidence, 3))
        for dl in prop.detected_languages
        if dl.language_code
    ]


def _block_text(block) -> str:
    """Reconstruct text from a block's word/symbol tree."""
    lines: list[str] = []
    for para in block.paragraphs:
        words: list[str] = []
        for word in para.words:
            words.append("".join(s.text for s in word.symbols))
        if words:
            lines.append(" ".join(words))
    return "\n".join(lines)


def _code_is_kannada(code: str | None) -> bool:
    c = (code or "").lower().replace("_", "-")
    return c == "kn" or c.startswith("kn-")


def _vision_block_has_kannada(block) -> bool:
    """True if the Vision block or any word is labeled with a Kannada language hint."""
    try:
        prop = getattr(block, "property", None)
        if prop and list(getattr(prop, "detected_languages", None) or []):
            for dl in prop.detected_languages:
                c = getattr(dl, "language_code", None) or ""
                if _code_is_kannada(c):
                    return True
        for para in getattr(block, "paragraphs", None) or []:
            for word in getattr(para, "words", None) or []:
                wprop = getattr(word, "property", None)
                if not wprop or not list(getattr(wprop, "detected_languages", None) or []):
                    continue
                for dl in wprop.detected_languages:
                    c = getattr(dl, "language_code", None) or ""
                    if _code_is_kannada(c):
                        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ── Vision client ────────────────────────────────────────────────────────────

def _build_client() -> vision.ImageAnnotatorClient:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        from google.api_core.client_options import ClientOptions
        return vision.ImageAnnotatorClient(
            client_options=ClientOptions(api_key=api_key)
        )
    return vision.ImageAnnotatorClient()


_client: vision.ImageAnnotatorClient | None = None
_agent: SignboardGeometryAgent | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _client, _agent
    try:
        init_db()
    except Exception as e:
        logging.getLogger(__name__).warning("Database init: %s", e)
    _client = _build_client()
    _agent = SignboardGeometryAgent(
        primary_language=os.getenv("PRIMARY_LANGUAGE", "kn"),
        compliance_threshold=float(os.getenv("COMPLIANCE_THRESHOLD", "0.60")),
        ink_tag_confidence=float(os.getenv("OCR_INK_TAG_CONFIDENCE", "0.85")),
    )
    logging.getLogger(__name__).info(
        "OCR: uploads accept common types; AVIF/HEIC/JXL are transcoded to PNG for Google Vision "
        "(Vision itself only supports JPEG, PNG, WebP, GIF, BMP, TIFF, PDF, etc.)."
    )
    yield
    _client = None
    _agent = None


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="OCR API (Google Vision)", lifespan=lifespan)

app.mount(
    "/files",
    StaticFiles(directory=str(ensure_local_dir().resolve())),
    name="files",
)

def _cors_allow_origins() -> list[str]:
    base = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    extra = os.getenv("CORS_EXTRA_ORIGINS", "")
    for part in extra.split(","):
        p = part.strip()
        if p:
            base.append(p)
    return base


# Regex: phones/tablets hitting Vite on LAN (192.168.x / 10.x / 172.16–31.x)
_CORS_LAN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_origin_regex=_CORS_LAN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _sniff_looks_like_vision_input(data: bytes) -> bool:
    """
    True if bytes look like a bitmap image, PDF, common camera raw container,
    or SVG (Vision supports common raster and PDF; other formats are passed through).
    """
    if not data or len(data) < 4:
        return False
    b = data
    if b[:3] == b"\xff\xd8\xff":  # JPEG
        return True
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":  # WebP
        return True
    if b[:4] == b"%PDF":  # PDF
        return True
    if b[:2] == b"BM":  # BMP
        return True
    if len(b) >= 4:
        if b[:2] == b"II" and b[2:4] == b"\x2a\x00":  # TIFF little-endian
            return True
        if b[:2] == b"MM" and b[2:4] == b"\x00\x2a":  # TIFF big-endian
            return True
    if len(b) >= 12 and b[4:8] == b"ftyp":  # HEIC, AVIF, some JPEG2000, MP4-island etc.
        return True
    head = b[: min(len(b), 4000)]
    t = head.lstrip()
    if t[:1] in (b"<", b"{"):
        h = t[:2000]
        if b"svg" in h.lower() or b"<svg" in h or h.startswith(b"<?xml"):
            return True
    if b"svg" in head[:500].lower() and b"<" in head[:50]:
        return True
    return False


# Broad filename fallbacks when the browser reports application/octet-stream or no type
_IMAGE_NAME = re.compile(
    r"\.(jpe?g|jpe|jfif|jif|pjp|pjpeg|png|gif|webp|bmp|dib|tiff?|tif|ico|"
    r"heic|heif|hif|avif|jxl|svgz?|jp2|j2[ck]|jpc|jpx|jxr|hdp|wdp|bpg|"
    r"apng|mng|qoi|dng|cr2|nef|raw?|psb?|jps|jph)$",
    re.IGNORECASE,
)
_PDF_NAME = re.compile(r"\.(pdf|ai)$", re.IGNORECASE)

# Some stacks send rasters with application/* (rare but seen with AVIF/HEIC uploaders)
_EXTRA_IMAGE_ALIASES = frozenset(
    {
        "application/avif",
        "application/x-avif",
        "application/heic",
        "application/heif",
        "application/x-heic",
    }
)


def _upload_allowed(
    declared_mime: str, filename: str | None, data: bytes
) -> bool:
    """
    `declared_mime` = Content-Type (first segment), lowercased, e.g. image/avif, application/pdf.
    """
    if not data:
        return False
    ct = declared_mime
    if ct.startswith("image/") or ct == "application/pdf" or ct in _EXTRA_IMAGE_ALIASES:
        return True
    if _sniff_looks_like_vision_input(data):
        return True
    if ct in ("application/octet-stream", "binary/octet-stream", "application/x-download", ""):
        n = (filename or "").lower()
        if n and (_IMAGE_NAME.search(n) or _PDF_NAME.search(n)):
            return True
    return False


def _file_is_pdf(declared: str, filename: str | None, buf: bytes) -> bool:
    d = (declared or "").split(";", 1)[0].strip().lower()
    if d == "application/pdf":
        return True
    if (filename or "").lower().endswith(".pdf"):
        return True
    return len(buf) > 4 and buf[:4] == b"%PDF"


def _location_compliance(
    geo: Optional[SignboardGeometrySummary],
    state_in: Optional[str],
    rules: Optional[dict[str, Any]],
    is_pdf: bool,
    blocks: Optional[list[TextBlock]] = None,
    verified_text_json: Optional[dict[str, Any]] = None,
) -> ComplianceInfo:
    if not state_in:
        return ComplianceInfo(
            status="pending",
            message=(
                "State/UT not set — pick it in the Run OCR form (or send state_code, or lat+lon) "
                "so threshed-ink and governance rules can run."
            ),
            ruleset_id=os.getenv("COMPLIANCE_RULESET_ID") or None,
        )
    if not rules:
        return ComplianceInfo(
            status="uncertain",
            message=f"No signboard language ruleset is configured in the engine for {state_in}.",
            ruleset_id=state_in,
        )
    if is_pdf or not geo:
        return ComplianceInfo(
            status="uncertain",
            message=(
                f"Ruleset {rules.get('name', state_in)} needs threshed-ink language ratios; "
                "not available for PDFs or if geometry could not be computed for this file."
            ),
            ruleset_id=str(rules.get("name", state_in)),
        )
    geo_d = geo.model_dump()
    blk_d = [b.model_dump(mode="json") for b in (blocks or [])]
    ev = evaluate_compliance(
        geo_d,
        rules,
        blk_d,
        dict(verified_text_json) if isinstance(verified_text_json, dict) else None,
        geo_d.get("detected_graphics") if isinstance(geo_d, dict) else None,
    )

    # Optional Gemini tie-breaker: if compliance fails because of a disallowed script ink mix,
    # ask Gemini what scripts are actually present in OCR text. If Gemini is confident (>=0.85)
    # that the disallowed script is NOT present, treat it as a misfire and re-score.
    try:
        use_gemini = (os.getenv("OCR_GEMINI_SCRIPT_OVERRIDE") or "1").strip().lower() in ("1", "true", "yes", "on")
        conf_thr = float(os.getenv("OCR_GEMINI_SCRIPT_CONFIDENCE_MIN", "0.85"))
    except Exception:
        use_gemini = True
        conf_thr = 0.85

    try:
        if use_gemini and isinstance(ev, dict) and str(ev.get("status", "")).lower() == "fail" and blocks:
            det = ev.get("details") if isinstance(ev.get("details"), dict) else None
            if det and (
                det.get("jurisdiction_unexpected_script_ink") is True
                or det.get("jurisdiction_script_mismatch") is True
            ):
                ratios = det.get("language_ratios")
                if isinstance(ratios, dict):
                    # Suspects: disallowed scripts (or dominant script mismatch)
                    suspects: set[str] = set()
                    dis = det.get("disallowed_ink")
                    if isinstance(dis, list):
                        for row in dis:
                            if isinstance(row, dict) and isinstance(row.get("script"), str):
                                suspects.add(row["script"].strip().lower())
                    if det.get("jurisdiction_script_mismatch") is True and isinstance(det.get("detected_dominant_script"), str):
                        suspects.add(det["detected_dominant_script"].strip().lower())

                    txts = [(b.text or "").strip() for b in blocks if (b.text or "").strip()]
                    if txts and suspects:
                        gconf = gemini_detect_script_confidences(txts)
                        if gconf:
                            cleaned = dict(geo_d)
                            lr0 = cleaned.get("language_ratios")
                            if isinstance(lr0, dict):
                                lr = {str(k): float(v) for k, v in lr0.items() if k is not None}
                                removed: list[str] = []
                                for s in list(suspects):
                                    s0 = s.split("-", 1)[0]
                                    c = float(gconf.get(s0, 0.0))
                                    # If Gemini is confident the suspect script is present, KEEP it.
                                    # If Gemini confidence is below threshold, DROP the suspect script from ratios.
                                    if c >= conf_thr:
                                        continue
                                    if s0 in lr and float(lr.get(s0, 0.0)) > 0.0:
                                        lr.pop(s0, None)
                                        removed.append(s0)
                                if removed:
                                    tot = sum(float(v) for v in lr.values())
                                    if tot > 0:
                                        lr = {k: float(v) / tot for k, v in lr.items()}
                                    cleaned["language_ratios"] = lr
                                    # Re-score with cleaned ratios
                                    ev2 = evaluate_compliance(
                                        cleaned,
                                        rules,
                                        blk_d,
                                        dict(verified_text_json) if isinstance(verified_text_json, dict) else None,
                                        geo_d.get("detected_graphics") if isinstance(geo_d, dict) else None,
                                    )
                                    if isinstance(ev2, dict):
                                        d2 = ev2.get("details") if isinstance(ev2.get("details"), dict) else None
                                        if isinstance(d2, dict):
                                            d2["gemini_script_override"] = {
                                                "confidence_min": conf_thr,
                                                "removed_scripts": removed,
                                                "gemini_confidences": {k: round(float(v), 3) for k, v in gconf.items()},
                                            }
                                            ev2["details"] = d2
                                    ev = ev2
    except Exception as e:
        logging.getLogger(__name__).warning("Gemini script override failed: %s", e, exc_info=True)

    return _compliance_info_from_eval(ev)


def _merge_compliance_with_governance(
    comp: ComplianceInfo,
    state_in: Optional[str],
    geo: Optional[SignboardGeometrySummary],
    blocks: list[TextBlock],
    layout_hints: Optional[dict[str, Any]] = None,
    full_text: Optional[str] = None,
    category: Optional[str] = None,
) -> ComplianceInfo:
    if not state_in or not (state_in or "").strip():
        return comp
    try:
        msg, st, gv = merge_governance_compliance(
            comp.message,
            comp.status,
            comp.ruleset_id,
            state_in,
            geo,
            blocks,
            layout_hints,
        )
        out = comp.model_copy(update={"message": msg, "status": st, "governance": gv})
        try:
            ov = evaluate_category_overlays(full_text or "", category or "", state_in or "")
            if ov:
                gov = dict(out.governance) if isinstance(out.governance, dict) else {}
                gov["category_overlays"] = ov
                # Surface category overlay checks in the same automated list the UI already renders.
                auto = list(gov.get("automated") or [])
                for c in (ov.get("checks") or []):
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("id") or "category_check")
                    ok = bool(c.get("ok"))
                    lab = str(c.get("label") or cid)
                    mt = c.get("match")
                    if ok:
                        detail = f"{lab}: present" + (f" ({mt})" if isinstance(mt, str) and mt else "")
                    else:
                        detail = f"{lab}: missing"
                    auto.append({"id": f"category_{cid}", "ok": ok, "detail": detail})
                gov["automated"] = auto
                if str(ov.get("overall") or "").lower() == "fail":
                    out = out.model_copy(
                        update={
                            "message": f"{out.message} · Category statutory data missing (see automated checks).",
                            "status": "fail" if out.status != "pending" else out.status,
                        }
                    )
                out = out.model_copy(update={"governance": gov})
        except Exception as e:
            logging.getLogger(__name__).warning("category overlays: %s", e, exc_info=True)
        return out
    except Exception as e:
        logging.getLogger(__name__).warning("Governance merge failed: %s", e, exc_info=True)
        return comp


def _blocks_from_stored_ocr(raw: dict[str, Any]) -> list[TextBlock]:
    """Rebuild TextBlock list from persisted JSON for governance checks on audit update."""
    raw_list = raw.get("blocks")
    if not isinstance(raw_list, list):
        return []
    out: list[TextBlock] = []
    for item in raw_list:
        if isinstance(item, dict):
            try:
                out.append(TextBlock.model_validate(item))
            except Exception:
                continue
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/compliance/states")
def compliance_states() -> dict[str, Any]:
    return {"states": list_state_rules()}


@app.get("/api/geolookup")
async def geolookup(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    try:
        r = await reverse_geocode(latitude, longitude)
        r.pop("raw", None)  # large; optional for response
        return r
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Geocoding failed: {e!s}"
        ) from e


@app.post("/api/translate", response_model=TranslateResponse)
def api_translate(req: TranslateRequest) -> TranslateResponse:
    if not req.texts:
        return TranslateResponse(translations=[])
    if req.languages is not None and len(req.languages) != len(req.texts):
        raise HTTPException(
            status_code=400,
            detail="languages must be the same length as texts when provided",
        )
    if len(req.texts) > 200:
        raise HTTPException(
            status_code=400, detail="Too many segments (max 200 per request).",
        )
    trs = translate_texts_on_demand(req.texts, req.languages)
    if req.texts and all(t is None for t in trs):
        hint = get_last_gemini_error() or (
            "Translation failed. Set GEMINI_API_KEY in backend/.env, restart the API, and ensure "
            "the Gemini API is enabled for the key. Try GEMINI_MODEL=gemini-flash-lite-latest."
        )
        return TranslateResponse(translations=trs, detail=hint)
    return TranslateResponse(translations=trs, detail=None)


@app.post("/api/suggest-corners", response_model=SuggestCornersResult)
async def suggest_corners(
    file: UploadFile = File(...),
    state_code: str = Form(""),
) -> SuggestCornersResult:
    """
    One Vision pass on the image, then `SignboardGeometryAgent.auto_corners` (filtered
    block boxes + minimum-area rotated rectangle, with hull fallback) for an initial
    quadrilateral. Does not store uploads;
    for PDFs use a raster first.
    """
    if _client is None:
        raise HTTPException(status_code=503, detail="Vision client is not initialized.")
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {e}") from e
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    declared = (file.content_type or "").split(";", 1)[0].strip().lower()
    if not _upload_allowed(declared, file.filename, image_bytes):
        raise HTTPException(
            status_code=415,
            detail="Could not use this file for corner detection. Use a supported image type.",
        )
    if _file_is_pdf(declared, file.filename, image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Corner helper works on raster images. Open the PDF in an image, or export a page as PNG first.",
        )

    # state_code accepted for API compatibility; Vision uses script auto-detect (no language hints).

    vision_buf = image_bytes
    if should_transcode_to_png_for_vision(image_bytes, declared, file.filename):
        try:
            vision_buf = transcode_to_png(image_bytes)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=415, detail=f"Could not transcode this image: {e!s}"
            ) from e

    nparr = np.frombuffer(vision_buf, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes.")
    bgr = bgr_ensure_min_long_edge(bgr, 1024)
    enc_corner, enc_corner_b = cv2.imencode(".png", bgr)
    if enc_corner:
        vision_buf = enc_corner_b.tobytes()
    img_h0, img_w0 = bgr.shape[0], bgr.shape[1]

    def _vcall(buf: bytes):
        return _client.annotate_image(
            request={
                "image": vision.Image(content=buf),
                "image_context": vision.ImageContext(),
                "features": [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
            }
        )

    try:
        response = _vcall(vision_buf)
        if response.error.message:
            em = response.error.message.lower()
            if "unsupported type" in em and vision_buf is image_bytes:
                vision_buf = transcode_to_png(image_bytes)
                bgr2 = cv2.imdecode(np.frombuffer(vision_buf, np.uint8), cv2.IMREAD_COLOR)
                if bgr2 is not None:
                    bgr2 = bgr_ensure_min_long_edge(bgr2, 1024)
                    e2, e2b = cv2.imencode(".png", bgr2)
                    if e2:
                        vision_buf = e2b.tobytes()
                    img_h0, img_w0 = bgr2.shape[0], bgr2.shape[1]
                response = _vcall(vision_buf)
    except (Unauthenticated, PermissionDenied) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Google Vision authentication failed: {e.message}",
        ) from e
    except GoogleAPICallError as e:
        raise HTTPException(status_code=502, detail=f"Vision API call error: {e.message}") from e

    if response.error.message:
        raise HTTPException(
            status_code=502, detail=f"Vision API error: {response.error.message}",
        )

    pages = list(response.full_text_annotation.pages) if response.full_text_annotation else []
    img_w = int(pages[0].width) if pages else img_w0
    img_h = int(pages[0].height) if pages else img_h0

    method = "full_image"
    quad: list[list[float]] = [
        [0.0, 0.0],
        [float(max(0, img_w - 1)), 0.0],
        [float(max(0, img_w - 1)), float(max(0, img_h - 1))],
        [0.0, float(max(0, img_h - 1))],
    ]

    auto = SignboardGeometryAgent.auto_corners_hybrid(bgr, response)
    if auto is not None and auto.shape[0] == 4:
        text_only = SignboardGeometryAgent.auto_corners(response)
        img_only = SignboardGeometryAgent.auto_corners_from_image(bgr)
        if img_only is not None and text_only is not None:
            method = "hybrid_text+edges"
        elif img_only is not None:
            method = "image_edges"
        else:
            method = "text_hull"
        quad = [[float(p[0]), float(p[1])] for p in auto]

    return SuggestCornersResult(
        corners=quad,
        method=method,
        image_width=img_w,
        image_height=img_h,
    )


@app.put("/api/audit/{audit_id}", response_model=OcrResult)
async def update_audit(
    audit_id: str,
    body: AuditUpdate,
    db: Session = Depends(get_db),
) -> OcrResult:
    try:
        uid = uuid.UUID(audit_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid audit_id") from e
    row = db.get(AuditRecord, uid)
    if not row:
        raise HTTPException(status_code=404, detail="Audit not found")

    raw: dict = dict(row.raw_ocr_json) if row.raw_ocr_json else {}
    # Optional category update (statutory overlays)
    if body.category is not None:
        c0 = (body.category or "").strip()
        raw["category"] = c0 or None
    if body.verified_text_json is not None:
        row.verified_text_json = body.verified_text_json
    u = (body.user_id or "").strip() or None
    if u is not None:
        row.user_id = u
    st_in = (body.state_code or "").strip().upper() or None
    ld0 = row.location_data if isinstance(row.location_data, dict) else {}
    if not st_in and body.latitude is not None and body.longitude is not None:
        try:
            geo = await reverse_geocode(float(body.latitude), float(body.longitude))
            st_in = (geo or {}).get("state_code")
        except Exception as e:
            logging.getLogger(__name__).warning("update_audit geocode: %s", e)
    if not st_in:
        st0 = (ld0.get("state_code") or "").strip()
        st_in = st0.upper() or None
    loc: dict = {**ld0, **({} if st_in is None else {"state_code": st_in})}
    if body.city:
        loc["city"] = body.city.strip()
    if body.latitude is not None and body.longitude is not None:
        loc["latitude"] = body.latitude
        loc["longitude"] = body.longitude
        loc["geocoded"] = True
    if st_in and not (body.latitude is not None and body.longitude is not None):
        loc.setdefault("source", "manual")
    row.location_data = loc
    st_final = (loc or {}).get("state_code")
    if isinstance(st_final, str):
        st_final = st_final.strip().upper() or None
    rules = get_rules_for_state(st_final) if st_final else None
    geom_data = raw.get("signboard_geometry")
    is_pdf = bool(raw.get("is_pdf_ocr", False))
    if geom_data and not isinstance(geom_data, dict):
        geom_data = None
    block_dicts: list[dict] = [dict(b) for b in (raw.get("blocks") or []) if isinstance(b, dict)]
    geom_for_eval = geom_data
    if (
        row.verified_text_json
        and isinstance(geom_data, dict)
        and block_dicts
    ):
        adj = recompute_geometry_after_verified_overrides(
            geom_data, block_dicts, dict(row.verified_text_json)
        )
        if adj is not None:
            geom_for_eval = adj
    geo: Optional[SignboardGeometrySummary] = None
    if isinstance(geom_for_eval, dict):
        try:
            geo = SignboardGeometrySummary.model_validate(geom_for_eval)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "update_audit: invalid signboard_geometry in stored JSON, skipping: %s", e
            )
            geo = None
    merged_block_dicts = block_dicts_with_verified_text(
        block_dicts, dict(row.verified_text_json) if row.verified_text_json else None
    )
    blocks_gov: list[TextBlock] = []
    for b in merged_block_dicts:
        try:
            blocks_gov.append(TextBlock.model_validate(b))
        except Exception:
            continue
    comp0 = _location_compliance(
        geo, st_final, rules, is_pdf, blocks_gov, dict(row.verified_text_json) if row.verified_text_json else None
    )
    comp = _merge_compliance_with_governance(
        comp0, st_final, geo, blocks_gov, None, str(raw.get("text") or ""), str(raw.get("category") or "")
    )
    row.compliance_status = _compliance_to_db_code(comp)
    for _k, _v in threshed_language_audit_columns(geo, st_final).items():
        setattr(row, _k, _v)
    raw["compliance"] = comp.model_dump()
    raw["is_pdf_ocr"] = is_pdf
    if row.verified_text_json is not None:
        raw["verified_text_json"] = row.verified_text_json
    if (
        row.verified_text_json
        and isinstance(geom_for_eval, dict)
        and geom_for_eval is not None
    ):
        raw["signboard_geometry"] = geom_for_eval
    if row.verified_text_json and merged_block_dicts:
        raw["blocks"] = merged_block_dicts
    row.raw_ocr_json = raw
    db.add(row)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Save failed: {e!s}") from e
    merged: dict = {
        **raw,
        "audit_id": str(row.id),
        "resolved_location": loc,
        "compliance": comp.model_dump(),
        "verified_text_json": row.verified_text_json,
    }
    return OcrResult.model_validate(merged)


@app.post("/api/ocr", response_model=OcrResult)
async def ocr(
    file: UploadFile = File(...),
    user_id: str = Form(""),
    state_code: str = Form(""),
    latitude: str = Form(""),
    longitude: str = Form(""),
    category: str = Form(""),
    db: Session = Depends(get_db),
) -> OcrResult:
    """
    Google Vision does **not** use language hints: script is auto-detected for all of India.
    **state_code** (e.g. ``KA``, ``MH``, ``DL``) selects the COMPLai threshed-ink rule row only;
    if omitted, ``compliance.status`` stays **pending** until a state is supplied.
    """
    if _client is None:
        raise HTTPException(status_code=503, detail="Vision client is not initialized.")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {e}")
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Single normalized MIME (handles charset etc.) — must accept image/avif, image/jxl, …
    declared = (file.content_type or "").split(";", 1)[0].strip().lower()
    if not _upload_allowed(declared, file.filename, image_bytes):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Could not use this file as an image or PDF (declared MIME: {declared or 'unknown'}). "
                "Supported: any image/* type (incl. AVIF, HEIC, WebP, TIFF), application/pdf, "
                "or octet-stream with a normal image/PDF filename or known magic bytes."
            ),
        )
    # Reject very blurry rasters before storage / Vision; PDFs are not checked here.
    pre_buf: bytes = image_bytes
    if should_transcode_to_png_for_vision(image_bytes, declared, file.filename):
        try:
            pre_buf = transcode_to_png(image_bytes)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=415, detail=f"Could not prepare image for blur check: {e!s}"
            ) from e
    if not _file_is_pdf(declared, file.filename, pre_buf):
        npr0 = np.frombuffer(pre_buf, np.uint8)
        bgr_probe = cv2.imdecode(npr0, cv2.IMREAD_COLOR)
        if bgr_probe is not None:
            bgr_probe = bgr_ensure_min_long_edge(bgr_probe, 1024)
            too_bl, score, mmin = is_too_blurry_bgr(bgr_probe)
            if too_bl:
                log = logging.getLogger(__name__)
                log.info(
                    "ocr reject blurry image: laplacian_var=%.1f (min=%.1f)", score, mmin
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "This image is too blurry for a reliable scan. Please take a new photo "
                        "with the sign in focus and well lit, then upload it again. "
                        "No OCR result is returned until a clearer image is provided."
                    ),
                )

    try:
        image_url = save_image_bytes(
            image_bytes, filename=file.filename, content_type=declared
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {e!s}") from e

    lat_s = (latitude or "").strip()
    lon_s = (longitude or "").strip()
    lat_p: Optional[float] = None
    lon_p: Optional[float] = None
    if lat_s and lon_s:
        try:
            lat_p = float(lat_s)
            lon_p = float(lon_s)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="latitude and longitude must be valid numbers if provided."
            ) from None

    state_in = (state_code or "").strip().upper() or None
    cat_in = (category or "").strip() or None
    if state_in and state_in.startswith("IN-"):
        state_in = state_in[3:]
    resolved: Optional[dict] = None
    if state_in is None and lat_p is not None and lon_p is not None:
        try:
            resolved = await reverse_geocode(lat_p, lon_p)
            st = (resolved or {}).get("state_code")
            if st:
                state_in = st
        except Exception as e:
            logging.getLogger(__name__).warning("reverse_geocode: %s", e)
            resolved = None

    out_resolved: Optional[dict] = None
    if state_in or lat_p is not None or resolved is not None:
        out_resolved = {
            "state_code": state_in,
            "city": (resolved or {}).get("city") if resolved else None,
            "state_name": (resolved or {}).get("state_name") if resolved else None,
            "latitude": lat_p,
            "longitude": lon_p,
        }
    location_data = None
    if out_resolved is not None:
        location_data = {k: v for k, v in out_resolved.items() if v is not None}
    uid_s = (user_id or "").strip() or None
    rules0 = get_rules_for_state(state_in) if state_in else None

    is_pdf_ocr: bool = False  # set after successful Vision read
    # No language_hints: Vision auto-detects script; state is for COMPLai rules only (universal across India).

    def _vision_call(buf: bytes):
        return _client.annotate_image(
            request={
                "image": vision.Image(content=buf),
                "image_context": vision.ImageContext(),
                "features": [
                    vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                    vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
                ],
            }
        )

    try:
        # AVIF/HEIC/JXL are not accepted by Vision; decode to PNG. If Google still
        # says "Unsupported type" for another codec, transcode and retry once.
        vision_buf = image_bytes
        if should_transcode_to_png_for_vision(image_bytes, declared, file.filename):
            vision_buf = transcode_to_png(image_bytes)
        # Light-aware CLAHE + HSV “pop” for signboards (see vision_preprocess)
        if not _file_is_pdf(declared, file.filename, vision_buf):
            npr = np.frombuffer(vision_buf, np.uint8)
            bgr0 = cv2.imdecode(npr, cv2.IMREAD_COLOR)
            if bgr0 is not None:
                bgr0 = bgr_ensure_min_long_edge(bgr0, 1024)
                try:
                    bgr1 = prepare_bgr_for_document_ocr(bgr0)
                    enc_ok, enc1 = cv2.imencode(".png", bgr1)
                    if enc_ok:
                        vision_buf = enc1.tobytes()
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "prepare_bgr_for_document_ocr: %s", e, exc_info=True
                    )
        if should_prebinarize_for_vision_ocr() and not _file_is_pdf(
            declared, file.filename, vision_buf
        ):
            npr0 = np.frombuffer(vision_buf, np.uint8)
            bgr0 = cv2.imdecode(npr0, cv2.IMREAD_COLOR)
            if bgr0 is not None:
                bpr0 = prebinarize_bgr_smart(bgr0)
                enc_ok, enc0 = cv2.imencode(".png", bpr0)
                if enc_ok:
                    vision_buf = enc0.tobytes()
        response = _vision_call(vision_buf)
        if response.error.message:
            em = response.error.message.lower()
            if "unsupported type" in em and vision_buf is image_bytes:
                vision_buf = transcode_to_png(image_bytes)
                response = _vision_call(vision_buf)

        if response.error.message:
            raise HTTPException(
                status_code=502,
                detail=f"Vision API error: {response.error.message}",
            )

        # ── Full text ────────────────────────────────────────────────────────
        full_text: str = response.full_text_annotation.text or ""

        # ── Image dimensions (from first page) ───────────────────────────────
        pages = list(response.full_text_annotation.pages)
        img_w: int = pages[0].width if pages else 0
        img_h: int = pages[0].height if pages else 0

        # ── Page-level languages ─────────────────────────────────────────────
        languages: list[LanguageHint] = []
        if pages:
            languages = _get_languages(pages[0].property)

        # ── Annotated file bytes = coordinate space of Vision (may be transcoded AVIF→PNG)
        annotated_image_bytes = vision_buf
        is_pdf_ocr = _file_is_pdf(declared, file.filename, vision_buf)

        # Threshed-pixel area % + language ratios (SignboardGeometryAgent), raster images only
        page_block_list: list = list(pages[0].blocks) if pages else []
        pcts_all: list[float] = [0.0] * len(page_block_list) if page_block_list else []
        geo_sum: Optional[SignboardGeometrySummary] = None
        detected_graphics: list[DetectedGraphic] = []
        if pages and (not is_pdf_ocr) and _agent is not None:
            nparr2 = np.frombuffer(annotated_image_bytes, np.uint8)
            bgr2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
            if bgr2 is not None:
                try:
                    gres = _agent.analyze(bgr2, response)
                    det = {k: asdict(v) for k, v in gres.language_details.items()}
                    pcts_all = gres.per_block_threshed_board_pct
                    if len(pcts_all) != len(page_block_list):
                        pcts_all = _agent.per_block_threshed_pcts_including_silent(
                            page_block_list
                        )
                    geo_sum = SignboardGeometrySummary(
                        language_ratios=gres.language_ratios,
                        language_details=det,
                        compliance_status=gres.compliance_status,
                        confidence_score=gres.confidence_score,
                        dominant_language=gres.dominant_language,
                        primary_language=gres.primary_language,
                        primary_ratio=gres.primary_ratio,
                        monolingual_single_lang=gres.monolingual_single_lang,
                        per_block_threshed_board_pct=pcts_all or None,
                        detected_graphics=gres.detected_graphics or None,
                    )
                    detected_graphics = [DetectedGraphic.model_validate(g) for g in (gres.detected_graphics or [])]
                except Exception as exc:
                    logging.getLogger(__name__).warning("SignboardGeometryAgent: %s", exc)
                    pcts_all = [0.0] * len(page_block_list) if page_block_list else []

        # Vision block indices to return: has OCR text, or threshed ink (silent) above floor
        include_idx: list[int] = []
        for i, block in enumerate(page_block_list):
            t0 = _block_text(block)
            th = pcts_all[i] if i < len(pcts_all) else 0.0
            if t0.strip():
                include_idx.append(i)
            elif th > float(SILENT_INK_PCT_FLOOR):
                include_idx.append(i)
        total_ink_w = sum(
            (pcts_all[i] if i < len(pcts_all) else 0.0) / 100.0 for i in include_idx
        )
        if total_ink_w <= 0.0:
            total_ink_w = 1.0

        # Blocks with per-word confidences, bbox + threshed area %, Gemini English where needed
        blocks: list[TextBlock] = []
        to_tr: list[tuple[str, str]] = []
        tr_idx: list[int] = []
        for i in include_idx:
            block = page_block_list[i]
            t = _block_text(block)
            b = _poly_to_bounds(block.bounding_box)
            langs = _get_languages(block.property)
            norm0 = _to_norm(b, img_w, img_h)
            th = pcts_all[i] if i < len(pcts_all) else 0.0
            w_share = ((th / 100.0) / total_ink_w) if total_ink_w > 0 else 0.0
            th_round = round(th, 2) if th is not None else None
            if norm0 is not None:
                norm = norm0.model_copy(
                    update={"threshed_area_pct": th_round if th_round is not None else None}
                )
            else:
                norm = None
            has_kn = _vision_block_has_kannada(block)
            wrows = extract_verified_words_from_block(block)
            wlist = [VerifiedWord(text=wt, confidence=wc, is_doubtful=wd) for wt, wc, wd in wrows]
            is_silent = (not t.strip() and th > float(SILENT_INK_PCT_FLOOR)) or (
                th > float(SILENT_INK_PCT_STRONG) and not has_kn
            )
            if is_silent and not t.strip():
                wlist = []
            pl = langs[0].code if langs else "und"
            if is_silent:
                block_doubt = True
                wtype: Optional[str] = "SILENT_REGION_DETECTED"
                wmsg: Optional[str] = (
                    "Visual markings detected, but AI failed to read text. Please manually verify."
                )
            else:
                block_doubt = False
                wtype, wmsg = None, None
            bi = len(blocks)
            blocks.append(
                TextBlock(
                    text=t,
                    bounds=b,
                    norm=norm,
                    languages=langs,
                    words=wlist,
                    translated_text=None,
                    is_doubtful=block_doubt,
                    warning_type=wtype,
                    warning_message=wmsg,
                    threshed_ink_share=round(w_share, 5),
                )
            )
            if t.strip() and should_translate_block(pl) and (not is_silent or wlist):
                to_tr.append((pl, t))
                tr_idx.append(bi)

        if to_tr:
            trs = translate_blocks_to_english(to_tr)
            for j, bi in enumerate(tr_idx):
                if j < len(trs) and trs[j]:
                    blocks[bi] = blocks[bi].model_copy(update={"translated_text": trs[j]})

        full_translated = "\n".join(
            (blk.translated_text or blk.text) for blk in blocks
        ) if blocks else ""

        # ── Logos ────────────────────────────────────────────────────────────
        logos: list[LogoResult] = []
        for logo in response.logo_annotations:
            b = _poly_to_bounds(logo.bounding_poly)
            logos.append(LogoResult(
                name=logo.description,
                confidence=round(logo.score, 3),
                bounds=b,
                norm=_to_norm(b, img_w, img_h),
            ))

    except (Unauthenticated, PermissionDenied) as e:
        raise HTTPException(
            status_code=502,
            detail=(
                "Google Vision authentication failed. "
                "Check GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY in backend/.env. "
                f"Detail: {e.message}"
            ),
        )
    except GoogleAPICallError as e:
        raise HTTPException(status_code=502, detail=f"Vision API call error: {e.message}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR failed: {e!s}")

    gstin_raw = enrich_gstins(full_text, blocks)
    gstin_out = [GstinFinding(**g) for g in gstin_raw]

    layout_hints: dict[str, Any] | None = None
    if (state_in or "") == "KA" and pages and img_w > 0 and img_h > 0:
        wk = ka_ocr_word_upper_layout_hint(response, img_w, img_h)
        if wk is not None:
            layout_hints = {"ka_ocr_word_upper": wk}

    comp0 = _location_compliance(geo_sum, state_in, rules0, is_pdf_ocr, blocks, None)
    compliance = _merge_compliance_with_governance(
        comp0, state_in, geo_sum, blocks, layout_hints, full_text, cat_in
    )
    img_meta = build_image_metadata(
        data=image_bytes,
        filename=file.filename,
        declared_mime=declared,
        is_pdf=is_pdf_ocr,
        vision_width=img_w,
        vision_height=img_h,
    )
    out = OcrResult(
        text=full_text,
        image_width=img_w,
        image_height=img_h,
        languages=languages,
        blocks=blocks,
        logos=logos,
        gstin=gstin_out,
        translated_text=full_translated,
        signboard_geometry=geo_sum,
        detected_graphics=detected_graphics,
        compliance=compliance,
        is_pdf_ocr=is_pdf_ocr,
        resolved_location=out_resolved,
        image_metadata=img_meta,
        category=cat_in,
    )
    payload: dict = out.model_dump(mode="json")
    ar: Optional[AuditRecord] = None
    try:
        ar = AuditRecord(
            user_id=uid_s,
            image_url=image_url,
            raw_ocr_json=payload,
            location_data=location_data,
            verified_text_json=None,
            compliance_status=_compliance_to_db_code(compliance),
            **threshed_language_audit_columns(geo_sum, state_in),
        )
        db.add(ar)
        db.commit()
        db.refresh(ar)
    except Exception as e:
        try:
            db.rollback()
        except Exception:  # noqa: S110
            pass
        logging.getLogger(__name__).warning("Could not save audit record: %s", e)
    if ar is not None:
        return out.model_copy(
            update={"audit_id": str(ar.id), "resolved_location": out_resolved}
        )
    return out
