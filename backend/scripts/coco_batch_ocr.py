#!/usr/bin/env python3
# ruff: noqa: T201
"""
TEMP: Run Google Vision + same raster preprocessing as ``main:ocr`` on a Roboflow/COCO
zip (train / test / valid). Writes a JSON report + prints a short summary.

Usage (from ``backend/``)::

  python scripts/coco_batch_ocr.py --zip "C:\\path\\to\\dataset.v1i.coco.zip" --state KA

Requires ``.env`` with Google Vision credentials (same as the API server).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Run from project ``backend/`` as cwd so imports match the API.
BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(BACKEND / ".env")

from dataclasses import asdict as dc_asdict
from google.cloud import vision
from signboard_agent import SignboardGeometryAgent, SILENT_INK_PCT_FLOOR

from image_prep import should_transcode_to_png_for_vision, transcode_to_png
from vision_preprocess import (
    prepare_bgr_for_document_ocr,
    prebinarize_bgr_smart,
    should_prebinarize_for_vision_ocr,
)

# Defer open cv2? already imported

# ---------------------------------------------------------------------------
# Minimal mirrors of main.py helpers (avoid importing FastAPI app)


def _file_is_pdf(declared: str, filename: str | None, buf: bytes) -> bool:
    d = (declared or "").split(";", 1)[0].strip().lower()
    if d == "application/pdf":
        return True
    if (filename or "").lower().endswith(".pdf"):
        return True
    return len(buf) > 4 and buf[:4] == b"%PDF"


def build_vision_client() -> vision.ImageAnnotatorClient:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        from google.api_core.client_options import ClientOptions

        return vision.ImageAnnotatorClient(
            client_options=ClientOptions(api_key=api_key)
        )
    return vision.ImageAnnotatorClient()


def declared_mime_for_name(name: str) -> str:
    n = name.lower()
    if n.endswith((".jpg", ".jpeg", ".jpe", ".jif", ".jfi")):
        return "image/jpeg"
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def pipeline_to_vision_buf(
    image_bytes: bytes, filename: str, declared: str
) -> bytes:
    """Same as ``/api/ocr`` raster path up to the Vision request buffer."""
    vision_buf = image_bytes
    if should_transcode_to_png_for_vision(image_bytes, declared, filename):
        vision_buf = transcode_to_png(image_bytes)
    if not _file_is_pdf(declared, filename, vision_buf):
        npr = np.frombuffer(vision_buf, np.uint8)
        bgr0 = cv2.imdecode(npr, cv2.IMREAD_COLOR)
        if bgr0 is not None:
            bgr1 = prepare_bgr_for_document_ocr(bgr0)
            enc_ok, enc1 = cv2.imencode(".png", bgr1)
            if enc_ok:
                vision_buf = enc1.tobytes()
    if should_prebinarize_for_vision_ocr() and not _file_is_pdf(
        declared, filename, vision_buf
    ):
        npr0 = np.frombuffer(vision_buf, np.uint8)
        bgr0 = cv2.imdecode(npr0, cv2.IMREAD_COLOR)
        if bgr0 is not None:
            bpr0 = prebinarize_bgr_smart(bgr0)
            enc_ok, enc0 = cv2.imencode(".png", bpr0)
            if enc_ok:
                vision_buf = enc0.tobytes()
    return vision_buf


@dataclass
class OneResult:
    split: str
    relpath: str
    ok: bool
    error: str | None = None
    full_text_len: int = 0
    vision_blocks: int = 0
    returned_text_blocks: int = 0
    img_w: int = 0
    img_h: int = 0
    is_pdf: bool = False
    geometry_conf: float | None = None
    compliance_status: str | None = None
    dom_lang: str | None = None
    duration_s: float = 0.0
    has_silent_ink: bool = False
    # GSTIN (same helper as API)
    gstin_count: int = 0


def run_one_image(
    zf: zipfile.ZipFile,
    name: str,
    split: str,
    client: vision.ImageAnnotatorClient,
    agent: SignboardGeometryAgent | None,
    lang_hints: list[str],
    gstin_fn,
) -> OneResult:
    t0 = time.perf_counter()
    base = Path(name).name
    declared = declared_mime_for_name(base)
    raw = zf.read(name)
    out = OneResult(split=split, relpath=name, ok=False)
    if not raw:
        out.error = "empty"
        return out
    try:
        vision_buf = pipeline_to_vision_buf(raw, base, declared)
        is_pdf = _file_is_pdf(declared, base, vision_buf)
        out.is_pdf = is_pdf
        if is_pdf:
            out.error = "pdf_in_zip_unsupported_in_batch"
            return out
        def _vis(buf: bytes):
            return client.annotate_image(
                request={
                    "image": vision.Image(content=buf),
                    "image_context": vision.ImageContext(language_hints=lang_hints),
                    "features": [
                        vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                    ],
                }
            )
        response = _vis(vision_buf)
        if response.error.message:
            em = (response.error.message or "").lower()
            if "unsupported type" in em and vision_buf is raw:
                vision_buf = transcode_to_png(raw)
                response = _vis(vision_buf)
        if response.error.message:
            out.error = response.error.message
            return out
        fta = response.full_text_annotation
        full = (fta.text or "") if fta else ""
        out.full_text_len = len(full)
        pages = list(fta.pages) if fta else []
        out.img_w = pages[0].width if pages else 0
        out.img_h = pages[0].height if pages else 0
        page_block_list: list = list(pages[0].blocks) if pages else []
        out.vision_blocks = len(page_block_list)
        if not is_pdf and agent and pages and page_block_list:
            bgr2 = np.frombuffer(vision_buf, np.uint8)
            bgr2 = cv2.imdecode(bgr2, cv2.IMREAD_COLOR)
            if bgr2 is not None:
                gres = agent.analyze(bgr2, response)
                out.geometry_conf = float(gres.confidence_score)
                out.compliance_status = gres.compliance_status
                out.dom_lang = gres.dominant_language
                pcts = gres.per_block_threshed_board_pct
                if len(pcts) != len(page_block_list):
                    pcts = agent.per_block_threshed_pcts_including_silent(
                        page_block_list
                    )
                for i, blk in enumerate(page_block_list):
                    t0b = _block_text_light(blk)
                    th = pcts[i] if i < len(pcts) else 0.0
                    if t0b.strip() or th <= float(SILENT_INK_PCT_FLOOR):
                        out.returned_text_blocks += 1
                    if (not t0b.strip()) and th > float(SILENT_INK_PCT_FLOOR):
                        out.has_silent_ink = True
        else:
            out.returned_text_blocks = out.vision_blocks
        if gstin_fn and full:
            try:
                g = gstin_fn(full, [])
                out.gstin_count = len(g) if g else 0
            except Exception:  # noqa: BLE001
                out.gstin_count = 0
        out.ok = True
    except Exception as e:  # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"
    out.duration_s = time.perf_counter() - t0
    return out


def _block_text_light(block) -> str:
    lines: list[str] = []
    for para in block.paragraphs:
        words: list[str] = []
        for word in para.words:
            words.append("".join(s.text for s in word.symbols))
        if words:
            lines.append(" ".join(words))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch OCR (Vision) for COCO zip")
    ap.add_argument(
        "--zip",
        required=True,
        help="Path to merged-storeboard-*.v1i.coco.zip",
    )
    ap.add_argument(
        "--out",
        default="",
        help="JSON report path (default: backend/data/coco_batch_ocr_report.json)",
    )
    ap.add_argument(
        "--state",
        default="KA",
        help="BCP-47 / state for Vision language hints (e.g. KA for kn,en). Use '' for default hints.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (rate limit)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images per split (0 = all)",
    )
    args = ap.parse_args()
    zip_path = Path(args.zip)
    if not zip_path.is_file():
        print("ZIP not found:", zip_path)
        return 1
    out_path = Path(args.out) if args.out else BACKEND / "data" / "coco_batch_ocr_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = (args.state or "").strip().upper() or None
    if st and st.startswith("IN-"):
        st = st[3:]

    client = build_vision_client()
    agent = SignboardGeometryAgent(
        primary_language=os.getenv("PRIMARY_LANGUAGE", "kn"),
        compliance_threshold=float(os.getenv("COMPLIANCE_THRESHOLD", "0.60")),
        ink_tag_confidence=float(os.getenv("OCR_INK_TAG_CONFIDENCE", "0.85")),
    )
    # Match production /api/ocr: Vision auto-detects script; no language_hints.
    lang_hints: list[str] = []

    # Lazy import to avoid App init
    try:
        from gstin_extract import enrich_gstins
        _gstin = enrich_gstins
    except Exception:  # noqa: BLE001
        _gstin = None
    if _gstin is None:
        def _none_gstin(_a, _b):
            return []
    else:
        _none_gstin = _gstin

    splits: dict[str, list[str]] = {s: [] for s in ("train", "test", "valid", "val")}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name or name.endswith("/") or "annotations" in name.lower():
                continue
            parts = name.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            sp = parts[0].lower()
            if sp not in splits:
                continue
            if sp == "val":
                spk = "valid"
            else:
                spk = sp
            if not str(parts[-1]).lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            ):
                continue
            if spk not in splits:
                spk = sp
            splits.setdefault(spk, [])
            if spk in splits:
                splits[spk].append(name)

    # Normalize valid split key
    if not splits.get("valid") and splits.get("val"):
        splits["valid"] = list(splits["val"])

    per_split_planned: dict[str, int] = {}
    for sp in ("train", "test", "valid"):
        names0 = sorted(splits.get(sp, []))
        if args.limit and args.limit > 0:
            per_split_planned[sp] = min(int(args.limit), len(names0))
        else:
            per_split_planned[sp] = len(names0)

    t_wall = time.perf_counter()
    results: list[OneResult] = []
    with zipfile.ZipFile(zip_path) as zf:
        for sp in ("train", "test", "valid"):
            names = sorted(splits.get(sp, []))
            lim = args.limit if args.limit and args.limit > 0 else len(names)
            names = names[:lim]
            for name in names:
                r = run_one_image(
                    zf=zf,
                    name=name,
                    split=sp,
                    client=client,
                    agent=agent,
                    lang_hints=lang_hints,
                    gstin_fn=_none_gstin,
                )
                results.append(r)
                if args.delay > 0:
                    time.sleep(args.delay)
                if not r.ok:
                    print("FAIL", sp, Path(name).name, r.error)

    report = {
        "zip": str(zip_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_hint": st,
        "vision_language_hints": lang_hints,
        "per_split_planned": per_split_planned,
        "total_images_run": len(results),
        "wall_time_s": round(time.perf_counter() - t_wall, 1),
        "per_split": {},
        "results": [dc_asdict(r) for r in results],
    }

    for sp in ("train", "test", "valid"):
        sub = [r for r in results if r.split == sp]
        okc = sum(1 for r in sub if r.ok)
        ftc = [r for r in sub if not r.ok]
        texts = [r.full_text_len for r in sub if r.ok]
        gconf = [r.geometry_conf for r in sub if r.ok and r.geometry_conf is not None]
        durs = [r.duration_s for r in sub if r.ok]
        report["per_split"][sp] = {
            "total": len(sub),
            "ok": okc,
            "failed": len(ftc),
            "mean_full_text_len": round(sum(texts) / max(len(texts), 1), 1) if texts else 0.0,
            "mean_vision_blocks": round(
                sum(r.vision_blocks for r in sub if r.ok) / max(okc, 1), 1
            )
            if okc
            else 0.0,
            "mean_geometry_confidence": round(sum(gconf) / max(len(gconf), 1), 3)
            if gconf
            else None,
            "mean_vision_call_duration_s": round(
                sum(durs) / max(len(durs), 1), 2
            )
            if durs
            else None,
            "silent_ink_flags": sum(1 for r in sub if r.ok and r.has_silent_ink),
        }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", out_path)
    for sp, agg in report["per_split"].items():
        print(f"{sp}: {agg['ok']}/{agg['total']} ok, mean_text_len={agg['mean_full_text_len']}, "
              f"mean_vision_blocks={agg['mean_vision_blocks']}, mean_geo_conf={agg['mean_geometry_confidence']}")
    fails = [r for r in results if not r.ok]
    if fails:
        print("Failures:", len(fails))
        for r in fails[:15]:
            print(" ", r.relpath, r.error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
