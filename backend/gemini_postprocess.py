"""
Optional Gemini (Flash) batch translation of non-English text blocks to English.
Uses GEMINI_API_KEY or GOOGLE_API_KEY (AI Studio / Generative Language API key).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

_log = logging.getLogger(__name__)

# Set on failed translate so /api/translate can return a useful message (not just "set API key")
_last_gemini_error: str | None = None


def get_last_gemini_error() -> str | None:
    return _last_gemini_error


def _get_api_key() -> str | None:
    k = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return k or None


def _parse_json_loose(text: str) -> dict:
    t = (text or "").strip()
    if "```" in t:
        t = re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    if m:
        return json.loads(m.group(0))
    return json.loads(t)


def translate_blocks_to_english(
    block_specs: list[tuple[str, str]],  # (bcp47_lang, text)
) -> list[Optional[str]]:
    """
    Returns one English string per input block, or None on failure / no API key.
    """
    global _last_gemini_error
    _last_gemini_error = None

    if not block_specs:
        return []
    key = _get_api_key()
    if not key:
        _log.debug("No GEMINI_API_KEY/GOOGLE_API_KEY — skipping translation")
        _last_gemini_error = (
            "No API key: set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env and restart the server."
        )
        return [None] * len(block_specs)

    try:
        import google.generativeai as genai
    except ImportError as e:
        _log.warning("google-generativeai not installed: %s", e)
        _last_gemini_error = "Install the google-generativeai package: pip install google-generativeai"
        return [None] * len(block_specs)

    # Model selection:
    # - `GEMINI_MODEL` overrides everything
    # - otherwise try a small set of current Flash models (some older ids 404 for new users)
    override = (os.getenv("GEMINI_MODEL") or "").strip()
    # NOTE: The deprecated `google.generativeai` client expects model ids without the leading
    # "models/" prefix; it will call the API with "models/<id>".
    # These defaults are chosen to be broadly available as of 2026 (AI Studio keys),
    # with preview ids as fallbacks.
    candidates = (
        [override]
        if override
        else [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
        ]
    )
    genai.configure(api_key=key)

    arr = [
        {"i": i, "lang": lang, "text": text}
        for i, (lang, text) in enumerate(block_specs)
    ]
    prompt = (
        "You are a professional translator. Given the JSON array below, translate each item's "
        '"text" field to clear English. Preserve proper names, brand names, numbers, addresses, '
        "and GSTINs exactly. Do not add commentary.\n"
        "Return ONLY a JSON object of the form: "
        '{"translations": ["…", "…", …] } with the same array length and order as input (index i in order 0,1,2,…).'
        f"\n\nINPUT:\n{json.dumps(arr, ensure_ascii=False)}"
    )
    last_exc: Exception | None = None
    tried: list[str] = []
    for model_name in [c for c in candidates if c]:
        tried.append(model_name)
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 8192},
            )
            raw = (resp.text or "").strip()
            data = _parse_json_loose(raw)
            trs: list = data.get("translations", [])
            if not isinstance(trs, list) or len(trs) != len(block_specs):
                _log.warning(
                    "Gemini translation length mismatch: model=%s want %d got %r",
                    model_name,
                    len(block_specs),
                    trs,
                )
                if isinstance(trs, list) and trs:
                    out: list[Optional[str]] = []
                    for i in range(len(block_specs)):
                        out.append(str(trs[i]) if i < len(trs) else None)
                    return out
                _last_gemini_error = (
                    "Gemini returned JSON without a valid translations list; "
                    "try setting GEMINI_MODEL to a different model id."
                )
                return [None] * len(block_specs)
            return [str(x) if x is not None else None for x in trs]
        except Exception as e:
            last_exc = e
            # If the model is unavailable (common 404), try the next candidate.
            msg = f"{type(e).__name__}: {e!s}"
            _log.warning("Gemini model failed (%s): %s", model_name, msg)
            continue

    tried_s = ", ".join(tried[:8]) + ("…" if len(tried) > 8 else "")
    if last_exc is not None:
        _last_gemini_error = f"{type(last_exc).__name__}: {last_exc!s} (tried: {tried_s})"[:2000]
    else:
        _last_gemini_error = f"Gemini translation failed (tried: {tried_s})"
    return [None] * len(block_specs)


def translate_texts_on_demand(
    texts: list[str],
    languages: list[str] | None = None,
) -> list[Optional[str]]:
    """
    Translate each line to English (for UI on-demand / when OCR-time translation was skipped).
    `languages` is optional, same length as `texts` (per-block BCP-47 from Vision); else `und` is used.
    """
    if not texts:
        return []
    if languages and len(languages) == len(texts):
        specs: list[tuple[str, str]] = [
            ((languages[i] or "und").strip() or "und", t) for i, t in enumerate(texts)
        ]
    else:
        specs = [("und", t) for t in texts]
    return translate_blocks_to_english(specs)


def should_translate_block(lang_code: str) -> bool:
    base = (lang_code or "und").lower().split("-", 1)[0]
    return base not in ("en", "und", "")


def gemini_detect_script_confidences(
    texts: list[str],
    *,
    max_items: int = 24,
    max_chars_each: int = 280,
) -> dict[str, float]:
    """
    Return {bcp47_base: confidence_0_1} for scripts/languages present in OCR text.
    Used only as a tie-breaker / misfire filter; not a replacement for threshed ink ratios.
    """
    global _last_gemini_error
    _last_gemini_error = None

    key = _get_api_key()
    if not key:
        return {}
    try:
        import google.generativeai as genai
    except ImportError as e:
        _log.warning("google-generativeai not installed: %s", e)
        _last_gemini_error = "Install the google-generativeai package: pip install google-generativeai"
        return {}

    override = (os.getenv("GEMINI_MODEL") or "").strip()
    model_name = override or "gemini-flash-lite-latest"
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)

    # Keep prompt small and robust for mobile/edge cases.
    clean: list[str] = []
    for t in texts:
        s = (t or "").strip()
        if not s:
            continue
        s = s.replace("\u0000", "")
        if len(s) > max_chars_each:
            s = s[: max_chars_each - 1] + "…"
        clean.append(s)
        if len(clean) >= max_items:
            break
    if not clean:
        return {}

    prompt = (
        "You are a language/script detector for OCR outputs.\n"
        "Given the TEXT SNIPPETS below (which may be noisy OCR), identify which writing scripts/languages "
        "are actually present. Return ONLY JSON of the form:\n"
        '{"scripts":[{"code":"kn","confidence":0.0},{"code":"en","confidence":0.0}]}\n'
        "- `code` must be the BCP-47 base code like en, hi, kn, ta, te, ml, mr, gu, pa, bn, ur, or `und`.\n"
        "- `confidence` must be a number 0..1.\n"
        "- Only include codes you believe are present (do not include everything).\n\n"
        f"TEXT SNIPPETS:\n{json.dumps(clean, ensure_ascii=False)}"
    )
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 1024})
        raw = (resp.text or "").strip()
        data = _parse_json_loose(raw)
        arr = data.get("scripts", [])
        out: dict[str, float] = {}
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                code = (it.get("code") or "").strip().lower()
                if not code:
                    continue
                base = code.split("-", 1)[0]
                try:
                    conf = float(it.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                if conf < 0.0:
                    conf = 0.0
                if conf > 1.0:
                    conf = 1.0
                # Keep max confidence per code
                out[base] = max(out.get(base, 0.0), conf)
        return out
    except Exception as e:
        _log.exception("Gemini script detect failed: %s", e)
        _last_gemini_error = f"{type(e).__name__}: {e!s}"[:2000]
        return {}
