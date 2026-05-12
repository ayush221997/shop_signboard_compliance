# OCR Pipeline — Limitations, Failure Modes, and Mitigations

This document describes what the COMPLai OCR **system** can and cannot do. The stack is not a single in-house trained model: **text and logos are recognized by Google Cloud Vision** (`DOCUMENT_TEXT_DETECTION`, `LOGO_DETECTION`). This service adds **OpenCV-based signboard geometry**, **optional Gemini** (translation and script tie-breaks), **rule-based GSTIN extraction**, and **state/category compliance** on top of that API.

---

## 1. Core OCR (Google Cloud Vision)

### Limitations

- **Vendor black box**: You cannot change Vision’s training data or decoding; behavior follows Google’s model updates and quotas.
- **No explicit language hints in the main flow**: Script/language is **auto-detected** for India-wide use. Unusual fonts, mixed scripts in one line, or decorative type can be mislabeled or split oddly.
- **Very small, faint, or low-contrast text** may be dropped or merged incorrectly, especially on noisy backgrounds (mesh, foliage, reflections).
- **Non-text graphics** (logos, icons, ornamental borders) may sometimes produce spurious characters or be ignored.
- **Handwriting and highly stylized lettering** are less reliable than printed shop signage.

### When it tends to fail

- Extreme glare, deep shadows, or **motion blur** (even if the Laplacian gate accepts the image).
- Strong **perspective** before correction, or text running along curves.
- **Occlusion** (poles, vehicles, people covering part of the board).

### Mitigations

- Capture **straight-on, well-lit** photos; use the app’s **corner + warp** flow so text is roughly fronto-parallel before OCR.
- **Retake** with higher resolution and focus; avoid digital zoom that adds blur.
- For persistent misreads, treat OCR as **assistive**: allow **human verification** of blocks (the API supports verified text / review flows where implemented).

---

## 2. Blur gate (raster images only)

### Behavior

- Before OCR, **raster** uploads can be **rejected** if Laplacian variance is below `OCR_BLUR_LAPLACIAN_VAR_MIN` (default **100**). See `backend/blur_check.py`.
- **PDF uploads are not blur-checked**; a soft or low-quality raster inside a PDF may still reach Vision.

### When it fails (product behavior)

- **False reject**: Very uniform boards (large flat color regions) can yield **low edge energy** and trip the blur detector even when text is readable.
- **False accept**: PDFs skip the gate; heavy motion blur in a photo might occasionally pass if above threshold.

### Mitigations

- **Operators**: Refocus and increase light; avoid handheld shake.
- **Admins**: Tune `OCR_BLUR_LAPLACIAN_VAR_MIN` or, only if necessary, set `OCR_DISABLE_BLUR_CHECK=1` (accepts weaker images but increases bad OCR risk).

---

## 3. File format and transcoding

### Limitations

- Vision expects common raster types (JPEG, PNG, WebP, etc.). **AVIF, HEIC, JXL** are **transcoded to PNG** in `backend/image_prep.py`; failure to decode yields **415**.
- Unsupported or corrupted files are rejected with **415**; empty uploads return **400**.

### Mitigations

- Upload **PNG or JPEG** from the device when possible.
- Ensure PDFs are valid; for corner detection, the API may require **raster exports** (see `/api/suggest-corners` constraints in `backend/main.py`).

---

## 4. Signboard geometry and compliance (OpenCV)

### Limitations

- **Perspective correction** depends on **accurate corners**. Auto-suggested corners can be wrong on cluttered scenes, partial boards, or very small text hulls (thresholds in `backend/signboard_agent.py`, e.g. minimum block area fraction for auto corners).
- **Threshed “ink” metrics** (adaptive threshold) can **over- or under-segment** ink on unusual materials (neon, metallic, gradient flex).
- **“Silent ink”** regions (Vision misses text but OpenCV sees ink) use floors like `SILENT_INK_PCT_FLOOR` / `SILENT_INK_PCT_STRONG` — false positives/negatives affect **language ratio** and review flags.
- **Compliance** needs a **`state_code`** for state rules; without it, compliance may stay **pending** (by design in the main OCR docstring).
- Geometry/signboard summary may be **limited or unavailable for PDFs** or when geometry cannot be computed (see API messages in `main.py`).

### Mitigations

- Use the UI to **adjust corners** manually before warp.
- Always send **`state_code`** when compliance is required.
- Use **Gemini script tie-break** (when configured) only as a **secondary** signal; disputed cases should go to **human review**.

---

## 5. Word confidence and “doubtful” flags

### Behavior

- Per-word confidence uses Vision symbol/word scores; **`is_doubtful`** when confidence **&lt; 0.85** (`backend/vision_verification.py`).

### Limitations

- High confidence does not guarantee semantic correctness (e.g. wrong word boundary).
- Missing symbol confidences fall back to defaults that may **under-flag** doubt.

### Mitigations

- Surface **doubtful** words in the UI and require confirmation for regulated fields.
- Re-capture or crop tighter on ambiguous strings.

---

## 6. GSTIN and structured fields

### Limitations

- GSTINs are found with a **regex + check-digit** validator (`backend/gstin_extract.py`). **OCR must produce a contiguous valid pattern**; split digits across lines, `O` vs `0`, or punctuation can prevent a match.
- Invalid or **syntactically correct but wrong** GSTINs are not verified against a government registry in this pipeline.

### Mitigations

- Manual correction when the UI shows near-misses.
- Optional downstream **GSTIN API verification** if the product requires it.

---

## 7. Gemini (translation and script assistance)

### Limitations

- **No API key** (`GEMINI_API_KEY` / `GOOGLE_API_KEY`): **no English translation** of blocks; `translated_text` may be empty or partial.
- **Model availability**: wrong or deprecated `GEMINI_MODEL` can cause failures; the code tries several Flash variants.
- **LLM behavior**: occasional **JSON shape errors**, length mismatches, or subtle **translation** mistakes; not deterministic across time.
- Script-disambiguation is a **heuristic tie-break**, not a legal determination.

### Mitigations

- Configure a valid key and `GEMINI_MODEL` (e.g. `gemini-flash-lite-latest` as noted in API errors).
- Install `google-generativeai` if translation is required.
- Never rely on Gemini alone for **compliance**; keep **human review** for borderline script mix cases.

---

## 8. Operations and infrastructure

### Failure modes

- **503**: Vision client not initialized (credentials / startup failure).
- **401 / permission errors**: Misconfigured `GOOGLE_APPLICATION_CREDENTIALS` or API key; Vision API not enabled; billing.
- **502**: Vision API errors, network issues, or quota exhaustion.
- **500**: Storage failures, unexpected server errors.

### Mitigations

- Verify **GCP project**, **Vision API** enabled, and credentials on the host.
- Monitor **quotas and latency**; add retries or backoff at the integration layer where appropriate.
- Ensure **local/S3 storage** paths and permissions are valid (`backend/storage.py`).

---

## 9. Quick reference — symptom → likely cause → action

| Symptom | Likely cause | Practical action |
|--------|----------------|------------------|
| HTTP 400 “too blurry” | Laplacian below threshold | Sharper photo; tune threshold or disable gate only if necessary |
| HTTP 415 | Unsupported or undecodable file | Convert to PNG/JPEG; fix corrupt PDF |
| HTTP 503 | Vision client not ready | Fix credentials and restart service |
| HTTP 502 from Vision | API/quota/network | Check GCP status, quotas, retries |
| Wrong script / garbled text | Vision auto-detect limits | Better image, warp, crop; human verify |
| Compliance stuck or wrong | Missing `state_code`, bad corners, ink metrics | Send state; fix perspective; review silent ink flags |
| No English `translated_text` | No Gemini key or API error | Configure Gemini; check logs / `get_last_gemini_error` |
| Missing GSTIN in response | OCR didn’t yield valid 15-char pattern | Manual entry or re-scan |

---

## 10. Summary

The system is strong for **standard printed shop signage** with **reasonable capture quality** and **correct perspective correction**. It is weaker for **severe blur, extreme angles, heavy occlusion, exotic fonts, and handwritten text**. **Compliance** depends on **state context**, **geometry**, and **thresholded ink math**, not only raw OCR. Treat **Vision + geometry + optional Gemini** as an **accelerator**; reserve **human review** for high-stakes or ambiguous cases.
