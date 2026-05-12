# Backend Integration Handoff (OCR + Compliance API)

This guide explains backend structure, workflows, and merge points so another engineer can integrate this module into a larger system quickly.

## 1) Tech Stack

- API framework: FastAPI
- OCR: Google Cloud Vision (`DOCUMENT_TEXT_DETECTION`, `LOGO_DETECTION`)
- CV: OpenCV + NumPy
- Translation/script-confidence: Gemini
- Persistence: SQLAlchemy (SQLite default, PostgreSQL-ready)
- Storage: local file storage with optional S3-compatible backend

Main server file: `backend/main.py`

## 2) Backend Directory Map

- `backend/main.py`
  - FastAPI app, all endpoint handlers, request/response models, orchestration pipeline.
- `backend/signboard_agent.py`
  - Rectification + threshed ink geometry + language share + residual graphic detection.
- `backend/compliance_rules.py`
  - State rule registry + compliance evaluation + category overlays + verified text/graphic reweighting.
- `backend/governance_evaluate.py`
  - Additional heuristic governance checks (position/order/layout checks per state).
- `backend/governance_data.py`
  - Governance rule metadata table.
- `backend/vision_preprocess.py`
  - Light-aware pre-OCR image enhancement, optional smart pre-binarization.
- `backend/image_prep.py`
  - AVIF/HEIC/JXL handling, transcoding to PNG for Vision compatibility.
- `backend/blur_check.py`
  - Blur rejection gate (Laplacian variance).
- `backend/vision_verification.py`
  - Symbol/word confidence extraction for doubtful flags.
- `backend/gemini_postprocess.py`
  - Translation + script confidence checks.
- `backend/gstin_extract.py`
  - GSTIN extraction/validation mapping to blocks.
- `backend/geocoding.py`
  - Reverse geocode helper.
- `backend/db_models.py`
  - SQLAlchemy models (`AuditRecord`).
- `backend/database.py`
  - Engine/session factory + lightweight schema evolution (`ALTER TABLE` additions).
- `backend/storage.py`
  - Upload persistence (local/S3), public URL generation.
- `backend/language_columns.py`
  - SQL-report-friendly denormalized threshed language fields.
- `backend/run_network.ps1`
  - LAN-friendly startup script (`0.0.0.0:8000`).

## 3) API Endpoints and Responsibility

- `POST /api/ocr`
  - full OCR + preprocessing + geometry + compliance + persistence
  - returns `OcrResult`
- `POST /api/suggest-corners`
  - auto corner estimate (hybrid text+edges)
- `PUT /api/audit/{audit_id}`
  - applies verified text/logo decisions and recomputes compliance
- `POST /api/translate`
  - on-demand translation for text snippets
- `GET /api/compliance/states`
  - supported state list for UI selector
- `GET /api/geolookup`
  - reverse geocode latitude/longitude

## 4) `POST /api/ocr` Workflow (Core Flow)

1. Validate upload type and bytes.
2. Optional transcode for Vision-incompatible modern formats (AVIF/HEIC/JXL -> PNG).
3. Blur gate on raster images (`blur_check.py`) rejects low-quality images early.
4. Persist original upload (`storage.py`) and keep image URL for audit.
5. Resolve state from `state_code` or optional geolocation.
6. Preprocess image (`vision_preprocess.py`) for OCR robustness.
7. Vision call:
   - document text detection
   - logo detection
8. Build block-level OCR objects:
   - text, boxes, normalized areas
   - word confidences and doubtful flags
9. Run `SignboardGeometryAgent`:
   - canonical rectification,
   - threshed language ratio computation,
   - silent region metrics,
   - residual contour graphics (`detected_graphics`).
10. Optional Gemini translation per block + full translated text.
11. GSTIN extraction and block linking.
12. Compliance:
   - state ratio rules + script allowlist (`compliance_rules.py`)
   - governance overlays (`governance_evaluate.py`)
   - category/statutory overlays.
13. Save `AuditRecord` + return `OcrResult`.

## 5) Corner Detection Workflow (`POST /api/suggest-corners`)

Corner logic is now hybrid:

- text-based corners from OCR blocks (`auto_corners`)
- image edge/contour rectangle corners (`auto_corners_from_image`)
- fusion heuristic (`auto_corners_hybrid`)

Response returns:

- `corners` `[TL, TR, BR, BL]`
- `method`:
  - `hybrid_text+edges`
  - `image_edges`
  - `text_hull`
  - fallback `full_image`

## 6) Text/Graphic Separation and Brand Logo Verification

Implemented in `signboard_agent.py` and consumed in `main.py` + compliance engine:

- Build threshed foreground map.
- Build OCR text mask from word/block polygons.
- Subtract text mask -> residual foreground.
- Contour residuals -> graphic bounding boxes.
- Reject contours with >20% overlap against OCR text boxes.
- Return `detected_graphics` with `type: "unknown_graphic"` and share metrics.

Audit stage (`PUT /api/audit/{id}`):

- reads `verified_text_json`
- supports `graphic_<idx>: { type: "brand_logo", brand_logo: true }`
- compliance engine can use verified logos in text-vs-graphic ratio checks.

## 7) Compliance Engine Composition

Primary rule engine (`compliance_rules.py`):

- per-state primary script and minimum share requirements
- script allowlist checks with noise-floor logic (e.g., Hmong misfire handling)
- optional text-vs-graphic thresholds (e.g., Delhi overlay)
- category overlays:
  - Pharmacy, Clinic, Jewelry, Liquor, Bank, General Retail
  - checks for mandatory IDs like DL, registration, IFSC, GSTIN

Governance layer (`governance_evaluate.py`):

- positional/sequence checks (top-line/upper-half/first-sequence etc.)
- state-specific heuristic reporting merged with compliance message/status.

## 8) Audit and Persistence Model

`AuditRecord` stores:

- upload URL
- raw OCR payload (`raw_ocr_json`)
- user-verified payload (`verified_text_json`)
- resolved location
- compliance status
- denormalized threshed-language columns for reporting

`database.py` includes `init_db()` and automatic missing-column addition for evolving schemas.

## 9) Environment and Deployment

From `backend/.env.example` important settings:

- Vision auth:
  - `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_API_KEY`
- Optional Gemini:
  - `GEMINI_API_KEY`
  - `GEMINI_MODEL`
- DB:
  - `DATABASE_URL` (falls back to SQLite if unset)
- CORS:
  - `CORS_EXTRA_ORIGINS`
- Storage:
  - `PUBLIC_BASE_URL`, `LOCAL_STORAGE_DIR`, optional `S3_*`
- Preprocessing flags:
  - `VISION_OCR_PREBINARIZE`
  - `OCR_DISABLE_LIGHT_AWARE`

LAN startup:

- `backend/run_network.ps1`
- or `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

## 10) Merge Checklist for Your Friend

1. Copy all backend files listed in section 2.
2. Install dependencies from `backend/requirements.txt`.
3. Configure `.env` for Vision auth first, then optional Gemini.
4. Ensure frontend contract alignment with `OcrResult` (especially `detected_graphics`, compliance payloads).
5. Verify these flows in order:
   - `/api/suggest-corners`
   - `/api/ocr`
   - `/api/audit/{id}` with `verified_text_json`
   - `/api/translate`
6. Validate DB initialization on first boot (`init_db()`).
7. Validate static `/files` route if using local storage.

## 11) Known Integration-Sensitive Areas

- `main.py` is orchestration-heavy; endpoint model changes can ripple into frontend type contracts.
- `verified_text_json` stores both block corrections and graphic logo decisions.
- Compliance output has multiple layers:
  - base compliance,
  - governance automated checks,
  - category overlays.
- If your friend already has a compliance engine, merge `evaluate_compliance` carefully and preserve `recompute_geometry_after_verified_overrides`.

