# Frontend Integration Handoff (OCR UI)

This guide explains how the frontend is structured, how each workflow runs, and how to merge it into another project with minimal guesswork.

## 1) Tech Stack and Entry Points

- Framework: React + TypeScript + Vite
- Main entry: `frontend/src/App.tsx`
- Dev server/proxy config: `frontend/vite.config.ts`
- API base resolution helper: `frontend/src/apiBase.ts`
- Shared API response types: `frontend/src/ocrTypes.ts`

## 2) Frontend Directory Map

Only app-relevant files are listed (excluding `node_modules`, `dist`):

- `frontend/src/App.tsx`
  - Main screen state machine (upload -> corners -> warp -> OCR -> review/audit).
- `frontend/src/App.css`
  - Complete styling for upload, corner UI, results, review modal, governance, debug, etc.
- `frontend/src/CornerCanvas.tsx`
  - Corner suggestion + drag/manual corner editing + perspective apply.
- `frontend/src/perspective.ts`
  - Client-side homography/perspective warp and mobile-safe output sizing.
- `frontend/src/imageRotate.ts`
  - Post-warp 90-degree image rotation utility.
- `frontend/src/LocationPanel.tsx`
  - Geolocation confirmation and manual state override flow.
- `frontend/src/LocationMapPreview.tsx`
  - Optional map iframe preview.
- `frontend/src/ImageMetadataPanel.tsx`
  - Displays backend-returned image metadata.
- `frontend/src/ocrTypes.ts`
  - TS interfaces matching backend response models (`OcrResult`, `TextBlock`, `ComplianceInfo`, etc.).
- `frontend/src/apiBase.ts`
  - Handles localhost-vs-LAN API base behavior.
- `frontend/.env.example`
  - Frontend environment variables.

## 3) Runtime Workflow (End-to-End)

## 3.1 Image Intake

- User can:
  - upload file from gallery,
  - drag-and-drop,
  - capture from mobile camera (`capture="environment"` path in UI).
- Supported flow includes PDF and image upload; image flow uses corner/wrap tools.

## 3.2 Corner Detection and Warp

- `CornerCanvas.tsx` uploads image to `POST /api/suggest-corners`.
- Server returns:
  - `corners`: `[TL, TR, BR, BL]`
  - `method`: now may be `hybrid_text+edges`, `image_edges`, `text_hull`, or fallback
- User can drag corner handles or click manually.
- `perspective.ts` applies homography in-browser and returns warped canvas/blob.
- Mobile safeguards:
  - source downscale threshold,
  - output side/pixel caps,
  - pointer/touch-safe behavior.

## 3.3 OCR Run

- `App.tsx` builds `FormData` and calls `POST /api/ocr`.
- Includes:
  - `file`
  - `state_code` (if selected)
  - `category` (store category for statutory checks)
  - optional user/location info
- OCR result updates:
  - text regions,
  - languages,
  - logos,
  - compliance + governance hints,
  - signboard geometry,
  - detected residual graphics,
  - audit id.

## 3.4 Review and Audit

- Doubtful OCR words/blocks highlighted.
- Review modal allows correction and verified language assignment.
- New graphics review section:
  - "Unrecognized Graphics Detected"
  - thumbnail preview per graphic region
  - toggle "Is this a Brand Logo?"
- Confirmation calls `PUT /api/audit/{audit_id}` with `verified_text_json`.
  - Block entries: corrected text/language.
  - Graphic entries: `graphic_<index>: { type: "brand_logo", brand_logo: true }`.

## 3.5 Translation + Geo State Confirmation

- `POST /api/translate` for on-demand English rendering.
- `LocationPanel.tsx`:
  - checks browser geolocation,
  - calls `/api/geolookup`,
  - asks user to confirm,
  - writes resolved state/city back to audit via `PUT /api/audit/{id}`.

## 4) API Contract Used by Frontend

Primary endpoints consumed:

- `POST /api/suggest-corners`
- `POST /api/ocr`
- `PUT /api/audit/{audit_id}`
- `POST /api/translate`
- `GET /api/compliance/states`
- `GET /api/geolookup`
- Static images: `/files/...`

Important response structures already represented in `ocrTypes.ts`:

- `OcrResult`
- `TextBlock` + `VerifiedWord`
- `ComplianceInfo` + governance payload
- `SignboardGeometrySummary`
- `detected_graphics` and `signboard_geometry.detected_graphics`

## 5) Environment and Networking

From `frontend/.env.example`:

- `VITE_API_URL`
  - keep empty in dev to use Vite proxy (`/api`, `/files`).
  - set only if backend is on separate origin.
- `VITE_GOOGLE_MAPS_API_KEY` (optional map preview).

LAN behavior:

- `apiBase.ts` prevents mobile devices from accidentally calling `127.0.0.1` as their own loopback.
- `vite.config.ts` sets `host: true` and proxies `/api` + `/files` to backend.

## 6) Merge Checklist for Your Friend

1. Copy frontend app files listed in section 2.
2. Ensure backend exposes all endpoints in section 4.
3. Keep `ocrTypes.ts` in sync with backend Pydantic response models.
4. Preserve Vite proxy config for local/LAN dev.
5. Verify corner flow:
   - upload image -> suggested corners -> warp -> run OCR.
6. Verify review flow:
   - doubtful block correction,
   - brand logo toggle persistence through `PUT /api/audit`.
7. Run:
   - `npm install`
   - `npm run build`
   - `npm run dev`

## 7) Known Integration-Sensitive Points

- `verified_text_json` format is overloaded (text-block and graphic decisions in one map).
- `detected_graphics` coordinates are in image pixel space used by OCR response.
- If backend response shape changes, update `ocrTypes.ts` first, then UI usage.
- Geolocation is optional but state selection is critical for compliance calculations.

