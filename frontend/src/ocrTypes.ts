/** Shapes for OCR API JSON (matches backend Pydantic models). */

export interface BoundingBox { x: number; y: number; w: number; h: number }
export interface NormArea {
  x_pct: number; y_pct: number; w_pct: number; h_pct: number; area_pct: number
  threshed_area_pct?: number | null
}
export interface LanguageHint { code: string; confidence: number }
export interface VerifiedWord { text: string; confidence: number; is_doubtful: boolean }
export interface TextBlock {
  text: string
  bounds: BoundingBox | null
  norm: NormArea | null
  languages: LanguageHint[]
  words?: VerifiedWord[]
  translated_text: string | null
  /** Block-level: silent region, etc. (words may be empty) */
  is_doubtful?: boolean
  warning_type?: string | null
  warning_message?: string | null
  /** Share of total threshed ink in returned blocks; used for manual language → compliance re-score */
  threshed_ink_share?: number | null
}
export interface LogoResult {
  name: string; confidence: number; bounds: BoundingBox | null; norm: NormArea | null
}
export interface GstinFinding {
  value: string
  format_valid: boolean
  checksum_valid: boolean
  block_index: number | null
}
/** Heuristic public signboard policy evaluation for the selected state/UT. */
export interface ComplianceGovernance {
  state_code?: string
  jurisdiction?: string
  overall?: string
  headline?: string
  automated?: { id: string; ok: boolean; detail: string }[]
  not_verifiable?: { area: string; text: string }[]
  ink_rule_eval?: { status?: string; message?: string; details?: unknown } | null
  /** Category-specific statutory overlays (pharmacy/clinic/etc.). */
  category_overlays?: unknown
  citations?: unknown[]
}
export interface ComplianceInfo {
  status: 'pending' | 'pass' | 'fail' | 'uncertain' | 'partial' | string
  message: string
  ruleset_id: string | null
  governance?: ComplianceGovernance | null
}
export interface SignboardGeometrySummary {
  language_ratios: Record<string, number>
  language_details: Record<string, { ink_pixels: number; board_ratio: number; text_ratio: number; block_count: number }>
  compliance_status: string
  confidence_score: number
  dominant_language: string
  primary_language: string
  primary_ratio: number
  /** BCP-47: ≈100% of threshed ink in this script; enables monolingual min-% pass */
  monolingual_single_lang?: string | null
  /** Threshed ink % of full rectified board, Vision page-block order (parallel to internal OCR blocks before filtering). */
  per_block_threshed_board_pct?: number[] | null
  detected_graphics?: DetectedGraphic[] | null
}
export interface DetectedGraphic {
  x: number
  y: number
  w: number
  h: number
  type?: string
  threshed_ink_share?: number | null
  bbox_area_share?: number | null
}
export interface ResolvedLocation {
  state_code?: string | null
  city?: string | null
  state_name?: string | null
  latitude?: number | null
  longitude?: number | null
}
/** File + technical metadata returned with OCR (also stored in audit `raw_ocr_json`). */
export interface ImageFileMeta {
  name: string
  size_bytes: number
  content_type: string
}
export interface ImageMetadata {
  file: ImageFileMeta
  /** Page size used by Vision (may differ from embedded decode if re-encoded). */
  vision?: { width: number; height: number }
  pdf?: { is_pdf?: boolean }
  /** Decoded dimensions, format, color mode, optional `exif` map, or `error` / `note`. */
  raster?: Record<string, unknown>
}
export interface OcrResult {
  text: string
  image_width: number
  image_height: number
  languages: LanguageHint[]
  blocks: TextBlock[]
  logos: LogoResult[]
  gstin: GstinFinding[]
  compliance: ComplianceInfo
  translated_text: string
  signboard_geometry: SignboardGeometrySummary | null
  detected_graphics?: DetectedGraphic[]
  is_pdf_ocr?: boolean
  audit_id?: string | null
  resolved_location?: ResolvedLocation | null
  /**
   * Per block index: legacy string, or { text, language, verified_language } for compliance re-attribution
   * (`verified_language` is the user’s script choice for threshed-ink bucketing).
   */
  verified_text_json?: Record<
    string,
    string | { text: string; language?: string; verified_language?: string }
  > | null
  image_metadata?: ImageMetadata | null
  category?: string | null
}
