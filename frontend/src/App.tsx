import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
} from 'react'
import CornerCanvas from './CornerCanvas'
import { getApiBase } from './apiBase'
import { rotateImageBlob } from './imageRotate'
import { ImageMetadataPanel } from './ImageMetadataPanel'
import { LocationPanel } from './LocationPanel'
import type {
  BoundingBox, DetectedGraphic, NormArea, OcrResult, TextBlock, VerifiedWord,
} from './ocrTypes'
import './App.css'

const API_BASE = getApiBase()

// ── Helpers ──────────────────────────────────────────────────────────────────

function langName(code: string): string {
  try { return new Intl.DisplayNames(['en'], { type: 'language' }).of(code) ?? code }
  catch { return code }
}

function pct(v: number) { return `${Math.round(v * 100)}%` }
function f1(v: number)   { return v.toFixed(1) }

const MANUAL_BCP47_OPTIONS = [
  'und', 'en', 'kn', 'hi', 'te', 'ta', 'ml', 'mr', 'gu', 'or', 'pa', 'as', 'bn', 'ur',
] as const

function blockHasDoubtful(blk: TextBlock): boolean {
  return (blk.words ?? []).some((w) => w.is_doubtful)
}

/** Threshed ink but no/low OCR, word doubt, or block flag — open review. */
function blockNeedsReview(blk: TextBlock): boolean {
  if (blk.is_doubtful || blk.warning_type === 'SILENT_REGION_DETECTED') return true
  return blockHasDoubtful(blk)
}

function correctionText(
  c: string | { text: string; language: string; verified_language?: string } | undefined,
  fallback: string,
): string {
  if (c == null) return fallback
  if (typeof c === 'string') return c
  return c.text
}

/** Match backend `SignboardGeometryAgent`: below this, status is UNCERTAIN (blurry / sparse / dark). */
const RETAKE_GEOMETRY_CONF = 0.3
const RETAKE_DOUBT_WORD_RATIO = 0.4
const RETAKE_MIN_WORDS_FOR_RATIO = 6

/**
 * When true, the UI should ask for a new photo: uncertain geometry (blur/ink) or too many low-confidence words.
 */
function computeRetakeLines(r: OcrResult): string[] {
  const g = r.signboard_geometry
  if (g) {
    const st = (g.compliance_status || '').toUpperCase()
    if (st === 'UNCERTAIN' || g.confidence_score < RETAKE_GEOMETRY_CONF) {
      return [
        'This photo looks too blurry, too dark, or has too little readable sign text for a reliable scan. Please take a new photo with the sign in focus and good lighting.',
      ]
    }
  }

  let n = 0
  let doubt = 0
  for (const b of r.blocks) {
    for (const w of b.words ?? []) {
      n += 1
      if (w.is_doubtful) doubt += 1
    }
  }
  if (n >= RETAKE_MIN_WORDS_FOR_RATIO && doubt / n >= RETAKE_DOUBT_WORD_RATIO) {
    return [
      'Most character reads in this image are low confidence. Upload a sharper, well-lit photo of the sign.',
    ]
  }
  if (n >= 3 && n === doubt) {
    return [
      'Every detected word is low confidence. Please upload a clearer photo before relying on this text.',
    ]
  }
  return []
}

/** Walks `words` in order and pieces together `text`, marking doubtful word spans. */
function BlockTextHighlighted({ text, words }: { text: string; words?: VerifiedWord[] }) {
  if (!words?.length) {
    return <p className="block-text">{text.length > 500 ? text.slice(0, 500) + '…' : text}</p>
  }
  const nodes: ReactNode[] = []
  let rest = text
  let k = 0
  for (const w of words) {
    if (!w.text) continue
    const idx = rest.indexOf(w.text)
    if (idx === -1) {
      nodes.push(<span key={k++}>{rest}</span>)
      rest = ''
      break
    }
    if (idx > 0) {
      nodes.push(<span key={k++}>{rest.slice(0, idx)}</span>)
    }
    nodes.push(
      w.is_doubtful ? (
        <mark key={k++} className="word--doubtful" title="Low confidence (OCR < 0.85)">
          {w.text}
        </mark>
      ) : (
        <span key={k++}>{w.text}</span>
      ),
    )
    rest = rest.slice(idx + w.text.length)
  }
  if (rest) {
    nodes.push(<span key={k++}>{rest}</span>)
  }
  return <p className="block-text block-text--with-marks">{nodes}</p>
}

function AreaInfo({ norm, bounds }: { norm: NormArea | null; bounds: BoundingBox | null }) {
  if (norm) {
    return (
      <span className="area-info">
        <span className="area-chip" title="Left edge position on signboard">⇥ {f1(norm.x_pct)}%</span>
        <span className="area-chip" title="Top edge position on signboard">↓ {f1(norm.y_pct)}%</span>
        <span className="area-chip area-chip--size" title="Width × Height on signboard">
          {f1(norm.w_pct)}% × {f1(norm.h_pct)}%
        </span>
        <span className="area-chip area-chip--area" title="Share of signboard (bounding box)">
          ▣ {f1(norm.area_pct)}% of board
        </span>
        {norm.threshed_area_pct != null && norm.threshed_area_pct !== undefined && (
          <span className="area-chip" title="Ink from adaptive threshold (rectified board)">◆ {f1(norm.threshed_area_pct)}% threshed</span>
        )}
      </span>
    )
  }
  if (bounds) {
    return <span className="block-bounds">({bounds.x}, {bounds.y}) {bounds.w}×{bounds.h} px</span>
  }
  return null
}

function GraphicThumb({
  src,
  g,
  imageWidth,
  imageHeight,
}: {
  src: string
  g: DetectedGraphic
  imageWidth: number
  imageHeight: number
}) {
  const [thumb, setThumb] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      try {
        const iw = Math.max(1, imageWidth || img.naturalWidth)
        const ih = Math.max(1, imageHeight || img.naturalHeight)
        const sx = img.naturalWidth / iw
        const sy = img.naturalHeight / ih
        const x = Math.max(0, Math.floor((g.x || 0) * sx))
        const y = Math.max(0, Math.floor((g.y || 0) * sy))
        const w = Math.max(2, Math.floor((g.w || 2) * sx))
        const h = Math.max(2, Math.floor((g.h || 2) * sy))
        const c = document.createElement('canvas')
        c.width = 96; c.height = 96
        const ctx = c.getContext('2d')
        if (!ctx) return
        ctx.fillStyle = '#111827'
        ctx.fillRect(0, 0, c.width, c.height)
        const scale = Math.min(c.width / w, c.height / h)
        const dw = Math.max(1, Math.floor(w * scale))
        const dh = Math.max(1, Math.floor(h * scale))
        const dx = Math.floor((c.width - dw) / 2)
        const dy = Math.floor((c.height - dh) / 2)
        ctx.drawImage(img, x, y, w, h, dx, dy, dw, dh)
        const u = c.toDataURL('image/png')
        if (alive) setThumb(u)
      } catch {
        if (alive) setThumb(null)
      }
    }
    img.onerror = () => { if (alive) setThumb(null) }
    img.src = src
    return () => { alive = false }
  }, [src, g.x, g.y, g.w, g.h, imageWidth, imageHeight])
  if (!thumb) return <div className="graphic-thumb graphic-thumb--empty">n/a</div>
  return <img src={thumb} alt="Detected graphic thumbnail" className="graphic-thumb" />
}

type Phase = 'idle' | 'corners' | 'warped'
type OcrStatus = 'idle' | 'loading' | 'success' | 'error'
type NetDebugLevel = 'off' | 'basic' | 'verbose'

type StoreCategory =
  | ''
  | 'General Retail'
  | 'Pharmacy'
  | 'Clinic'
  | 'Jewelry'
  | 'Liquor/Bars'
  | 'Bank/ATM'

// ── App ──────────────────────────────────────────────────────────────────────

function App() {
  /** Gallery / all supported types (incl. PDF) — not camera-only. */
  const inputRef = useRef<HTMLInputElement>(null)
  /** `capture` — on phones opens the device camera; desktop may fall back to a picker. */
  const cameraInputRef = useRef<HTMLInputElement>(null)

  const [phase, setPhase] = useState<Phase>('idle')
  const [file, setFile] = useState<File | null>(null)
  const [warpedBlob, setWarpedBlob] = useState<Blob | null>(null)
  const [warpedUrl, setWarpedUrl] = useState<string | null>(null)
  const [rotatingWarped, setRotatingWarped] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const [ocrStatus, setOcrStatus] = useState<OcrStatus>('idle')
  const [ocrResult, setOcrResult] = useState<OcrResult | null>(null)
  const [ocrError, setOcrError] = useState<string | null>(null)
  const [auditModalIndex, setAuditModalIndex] = useState<number | null>(null)
  const [auditDraft, setAuditDraft] = useState('')
  const [auditLang, setAuditLang] = useState('und')
  const [blockCorrections, setBlockCorrections] = useState<
    Record<number, string | { text: string; language: string; verified_language?: string }>
  >({})
  const [reviewedDoubtful, setReviewedDoubtful] = useState<Set<number>>(() => new Set())
  const [graphicDecisions, setGraphicDecisions] = useState<Record<number, boolean>>({})
  const [auditSessionComplete, setAuditSessionComplete] = useState(false)
  const [clientTranslations, setClientTranslations] = useState<Record<number, string>>({})
  const [translateLoading, setTranslateLoading] = useState(false)
  const [translateMessage, setTranslateMessage] = useState<string | null>(null)
  const [textView, setTextView] = useState<'original' | 'english'>('original')
  const [netDebugLevel, setNetDebugLevel] = useState<NetDebugLevel>('basic')
  const [netDebugLines, setNetDebugLines] = useState<string[]>([])
  /** State/UT for `/api/ocr` Form `state_code` — drives COMPLai + governance (OCR is script-agnostic). */
  const [complianceStates, setComplianceStates] = useState<{ code: string; name: string }[]>([])
  const [ocrStateCode, setOcrStateCode] = useState('')
  const [storeCategory, setStoreCategory] = useState<StoreCategory>('')

  /** Broad image extension match for browsers that omit or misreport MIME (e.g. some HEIC). */
  const looksLikeImageName = (f: File) =>
    /\.(jpe?g|jpe|jfif|pjp|pjpeg|png|gif|webp|bmp|dib|tiff?|tif|ico|heic|heif|hif|avif|jxl|svgz?|jp2|j2[ck]|jpc|jpx|jxr|hdp|wdp|bpg|apng|mng|qoi|dng|cr2|nef|raw?|jps|jph|xbm|pnm|pbm|pgm|ppm|pam|jng)$/i
      .test(f.name)

  const isImage = (f: File) =>
    f.type.startsWith('image/') || (f.type === '' || f.type === 'application/octet-stream' ? looksLikeImageName(f) : false)

  const isOcrInput = (f: File) => {
    if (f.type.startsWith('image/') || f.type === 'application/pdf') return true
    if (f.type === 'application/octet-stream' || f.type === '') {
      return looksLikeImageName(f) || /\.(pdf|ai)$/i.test(f.name)
    }
    return looksLikeImageName(f) || /\.(pdf|ai)$/i.test(f.name)
  }

  const resetOcr = () => {
    setOcrStatus('idle')
    setOcrResult(null)
    setOcrError(null)
    setBlockCorrections({})
    setReviewedDoubtful(new Set())
    setGraphicDecisions({})
    setAuditModalIndex(null)
    setAuditDraft('')
    setAuditLang('und')
    setAuditSessionComplete(false)
    setClientTranslations({})
    setTranslateMessage(null)
    setTextView('original')
  }

  const pushNetDebug = useCallback((msg: string) => {
    if (netDebugLevel === 'off') return
    const t = new Date()
    const hh = String(t.getHours()).padStart(2, '0')
    const mm = String(t.getMinutes()).padStart(2, '0')
    const ss = String(t.getSeconds()).padStart(2, '0')
    const line = `${hh}:${mm}:${ss} ${msg}`
    setNetDebugLines((prev) => [...prev.slice(-19), line])
  }, [netDebugLevel])

  useEffect(() => {
    if (netDebugLevel === 'off') return
    pushNetDebug(
      `[boot] origin=${typeof window !== 'undefined' ? window.location.origin : 'unknown'} apiBase=${API_BASE || '(same-origin)'}`,
    )
  }, [netDebugLevel, pushNetDebug])

  const pickFile = (f: File | null) => {
    if (!f) return
    if (f.type.startsWith('video/') || f.type.startsWith('audio/')) {
      setOcrError('Please choose an image or a PDF, not video or audio.')
      return
    }
    if (!isOcrInput(f)) {
      setOcrError('Please choose a supported image (any common format) or a PDF.')
      return
    }
    if (warpedUrl) URL.revokeObjectURL(warpedUrl)
    setFile(f); setWarpedBlob(null); setWarpedUrl(null)
    setRotatingWarped(false)
    resetOcr()
    setStoreCategory('')
    setPhase(isImage(f) && f.type !== 'application/pdf' && !/\.(pdf|ai)$/i.test(f.name) ? 'corners' : 'warped')
  }

  const handleWarped = useCallback((blob: Blob, url: string) => {
    if (warpedUrl) URL.revokeObjectURL(warpedUrl)
    setRotatingWarped(false)
    setWarpedBlob(blob); setWarpedUrl(url)
    setPhase('warped'); resetOcr()
  }, [warpedUrl])

  const onRotateWarped = useCallback(
    async (direction: 'cw' | 'ccw') => {
      if (!warpedBlob || !warpedUrl) return
      setRotatingWarped(true)
      setOcrError(null)
      try {
        const next = await rotateImageBlob(warpedBlob, direction)
        URL.revokeObjectURL(warpedUrl)
        setWarpedBlob(next)
        setWarpedUrl(URL.createObjectURL(next))
        resetOcr()
      } catch (e) {
        setOcrError(
          e instanceof Error ? e.message : 'Could not rotate image. Try a smaller file.',
        )
      } finally {
        setRotatingWarped(false)
      }
    },
    [warpedBlob, warpedUrl],
  )

  const onFilePicked = (e: ChangeEvent<HTMLInputElement>) => {
    pickFile(e.target.files?.[0] ?? null)
    e.target.value = ''
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault(); setIsDragging(false)
    pickFile(e.dataTransfer.files?.[0] ?? null)
  }

  const runOcr = async () => {
    if (!file) { setOcrError('Select a file first.'); return }
    setOcrStatus('loading'); setOcrError(null);     setOcrResult(null)
    setBlockCorrections({})
    setReviewedDoubtful(new Set())
    setAuditModalIndex(null)
    setAuditSessionComplete(false)
    setClientTranslations({})
    setTranslateMessage(null)
    setTextView('original')

    const upload: File = warpedBlob
      ? new File([warpedBlob], 'board.png', { type: 'image/png' })
      : file

    const form = new FormData()
    form.append('file', upload)
    const uid = typeof localStorage !== 'undefined' ? localStorage.getItem('ocr_user_id') : ''
    if (uid) form.append('user_id', uid)
    const st = (ocrStateCode || '').trim().toUpperCase()
    if (st) form.append('state_code', st)
    const cat = (storeCategory || '').trim()
    if (cat) form.append('category', cat)
    pushNetDebug(
      `[OCR start] base=${API_BASE || '(same-origin)'} file=${upload.name} size=${upload.size} state=${st || 'none'} category=${cat || 'none'}`,
    )

    try {
      const res = await fetch(`${API_BASE}/api/ocr`, { method: 'POST', body: form })
      pushNetDebug(`[OCR response] status=${res.status} ok=${String(res.ok)}`)
      const data: unknown = await res.json().catch(() => ({}))
      if (!res.ok) {
        let msg = `Request failed (${res.status})`
        if (typeof data === 'object' && data !== null && 'detail' in data) {
          const d = (data as { detail: unknown }).detail
          if (typeof d === 'string') msg = d
        }
        setOcrError(msg); setOcrStatus('error'); return
      }
      if (netDebugLevel === 'verbose') {
        const blocksN = typeof data === 'object' && data !== null && 'blocks' in data
          && Array.isArray((data as { blocks?: unknown[] }).blocks)
          ? (data as { blocks: unknown[] }).blocks.length
          : 'unknown'
        pushNetDebug(`[OCR parsed] blocks=${String(blocksN)}`)
      }
      setOcrResult(data as OcrResult); setOcrStatus('success')
    } catch (err) {
      pushNetDebug(
        `[OCR error] ${err instanceof Error ? err.message : 'network error / fetch failed'}`,
      )
      setOcrError(err instanceof Error ? err.message : 'Network error. Is the API running?')
      setOcrStatus('error')
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    void runOcr()
  }

  const { doubtfulBlockIndices, canConfirmAudit, retakeLines } = useMemo(() => {
    if (!ocrResult) {
      return {
        doubtfulBlockIndices: [] as number[],
        canConfirmAudit: true,
        retakeLines: [] as string[],
      }
    }
    const retakeLines = computeRetakeLines(ocrResult)
    const retakeNeeded = retakeLines.length > 0
    const doubtfulBlockIndices = ocrResult.blocks
      .map((b, i) => (blockNeedsReview(b) ? i : -1))
      .filter((i): i is number => i >= 0)
    const canConfirmAudit = !retakeNeeded
      && (doubtfulBlockIndices.length === 0
        || doubtfulBlockIndices.every((i) => reviewedDoubtful.has(i)))
    return { doubtfulBlockIndices, canConfirmAudit, retakeLines }
  }, [ocrResult, reviewedDoubtful])

  const { fullEnglishText, hasEnglishView } = useMemo(() => {
    if (!ocrResult) {
      return { fullEnglishText: '', hasEnglishView: false }
    }
    const serverT = (ocrResult.translated_text || '').trim()
    const fromBlocks = ocrResult.blocks
      .map((b, i) => b.translated_text || clientTranslations[i] || b.text)
      .join('\n')
    if (serverT && serverT !== ocrResult.text) {
      return { fullEnglishText: serverT, hasEnglishView: true }
    }
    const has =
      Object.keys(clientTranslations).length > 0
      || ocrResult.blocks.some((b) => b.translated_text)
      || (fromBlocks.length > 0 && fromBlocks !== ocrResult.text)
    return { fullEnglishText: fromBlocks, hasEnglishView: has }
  }, [ocrResult, clientTranslations])

  const onTranslate = useCallback(async () => {
    if (!ocrResult || ocrResult.blocks.length === 0) return
    setTranslateLoading(true)
    setTranslateMessage(null)
    try {
      const texts = ocrResult.blocks.map((b) => b.text)
      const languages = ocrResult.blocks.map(
        (b) => b.languages[0]?.code || 'und',
      )
      const res = await fetch(`${API_BASE}/api/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts, languages }),
      })
      const data: unknown = await res.json().catch(() => ({}))
      if (!res.ok) {
        const d = (data as { detail?: string }).detail
        setTranslateMessage(typeof d === 'string' ? d : `Request failed (${res.status})`)
        return
      }
      const trs = (data as { translations?: (string | null)[]; detail?: string | null })
        .translations
      const detail = (data as { detail?: string | null }).detail
      if (Array.isArray(trs)) {
        const next: Record<number, string> = {}
        trs.forEach((t, i) => {
          if (t) next[i] = t
        })
        setClientTranslations(next)
      }
      if (detail) {
        setTranslateMessage(detail)
      } else {
        setTranslateMessage(null)
        setTextView('english')
      }
    } catch (e) {
      setTranslateMessage(e instanceof Error ? e.message : 'Translation request failed')
    } finally {
      setTranslateLoading(false)
    }
  }, [ocrResult])

  const closeAuditModal = useCallback(() => {
    setAuditModalIndex(null)
  }, [])

  useEffect(() => {
    if (!ocrResult?.verified_text_json) return
    const next: Record<number, boolean> = {}
    for (const [k, v] of Object.entries(ocrResult.verified_text_json)) {
      if (!k.startsWith('graphic_')) continue
      const idx = Number(k.slice('graphic_'.length))
      if (!Number.isInteger(idx) || idx < 0) continue
      if (v && typeof v === 'object' && 'brand_logo' in v) {
        next[idx] = Boolean((v as { brand_logo?: unknown }).brand_logo)
      } else if (v && typeof v === 'object' && 'type' in v) {
        next[idx] = String((v as { type?: unknown }).type || '').toLowerCase() === 'brand_logo'
      }
    }
    if (Object.keys(next).length > 0) setGraphicDecisions(next)
  }, [ocrResult])

  const openAuditModal = useCallback(
    (blockIndex: number) => {
      if (!ocrResult) return
      const blk = ocrResult.blocks[blockIndex]
      if (!blockNeedsReview(blk)) return
      setAuditModalIndex(blockIndex)
      const c = blockCorrections[blockIndex]
      if (c != null && typeof c === 'object' && 'text' in c) {
        setAuditDraft(c.text)
        setAuditLang(c.verified_language ?? c.language)
      } else if (typeof c === 'string') {
        setAuditDraft(c)
        setAuditLang(blk.languages[0]?.code ?? 'und')
      } else {
        setAuditDraft(blk.text)
        setAuditLang(blk.languages[0]?.code ?? 'und')
      }
    },
    [ocrResult, blockCorrections],
  )

  const saveAuditModal = useCallback(() => {
    if (auditModalIndex == null || !ocrResult) return
    const i = auditModalIndex
    const blk = ocrResult.blocks[i]
    const trimmed = auditDraft.trim()
    if (blk.warning_type === 'SILENT_REGION_DETECTED' && !trimmed) {
      return
    }
    setBlockCorrections((prev) => ({
      ...prev,
      [i]: {
        text: trimmed || blk.text,
        language: auditLang,
        verified_language: auditLang,
      },
    }))
    setReviewedDoubtful((s) => new Set(s).add(i))
    setAuditSessionComplete(false)
    setAuditModalIndex(null)
  }, [auditModalIndex, auditDraft, auditLang, ocrResult])

  const onRetakeAnotherPhoto = useCallback(() => {
    if (warpedUrl) URL.revokeObjectURL(warpedUrl)
    setFile(null)
    setWarpedBlob(null)
    setWarpedUrl(null)
    setRotatingWarped(false)
    setPhase('idle')
    resetOcr()
  }, [warpedUrl])

  const onConfirmAudit = useCallback(async () => {
    if (!canConfirmAudit || !ocrResult) return
    const id = ocrResult.audit_id
    if (!id) {
      setAuditSessionComplete(true)
      return
    }
    const verified: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(blockCorrections)) {
      const idx = Number(k)
      if (!Number.isInteger(idx) || !ocrResult.blocks[idx]) continue
      const b = ocrResult.blocks[idx]
      if (v != null && typeof v === 'object' && 'text' in v) {
        const lang = v.verified_language ?? v.language
        verified[k] = { text: v.text, language: lang, verified_language: lang }
      } else {
        const t = typeof v === 'string' ? v : b.text
        const lang0 = b.languages[0]?.code || 'und'
        verified[k] = { text: t, language: lang0, verified_language: lang0 }
      }
    }
    for (const [k, v] of Object.entries(graphicDecisions)) {
      const idx = Number(k)
      if (!Number.isInteger(idx) || !v) continue
      verified[`graphic_${idx}`] = {
        type: 'brand_logo',
        brand_logo: true,
      }
    }
    const st = ocrResult.resolved_location?.state_code
    const city = ocrResult.resolved_location?.city
    try {
      const r = await fetch(`${API_BASE}/api/audit/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          verified_text_json: Object.keys(verified).length > 0 ? verified : null,
          user_id: typeof localStorage !== 'undefined' ? localStorage.getItem('ocr_user_id') : null,
          ...(st ? { state_code: st } : {}),
          ...(typeof city === 'string' && city ? { city } : {}),
        }),
      })
      const data: unknown = await r.json().catch(() => ({}))
      if (r.ok && data && typeof data === 'object') {
        setOcrResult(data as OcrResult)
        setAuditSessionComplete(true)
      } else {
        const detail = typeof data === 'object' && data !== null && 'detail' in data
          ? (data as { detail?: unknown }).detail
          : null
        setOcrError(
          typeof detail === 'string'
            ? `Could not save audit: ${detail}`
            : `Could not save audit (HTTP ${r.status}).`,
        )
      }
    } catch (e) {
      setOcrError(e instanceof Error ? e.message : 'Could not save audit.')
    }
  }, [canConfirmAudit, ocrResult, blockCorrections, graphicDecisions])

  useEffect(() => {
    if (auditModalIndex == null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeAuditModal()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [auditModalIndex, closeAuditModal])

  useEffect(() => {
    let ok = true
    void (async () => {
      try {
        const statesUrl = `${API_BASE}/api/compliance/states`
        pushNetDebug(`[states] request ${statesUrl}`)
        const res = await fetch(statesUrl)
        pushNetDebug(`[states] GET /api/compliance/states -> ${res.status}`)
        const data: unknown = await res.json().catch(() => ({}))
        if (!ok || !res.ok) return
        if (data && typeof data === 'object' && 'states' in data) {
          const s = (data as { states: { code: string; name: string }[] }).states
          if (Array.isArray(s)) setComplianceStates(s)
        }
      } catch (e) {
        const detail = e instanceof Error ? e.message : String(e)
        pushNetDebug(`[states] failed to load: ${detail}`)
        /* keep empty; user can type via LocationPanel */
      }
    })()
    return () => { ok = false }
  }, [])

  return (
    <div
      className={[
        'app',
        'app--mobile',
        phase === 'warped' ? 'app--bottom-cta' : '',
        ocrStatus === 'success' && ocrResult ? 'app--has-results' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <header className="app-bar" role="banner">
        <div className="app-bar__inner">
          <h1 className="app-bar__title">Signboard OCR</h1>
          <p className="app-bar__subtitle">Scan · straighten · compliance &amp; translate</p>
        </div>
      </header>
      <main className="app-main" id="main-content" tabIndex={-1}>

      {/* ── File picker ── */}
      {phase === 'idle' ? (
        <form className="panel panel--upload" onSubmit={onSubmit}>
          <div
            className="capture-actions"
            role="group"
            aria-label="Add an image: camera or file"
          >
            <button
              type="button"
              className="btn-primary btn-capture"
              onClick={() => cameraInputRef.current?.click()}
            >
              Take photo
            </button>
            <button
              type="button"
              className="btn-ghost btn-capture"
              onClick={() => inputRef.current?.click()}
            >
              Choose file
            </button>
          </div>
          <p className="upload-source-hint">
            On a phone, <strong>Take photo</strong> opens the camera. Use <strong>Choose file</strong> for gallery or a PDF. On a laptop, Take photo may open your webcam or a file chooser, depending on the browser.
          </p>
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="file-input"
            tabIndex={-1}
            aria-hidden="true"
            onChange={onFilePicked}
          />
          <div
            className={`dropzone${isDragging ? ' dropzone--active' : ''}`}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }}
            onDragEnter={() => setIsDragging(true)}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*,application/pdf,application/octet-stream,.pdf"
              className="file-input"
              tabIndex={-1}
              aria-hidden="true"
              onChange={onFilePicked}
            />
            <p className="dropzone__hint">Or drag a file here — any image or PDF (same as Choose file)</p>
          </div>
        </form>
      ) : (
        <div className="repick-bar">
          <span className="repick-name">{file?.name}</span>
          <button type="button" className="btn-ghost btn-sm" onClick={() => {
            if (warpedUrl) URL.revokeObjectURL(warpedUrl)
            setFile(null); setWarpedBlob(null); setWarpedUrl(null); setRotatingWarped(false); setPhase('idle'); resetOcr()
          }}>Change image</button>
        </div>
      )}

      {/* ── Corner selector ── */}
      {phase === 'corners' && file && (
        <div className="panel">
          <CornerCanvas file={file} onWarped={handleWarped} apiBase={API_BASE} stateCode={ocrStateCode} />
          <div className="skip-row">
            <button type="button" className="btn-ghost btn-sm" onClick={() => setPhase('warped')}>
              Skip — use full image
            </button>
          </div>
        </div>
      )}

      {/* ── Preview + OCR button ── */}
      {phase === 'warped' && (
        <form className="panel form-ocr" onSubmit={onSubmit}>
          {warpedUrl ? (
            <div className="warped-preview-wrap">
              <img src={warpedUrl} alt="Corrected preview" className="warped-preview" />
              <p className="warped-rotate-hint">Rotate so the sign reads upright, then run OCR.</p>
              <div
                className="warped-rotate"
                role="group"
                aria-label="Rotate image in 90 degree steps"
              >
                <button
                  type="button"
                  className="btn-ghost btn-sm w-rotate"
                  disabled={rotatingWarped}
                  onClick={() => { void onRotateWarped('ccw') }}
                  title="Turn image 90° left"
                  aria-label="Rotate 90 degrees counterclockwise"
                >
                  ↺ 90°
                </button>
                <button
                  type="button"
                  className="btn-ghost btn-sm w-rotate"
                  disabled={rotatingWarped}
                  onClick={() => { void onRotateWarped('cw') }}
                  title="Turn image 90° right"
                  aria-label="Rotate 90 degrees clockwise"
                >
                  90° ↻
                </button>
                {rotatingWarped && (
                  <span className="warped-rotate--busy" aria-live="polite">Rotating…</span>
                )}
              </div>
              <div className="warped-actions">
                <span className="badge">Perspective corrected</span>
                {isImage(file!) && (
                  <button type="button" className="btn-ghost btn-sm"
                    onClick={() => { setWarpedBlob(null); setWarpedUrl(null); setRotatingWarped(false); setPhase('corners'); resetOcr() }}>
                    Re-select corners
                  </button>
                )}
              </div>
            </div>
          ) : (
            <p className="file-name-ready">Ready: <strong>{file?.name}</strong></p>
          )}
          <div className="ocr-state-row">
            <label className="ocr-state-row__label" htmlFor="ocr-state-sel">State/UT (for language rules)</label>
            <select
              id="ocr-state-sel"
              className="ocr-state-row__select"
              value={ocrStateCode}
              onChange={(e) => setOcrStateCode(e.target.value)}
            >
              <option value="">— Not set: COMPLai PENDING (state required) —</option>
              {complianceStates.map((s) => (
                <option key={s.code} value={s.code}>
                  {s.name} ({s.code})
                </option>
              ))}
            </select>
            <p className="ocr-state-row__hint">
              Choose the shop&apos;s state so threshed-ink and signboard policy checks can run on this image. You can change it later in the location panel.
            </p>
          </div>
          <div className="ocr-state-row">
            <label className="ocr-state-row__label" htmlFor="ocr-cat-sel">Store category (for statutory checks)</label>
            <select
              id="ocr-cat-sel"
              className="ocr-state-row__select"
              value={storeCategory}
              onChange={(e) => setStoreCategory(e.target.value as StoreCategory)}
            >
              <option value="">— Not set —</option>
              <option value="General Retail">General Retail</option>
              <option value="Pharmacy">Pharmacy</option>
              <option value="Clinic">Clinic / Hospital</option>
              <option value="Jewelry">Jewelry / Bullion</option>
              <option value="Liquor/Bars">Liquor / Bars</option>
              <option value="Bank/ATM">Bank / ATM</option>
            </select>
            <p className="ocr-state-row__hint">
              Enables category-specific checks (e.g. DL No. for pharmacies, Reg No. for clinics, IFSC for banks).
            </p>
          </div>
          <div className="actions">
            <button type="button" disabled={ocrStatus === 'loading' || rotatingWarped} onClick={() => { void runOcr() }}>
              {ocrStatus === 'loading' ? 'Reading…' : 'Run OCR'}
            </button>
          </div>
        </form>
      )}

      {/* ── Error ── */}
      {ocrError && <div className="message message--error" role="alert">{ocrError}</div>}
      <div className="net-debug" role="region" aria-label="Network debug">
        <div className="net-debug__top">
          <strong>Network debug</strong>
          <div className="net-debug__controls">
            <select
              className="net-debug__sel"
              value={netDebugLevel}
              onChange={(e) => setNetDebugLevel(e.target.value as NetDebugLevel)}
            >
              <option value="off">Off</option>
              <option value="basic">Basic</option>
              <option value="verbose">Verbose</option>
            </select>
            <button type="button" className="btn-ghost btn-sm" onClick={() => setNetDebugLines([])}>
              Clear
            </button>
          </div>
        </div>
        {netDebugLevel === 'off' ? (
          <p className="net-debug__hint">Debug is off.</p>
        ) : netDebugLines.length === 0 ? (
          <p className="net-debug__hint">No network events yet.</p>
        ) : (
          <pre className="net-debug__log">{netDebugLines.join('\n')}</pre>
        )}
      </div>

      {/* ── Rich results ── */}
      {ocrStatus === 'success' && ocrResult && (
        <div className="results">

          {retakeLines.length > 0 && (
            <div className="retake-banner" role="status" aria-live="polite">
              <h2 className="retake-banner__title">Please upload another photo</h2>
              {retakeLines.map((line) => (
                <p key={line} className="retake-banner__text">
                  {line}
                </p>
              ))}
              <button type="button" className="btn-primary retake-banner__btn" onClick={onRetakeAnotherPhoto}>
                Choose another image
              </button>
            </div>
          )}

          <div className="translate-bar" role="region" aria-label="Translation">
            <div className="translate-bar__row">
              <span className="translate-bar__label">Translation</span>
              <button
                type="button"
                className="btn-primary translate-bar__btn"
                disabled={translateLoading || ocrResult.blocks.length === 0}
                onClick={() => void onTranslate()}
              >
                {translateLoading ? 'Translating…' : 'Translate to English'}
              </button>
              {hasEnglishView && (
                <div className="view-toggle" role="group" aria-label="Detected text view">
                  <button
                    type="button"
                    className={
                      'view-toggle__opt' + (textView === 'original' ? ' view-toggle__opt--active' : '')
                    }
                    onClick={() => setTextView('original')}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    className={
                      'view-toggle__opt' + (textView === 'english' ? ' view-toggle__opt--active' : '')
                    }
                    onClick={() => setTextView('english')}
                  >
                    English
                  </button>
                </div>
              )}
            </div>
            {translateMessage && (
              <p className="translate-bar__msg" role="status">
                {translateMessage}
              </p>
            )}
            <p className="translate-bar__hint">
              Uses the same on-server translation as OCR (Gemini). Configure <code className="translate-bar__code">GEMINI_API_KEY</code> or <code className="translate-bar__code">GOOGLE_API_KEY</code> in the backend.
            </p>
          </div>

          <LocationPanel
            key={ocrResult.audit_id || 'noid'}
            apiBase={API_BASE}
            auditId={ocrResult.audit_id ?? null}
            ocrResult={ocrResult}
            onOcrPatched={setOcrResult}
          />
          <ImageMetadataPanel meta={ocrResult.image_metadata} />
          {/** dynamic border uses compliance status * */}
          <section
            className={[
              'result-section',
              ocrResult.compliance.status === 'pass'
                ? 'compliance-pass'
                : ocrResult.compliance.status === 'fail'
                  ? 'compliance-fail'
                  : 'compliance-pending',
            ].join(' ')}
          >
            <h2 className="result-heading">Geo-compliance (threshed language rules)</h2>
            <p className="compliance-pending__msg">
              {ocrResult.compliance.message}
              {ocrResult.compliance.ruleset_id && (
                <span className="compliance-pending__id">Ruleset: {ocrResult.compliance.ruleset_id}</span>
              )}
            </p>
            {ocrResult.compliance.governance && (
              <div className="governance-hint">
                <p className="governance-hint__title">State / UT policy (image heuristics)</p>
                <p className="compliance-pending__msg governance-hint__line">
                  {ocrResult.compliance.governance.headline}
                </p>
                {ocrResult.compliance.governance.automated && ocrResult.compliance.governance.automated.length > 0 && (
                  <ul className="governance-hint__list">
                    {ocrResult.compliance.governance.automated.map((a) => (
                      <li key={a.id} className={a.ok ? 'governance-ok' : 'governance-fail'}>
                        {a.id}: {a.ok ? 'OK' : 'Review'} — {a.detail}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </section>

          {(() => {
            const g0 = (ocrResult.detected_graphics && ocrResult.detected_graphics.length > 0)
              ? ocrResult.detected_graphics
              : (ocrResult.signboard_geometry?.detected_graphics || [])
            const imgSrc = warpedUrl || null
            if (!g0.length) return null
            return (
              <section className="result-section">
                <h2 className="result-heading">Unrecognized Graphics Detected</h2>
                <p className="compliance-pending__msg">
                  Residual visual regions not matched to OCR text. Mark brand logos for compliance text-vs-graphic ratio.
                </p>
                <div className="graphic-list">
                  {g0.map((g, i) => (
                    <article key={i} className="graphic-card">
                      {imgSrc ? (
                        <GraphicThumb src={imgSrc} g={g} imageWidth={ocrResult.image_width} imageHeight={ocrResult.image_height} />
                      ) : (
                        <div className="graphic-thumb graphic-thumb--empty">n/a</div>
                      )}
                      <div className="graphic-meta">
                        <div className="graphic-bounds">#{i + 1} ({g.x},{g.y}) {g.w}×{g.h}</div>
                        <label className="graphic-toggle">
                          <input
                            type="checkbox"
                            checked={Boolean(graphicDecisions[i])}
                            onChange={(e) => {
                              const v = e.target.checked
                              setGraphicDecisions((prev) => ({ ...prev, [i]: v }))
                              setAuditSessionComplete(false)
                            }}
                          />
                          <span>Is this a Brand Logo?</span>
                        </label>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )
          })()}

          {ocrResult.gstin.length > 0 && (
            <section className="result-section">
              <h2 className="result-heading">GSTIN<span className="result-count">{ocrResult.gstin.length}</span></h2>
              <ul className="gstin-list">
                {ocrResult.gstin.map((g, i) => (
                  <li key={i} className="gstin-item">
                    <code className="gstin-value">{g.value}</code>
                    {g.checksum_valid ? (
                      <span className="chip chip--sm chip--ok">Format + check digit OK</span>
                    ) : g.format_valid ? (
                      <span className="chip chip--sm chip--warn">Format only</span>
                    ) : (
                      <span className="chip chip--sm chip--warn">Verify manually (OCR)</span>
                    )}
                    {g.block_index != null && (
                      <span className="gstin-block-ref">in text region #{g.block_index + 1}</span>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {ocrResult.signboard_geometry && (() => {
            const g = ocrResult.signboard_geometry
            const dom = g.dominant_language
            const domShare = g.language_ratios[dom]
            const domPct =
              dom && typeof domShare === 'number' && !Number.isNaN(domShare)
                ? (domShare * 100).toFixed(1)
                : '—'
            const pl = g.primary_language
            const prPct = (g.primary_ratio * 100).toFixed(1)
            return (
            <section className="result-section">
              <h2 className="result-heading">Threshed ink (OpenCV) — language share</h2>
              <p className="compliance-pending__msg threshed-ink__line">
                <span className="threshed-ink__pair">
                  <strong>Dominant threshed script</strong> (largest share):{' '}
                  {langName(dom)} ({dom}
                  {domPct !== '—' ? ` — ${domPct}%` : ''}
                  ).
                </span>{' '}
                <span
                  className="threshed-ink__pair"
                  title="The backend PRIMARY_LANGUAGE (often kn for this deployment). Used for the local compliance threshold, not the same as dominant when two scripts are close."
                >
                  <strong>{langName(pl)} (server primary {pl})</strong> threshed ink: {prPct}%.
                </span>
                <strong> Ruleset: {ocrResult.compliance.status}</strong>
                <span
                  className="threshed-ink__agent"
                  title="OpenCV: whether the PRIMARY_LANGUAGE share meets COMPLIANCE_THRESHOLD. Separate from the state/UT result above."
                >
                  {' '}
                  · OpenCV local threshold: {g.compliance_status}
                </span>
              </p>
              <p className="threshed-ink__micro" role="note">
                State rules (e.g. 60% Kannada) use the {pl} threshed share; &quot;dominant&quot; is
                simply whichever script has the most ink in this read (often English on
                mixed boards).
              </p>
            </section>
            )
          })()}

          <section className="result-section">
            <h2 className="result-heading">Detected text</h2>
            {ocrResult.text
              ? (
                <pre
                  className="result-pre"
                  lang={textView === 'english' && hasEnglishView ? 'en' : undefined}
                >
                  {textView === 'english' && hasEnglishView ? fullEnglishText : ocrResult.text}
                </pre>
                )
              : <p className="result-empty">No text detected.</p>}
          </section>

          {/* Languages */}
          {ocrResult.languages.length > 0 && (
            <section className="result-section">
              <h2 className="result-heading">Languages Detected</h2>
              <div className="chip-row">
                {ocrResult.languages.map((l) => (
                  <span key={l.code} className="chip chip--lang">
                    <span className="chip-label">{langName(l.code)}</span>
                    <span className="chip-code">{l.code}</span>
                    <span className="chip-conf">{pct(l.confidence)}</span>
                  </span>
                ))}
              </div>
              {ocrResult.languages.some(
                (l) => (l.code || '').toLowerCase() === 'hmn' || (l.code || '').toLowerCase().startsWith('hmn-'),
              ) && (
                <p className="lang-detect__hint" role="note">
                  Google Vision &ldquo;Languages detected&rdquo; is page-level: it often mis-labels
                  Indic text as <strong>Hmong (hmn)</strong>. Threshed-ink and compliance use
                  per-block script and any language you set in <strong>Text regions</strong>, not
                  this list alone.
                </p>
              )}
            </section>
          )}

          {/* Text blocks */}
          {ocrResult.blocks.length > 0 && (
            <section className="result-section result-section--blocks">
              <h2 className="result-heading">
                Text Regions
                <span className="result-count">{ocrResult.blocks.length}</span>
              </h2>
              <div className="audit-confirm-bar">
                <p className="audit-confirm-bar__hint">
                  {retakeLines.length > 0
                    ? 'Audit confirmation is disabled until you upload a clearer image (see notice above).'
                    : doubtfulBlockIndices.length > 0
                      ? 'Regions with low-confidence OCR, empty AI read with visible ink, or other warnings are highlighted. Open each to review, set language, and save.'
                      : 'No review-required regions; you can still confirm the audit for this result.'}
                </p>
                <div className="audit-confirm-bar__actions">
                  <button
                    type="button"
                    className="btn-audit-confirm"
                    disabled={!canConfirmAudit}
                    onClick={onConfirmAudit}
                    title={
                      !canConfirmAudit
                        ? (retakeLines.length > 0
                          ? 'Upload a clearer image first'
                          : 'Review every highlighted block first')
                        : undefined
                    }
                  >
                    Confirm &amp; Audit
                  </button>
                  {auditSessionComplete && <span className="audit-confirm-bar__ok">Saved locally for this result.</span>}
                </div>
              </div>
              <div className="block-list">
                {ocrResult.blocks.map((blk, i) => {
                  const needsR = blockNeedsReview(blk)
                  const reviewed = reviewedDoubtful.has(i)
                  const displayText = correctionText(blockCorrections[i], blk.text)
                  const blockEn = blk.translated_text || clientTranslations[i]
                  return (
                    <article
                      key={i}
                      className={[
                        'block-card',
                        needsR ? ' block-card--doubtful' : '',
                        reviewed ? ' block-card--doubtful-reviewed' : '',
                        needsR ? ' block-card--doubtful-active' : '',
                      ].join('')}
                      aria-label={
                        needsR
                          ? `Open review for text region ${i + 1} (low-confidence or silent region)`
                          : undefined
                      }
                      onClick={needsR ? () => openAuditModal(i) : undefined}
                      onKeyDown={
                        needsR
                          ? (e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              openAuditModal(i)
                            }
                          }
                          : undefined
                      }
                      role={needsR ? 'button' : undefined}
                      tabIndex={needsR ? 0 : undefined}
                    >
                      <div className="block-meta">
                        {needsR && (
                          <span
                            className="doubtful-pill"
                            title={
                              blk.warning_type === 'SILENT_REGION_DETECTED'
                                ? 'Visible ink but no OCR text; verify manually'
                                : 'This region needs review'
                            }
                            aria-hidden
                          >
                            <span className="doubtful-pill__icon">&#9888;</span>
                            {blk.warning_type === 'SILENT_REGION_DETECTED' ? 'Silent region' : 'Review'}
                          </span>
                        )}
                        <span className="block-index">#{i + 1}</span>
                        {blk.languages.map((l) => (
                          <span key={l.code} className="chip chip--sm chip--lang">
                            {langName(l.code)} {pct(l.confidence)}
                          </span>
                        ))}
                        <AreaInfo norm={blk.norm} bounds={blk.bounds} />
                        {needsR && reviewed && <span className="chip chip--sm chip--reviewed">Reviewed</span>}
                      </div>
                      {blockEn && (
                        <p className="block-translation" lang="en">
                          <em>EN:</em>{' '}
                          {blockEn.length > 180
                            ? blockEn.slice(0, 180) + '…'
                            : blockEn}
                        </p>
                      )}
                      {reviewed ? (
                        <p className="block-text block-text--verified">{displayText}</p>
                      ) : (
                        <BlockTextHighlighted text={blk.text} words={blk.words} />
                      )}
                    </article>
                  )
                })}
              </div>
              {auditModalIndex != null
                && ocrResult.blocks[auditModalIndex]
                && blockNeedsReview(ocrResult.blocks[auditModalIndex]) && (
                <div
                  className="audit-modal-backdrop"
                  onClick={closeAuditModal}
                  role="presentation"
                >
                  <div
                    className="audit-modal"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby="audit-modal-title"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <h3 id="audit-modal-title" className="audit-modal__title">
                      Review text region #
                      {auditModalIndex + 1}
                    </h3>
                    {ocrResult.blocks[auditModalIndex].warning_type === 'SILENT_REGION_DETECTED' && (
                      <p className="audit-modal__alert" role="status">
                        AI missed this text (e.g. low color contrast). Enter it manually and set the
                        language so threshed-ink rules can re-score (e.g. choose Kannada for Karnataka).
                      </p>
                    )}
                    <div className="audit-modal__field">
                      <div className="audit-modal__label">Original script (OCR)</div>
                      <p className="audit-modal__readonly">
                        {ocrResult.blocks[auditModalIndex].text.trim()
                          ? ocrResult.blocks[auditModalIndex].text
                          : '— (empty — threshed ink only)'}
                      </p>
                    </div>
                    <div className="audit-modal__field">
                      <div className="audit-modal__label">AI translation</div>
                      {(ocrResult.blocks[auditModalIndex].translated_text
                        || clientTranslations[auditModalIndex]) ? (
                        <p className="audit-modal__readonly" lang="en">
                          {ocrResult.blocks[auditModalIndex].translated_text
                            || clientTranslations[auditModalIndex]}
                        </p>
                        ) : (
                        <p className="audit-modal__empty">No translation for this line. Use &quot;Translate to English&quot; above.</p>
                        )}
                    </div>
                    <div className="audit-modal__field">
                      <label className="audit-modal__label" htmlFor="audit-block-lang">
                        Block language (manual)
                      </label>
                      <select
                        id="audit-block-lang"
                        className="audit-modal__select"
                        value={auditLang}
                        onChange={(e) => setAuditLang(e.target.value)}
                      >
                        {MANUAL_BCP47_OPTIONS.map((code) => (
                          <option key={code} value={code}>
                            {code === 'und' ? 'Undetermined (und)' : `${langName(code)} (${code})`}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="audit-modal__field">
                      <label className="audit-modal__label" htmlFor="audit-verify-input">
                        {ocrResult.blocks[auditModalIndex].warning_type === 'SILENT_REGION_DETECTED'
                          ? 'Enter the visible line as read by a human (required for silent regions)'
                          : 'Verify or correct the text'}
                      </label>
                      <textarea
                        id="audit-verify-input"
                        className="audit-modal__textarea"
                        value={auditDraft}
                        onChange={(e) => setAuditDraft(e.target.value)}
                        rows={4}
                        autoComplete="off"
                        autoFocus
                        spellCheck
                        placeholder={
                          ocrResult.blocks[auditModalIndex].warning_type === 'SILENT_REGION_DETECTED'
                            ? 'Type the script you see on the sign…'
                            : undefined
                        }
                      />
                    </div>
                    <div className="audit-modal__actions">
                      <button type="button" className="btn-ghost" onClick={closeAuditModal}>
                        Cancel
                      </button>
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={saveAuditModal}
                        disabled={
                          ocrResult.blocks[auditModalIndex].warning_type === 'SILENT_REGION_DETECTED'
                          && !auditDraft.trim()
                        }
                      >
                        Save
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Logos */}
          {ocrResult.logos.length > 0 ? (
            <section className="result-section">
              <h2 className="result-heading">
                Logos Detected
                <span className="result-count">{ocrResult.logos.length}</span>
              </h2>
              <div className="logo-list">
                {ocrResult.logos.map((logo, i) => (
                  <div key={i} className="logo-card">
                    <span className="logo-icon">◉</span>
                    <span className="logo-name">{logo.name}</span>
                    <span className="chip chip--sm chip--conf">{pct(logo.confidence)}</span>
                    <AreaInfo norm={logo.norm} bounds={logo.bounds} />
                  </div>
                ))}
              </div>
            </section>
          ) : ocrStatus === 'success' && (
            <section className="result-section">
              <h2 className="result-heading">Logos</h2>
              <p className="result-empty">No company logos detected.</p>
            </section>
          )}

        </div>
      )}
    </main>
    </div>
  )
}

export default App
