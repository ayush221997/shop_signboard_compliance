import { useCallback, useEffect, useRef, useState } from 'react'
import { getApiBase } from './apiBase'
import { sortCorners, warpPerspective, type Point } from './perspective'

const API_BASE = getApiBase()

type DetectStatus = 'idle' | 'loading' | 'ok' | 'err'
type ToolMode = 'auto_drag' | 'manual_click'

interface Props {
  file: File
  onWarped: (blob: Blob, previewUrl: string) => void
  /** If set, sent as `state_code` (optional; server uses auto script detection for Vision). */
  stateCode?: string
  apiBase?: string
}

const CORNER_COLORS = ['#ef4444', '#f97316', '#22c55e', '#3b82f6']
const LABELS = ['TL', 'TR', 'BR', 'BL']
const MAX_DISPLAY_HEIGHT = 420
const HANDLE_HIT_PX = 20

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n))
}

function fullImageCorners(natW: number, natH: number): [Point, Point, Point, Point] {
  const w = Math.max(1, natW)
  const h = Math.max(1, natH)
  return [
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1],
  ]
}

function naturalToCanvas(
  p: Point,
  img: HTMLImageElement,
  cW: number,
  cH: number,
): [number, number] {
  return [
    (p[0] / img.naturalWidth) * cW,
    (p[1] / img.naturalHeight) * cH,
  ]
}

function clientToCanvas(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rect: DOMRect,
): [number, number] {
  const sx = canvas.width / rect.width
  const sy = canvas.height / rect.height
  return [(clientX - rect.left) * sx, (clientY - rect.top) * sy]
}

function canvasToNatural(
  canvasX: number,
  canvasY: number,
  img: HTMLImageElement,
  cW: number,
  cH: number,
): Point {
  const nx = (canvasX / cW) * img.naturalWidth
  const ny = (canvasY / cH) * img.naturalHeight
  return [
    clamp(nx, 0, Math.max(0, img.naturalWidth - 1)),
    clamp(ny, 0, Math.max(0, img.naturalHeight - 1)),
  ]
}

export default function CornerCanvas({ file, onWarped, stateCode = '', apiBase = API_BASE }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [corners, setCorners] = useState<Point[]>([])
  const [applying, setApplying] = useState(false)
  const [mode, setMode] = useState<ToolMode>('auto_drag')
  const [detStatus, setDetStatus] = useState<DetectStatus>('idle')
  const [detMessage, setDetMessage] = useState<string>('')
  const [detMethod, setDetMethod] = useState<string | null>(null)
  const [dragI, setDragI] = useState<number | null>(null)
  const dragIRef = useRef<number | null>(null)
  dragIRef.current = dragI
  /** Bumps when a new `Image` finishes loading so the canvas repaints even before /api returns. */
  const [loadEpoch, setLoadEpoch] = useState(0)
  const [applyError, setApplyError] = useState<string | null>(null)

  const runDetect = useCallback(
    async (img: HTMLImageElement) => {
      setDetStatus('loading')
      setDetMessage('Detecting signboard outline from text…')
      setDetMethod(null)
      const fd = new FormData()
      fd.append('file', file, file.name)
      const st = (stateCode || '').trim()
      if (st) fd.append('state_code', st)
      try {
        const r = await fetch(`${apiBase}/api/suggest-corners`, { method: 'POST', body: fd })
        const j: unknown = await r.json().catch(() => ({}))
        if (!r.ok) {
          let m = `Request failed (${r.status})`
          if (typeof j === 'object' && j !== null && 'detail' in j) {
            const d = (j as { detail: unknown }).detail
            if (typeof d === 'string') m = d
          }
          throw new Error(m)
        }
        const d = j as { corners: number[][]; method: string; image_width: number; image_height: number }
        if (Array.isArray(d.corners) && d.corners.length === 4) {
          const pts: Point[] = d.corners.map(
            (c) => [Number(c[0]) || 0, Number(c[1]) || 0] as Point,
          )
          setCorners(pts)
          setMode('auto_drag')
          setDetStatus('ok')
          setDetMethod(d.method)
          if (d.method === 'text_hull') {
            setDetMessage('We estimated corners from the text region. Drag any handle to fix.')
          } else {
            setDetMessage('No text hull — using full image frame. Drag the corners inwards if needed.')
          }
          return
        }
        throw new Error('Invalid response shape')
      } catch (e) {
        setDetStatus('err')
        setDetMessage(
          e instanceof Error
            ? `${e.message} — using the full image frame. Drag corners to adjust.`
            : 'Detection failed. Using the full image frame.',
        )
        setCorners([...fullImageCorners(img.naturalWidth, img.naturalHeight)])
        setMode('auto_drag')
        setDetMethod('full_image_client')
      }
    },
    [apiBase, file, stateCode],
  )

  // Load the image, then run auto-detect
  useEffect(() => {
    imgRef.current = null
    setCorners([])
    setApplying(false)
    setMode('auto_drag')
    setDetStatus('idle')
    setDetMessage('')
    setDetMethod(null)
    setDragI(null)
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      imgRef.current = img
      setLoadEpoch((n) => n + 1)
      URL.revokeObjectURL(url)
      void runDetect(img)
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      setDetStatus('err')
      setDetMessage('Could not load this file as an image.')
    }
    img.src = url
  }, [file, runDetect])

  const redraw = (img: HTMLImageElement, pts: Point[], activeDrag: number | null) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const maxW = (canvas.parentElement?.clientWidth ?? 640) - 4
    const scale = Math.min(maxW / img.naturalWidth, MAX_DISPLAY_HEIGHT / img.naturalHeight, 1)
    const cW = Math.round(img.naturalWidth * scale)
    const cH = Math.round(img.naturalHeight * scale)
    canvas.width = cW
    canvas.height = cH

    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0, cW, cH)

    const toCanvas = (p: Point): [number, number] => naturalToCanvas(p, img, cW, cH)

    if (pts.length >= 2) {
      ctx.save()
      ctx.strokeStyle = 'rgba(99,102,241,0.9)'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      pts.forEach((p, i) => {
        const [cx, cy] = toCanvas(p)
        if (i === 0) ctx.moveTo(cx, cy)
        else ctx.lineTo(cx, cy)
      })
      if (pts.length === 4) ctx.closePath()
      ctx.stroke()
      ctx.restore()
    }

    pts.forEach((p, i) => {
      const [cx, cy] = toCanvas(p)
      const isHot = (activeDrag != null && activeDrag === i) || false
      ctx.save()
      ctx.shadowColor = 'rgba(0,0,0,0.4)'
      ctx.shadowBlur = isHot ? 8 : 4
      ctx.fillStyle = CORNER_COLORS[i]
      ctx.beginPath()
      ctx.arc(cx, cy, 11, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 10px system-ui'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(LABELS[i] ?? String(i + 1), cx, cy)
    })
  }

  // Redraw when corners / drag or image first loads
  useEffect(() => {
    if (imgRef.current) redraw(imgRef.current, corners, dragI)
  }, [corners, dragI, detStatus, loadEpoch])

  const pickHandleIndex = (canvas: HTMLCanvasElement, img: HTMLImageElement, canvasX: number, canvasY: number): number | null => {
    for (let i = corners.length - 1; i >= 0; i--) {
      const [px, py] = naturalToCanvas(corners[i], img, canvas.width, canvas.height)
      if (Math.hypot(canvasX - px, canvasY - py) <= HANDLE_HIT_PX) return i
    }
    return null
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (applying) return
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    e.preventDefault()
    const rect = canvas.getBoundingClientRect()
    const [cx, cy] = clientToCanvas(e.clientX, e.clientY, canvas, rect)

    if (mode === 'manual_click') {
      if (corners.length >= 4) return
      setCorners((prev) => [...prev, canvasToNatural(cx, cy, img, canvas.width, canvas.height)])
      return
    }

    if (corners.length === 4) {
      const hi = pickHandleIndex(canvas, img, cx, cy)
      if (hi == null) return
      setDragI(hi)
      canvas.setPointerCapture(e.pointerId)
    }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const i = dragIRef.current
    if (i == null) return
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const rect = canvas.getBoundingClientRect()
    const [cx, cy] = clientToCanvas(e.clientX, e.clientY, canvas, rect)
    const p = canvasToNatural(cx, cy, img, canvas.width, canvas.height)
    setCorners((prev) => {
      if (i < 0 || i >= prev.length) return prev
      const n = [...prev]
      n[i] = p
      return n
    })
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragI == null) return
    e.preventDefault()
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* not captured */
    }
    setDragI(null)
  }

  const onPointerCancel = (e: React.PointerEvent<HTMLCanvasElement>) => {
    onPointerUp(e)
  }

  const handleApply = useCallback(() => {
    const img = imgRef.current
    if (!img || corners.length < 4) return
    setApplyError(null)
    setApplying(true)
    const finish = (err?: string) => {
      setApplying(false)
      if (err) setApplyError(err)
    }
    const run = () => {
      try {
        const sorted = sortCorners(corners) as [Point, Point, Point, Point]
        const warpedCanvas = warpPerspective(img, sorted)
        const done = (blob: Blob) => {
          const url = URL.createObjectURL(blob)
          onWarped(blob, url)
          finish()
        }
        warpedCanvas.toBlob(
          (blob) => {
            if (blob) {
              done(blob)
              return
            }
            // iOS / some WebKit builds return null for large PNG blobs — fall back
            void (async () => {
              try {
                const d = warpedCanvas.toDataURL('image/png')
                const r = await fetch(d)
                const b = await r.blob()
                if (b.size) done(b)
                else finish('Could not export corrected image. Try a smaller photo or re-crop.')
              } catch {
                finish('Could not export corrected image. Try a smaller photo or re-crop.')
              }
            })()
          },
          'image/png',
        )
      } catch (e) {
        finish(
          e instanceof Error
            ? e.message
            : 'Perspective correction failed. Try “Use full image frame” or a smaller image.',
        )
      }
    }
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(run)
    } else {
      setTimeout(run, 0)
    }
  }, [corners, onWarped])

  const goFullFrame = useCallback(() => {
    setApplyError(null)
    const img = imgRef.current
    if (!img) return
    setCorners([...fullImageCorners(img.naturalWidth, img.naturalHeight)])
    setMode('auto_drag')
    setDetMessage('Full image frame. Drag the corners to match the signboard.')
    setDetMethod('full_image')
    setDetStatus('ok')
  }, [])

  const reDetect = useCallback(() => {
    setApplyError(null)
    const img = imgRef.current
    if (img) void runDetect(img)
  }, [runDetect])

  const goManual = useCallback(() => {
    setApplyError(null)
    setMode('manual_click')
    setCorners([])
    setDetMessage('Click in order: top-left, top-right, bottom-right, bottom-left.')
  }, [])

  const statusLine =
    detStatus === 'loading' ? (detMessage || 'Working…') : detMessage
      || (mode === 'manual_click'
        ? `Click the ${LABELS[corners.length]} corner (${corners.length}/4)`
        : 'Drag the four corners to line up with the signboard, then apply.')

  return (
    <div className="corner-wrap">
      {detStatus === 'loading' && (
        <p className="corner-status corner-status--load" role="status">
          {statusLine}
        </p>
      )}
      {detStatus !== 'loading' && <p className="corner-hint">{statusLine}</p>}

      <canvas
        ref={canvasRef}
        className={
          `corner-canvas${
            mode === 'manual_click' && corners.length < 4 && !applying
              ? ' corner-canvas--picking'
              : ''
          }${corners.length === 4 && mode === 'auto_drag' && !applying ? ' corner-canvas--drag' : ''}`
        }
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onPointerLeave={(e) => { if (dragI != null) onPointerUp(e) }}
      />

      <div className="corner-toolbar" role="toolbar" aria-label="Corner tools">
        <button
          type="button"
          className="btn-ghost btn-sm"
          onClick={reDetect}
          disabled={applying || detStatus === 'loading'}
        >
          Re-detect
        </button>
        <button
          type="button"
          className="btn-ghost btn-sm"
          onClick={goFullFrame}
          disabled={applying || detStatus === 'loading'}
        >
          Use full image frame
        </button>
        <button
          type="button"
          className="btn-ghost btn-sm"
          onClick={goManual}
          disabled={applying || detStatus === 'loading'}
        >
          Place 4 points manually
        </button>
      </div>
      {detStatus === 'ok' && detMethod && (
        <p className="corner-meta" aria-hidden>
          Source:{' '}
          {detMethod === 'text_hull'
            ? 'text region hull (Vision + convex hull)'
            : detMethod === 'full_image_client'
              ? 'client fallback: full image frame'
              : 'full image frame'}
        </p>
      )}

      <div className="corner-btns">
        {applyError && (
          <p className="corner-apply-err" role="alert">
            {applyError}
          </p>
        )}
        {corners.length === 4 && (
          <button
            type="button"
            className="btn-primary"
            onClick={handleApply}
            disabled={applying || detStatus === 'loading'}
          >
            {applying ? 'Applying…' : 'Apply perspective correction'}
          </button>
        )}
      </div>
    </div>
  )
}
