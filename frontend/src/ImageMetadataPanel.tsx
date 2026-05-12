import type { ImageMetadata } from './ocrTypes'

function formatBytes(n: number) {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function exifValue(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (Array.isArray(v)) return v.map((x) => exifValue(x)).join(', ')
  return String(v)
}

/**
 * Renders `image_metadata` from `/api/ocr` (file info, optional EXIF, Vision dimensions).
 */
export function ImageMetadataPanel({ meta }: { meta: ImageMetadata | null | undefined }) {
  if (!meta || typeof meta !== 'object') return null

  const f = meta.file
  const vis = meta.vision
  const r = meta.raster
  const pdf = meta.pdf
  if (!f && !vis && !r && !pdf) return null

  const exif =
    r && typeof r === 'object' && 'exif' in r && r.exif && typeof r.exif === 'object'
      ? (r.exif as Record<string, unknown>)
      : null

  return (
    <section className="result-section result-section--meta" aria-label="Image metadata">
      <h2 className="result-heading">Image metadata</h2>
      <dl className="meta-dl">
        {f && (
          <>
            <dt>File name</dt>
            <dd className="meta-dl__mono">{f.name || '—'}</dd>
            <dt>Size</dt>
            <dd>
              {formatBytes(f.size_bytes ?? 0)} (
              {typeof f.size_bytes === 'number' ? f.size_bytes.toLocaleString() : '—'} bytes)
            </dd>
            <dt>Declared type</dt>
            <dd className="meta-dl__mono">{f.content_type || '—'}</dd>
          </>
        )}
        {vis && (
          <>
            <dt>Dimensions (OCR / Vision)</dt>
            <dd>
              {vis.width} × {vis.height} px
            </dd>
          </>
        )}
        {r && typeof r === 'object' && 'width_px' in r && r.width_px != null && (
          <>
            <dt>Dimensions (decoded)</dt>
            <dd>
              {Number(r.width_px)} × {Number((r as { height_px?: number }).height_px)} px
              {typeof (r as { color_mode?: string }).color_mode === 'string' && (
                <span className="meta-dl__sub"> · {String((r as { color_mode: string }).color_mode)}</span>
              )}
              {typeof (r as { format?: string }).format === 'string' && (
                <span className="meta-dl__sub"> · {String((r as { format: string }).format)}</span>
              )}
            </dd>
          </>
        )}
        {pdf?.is_pdf && <dt>Format</dt>}
        {pdf?.is_pdf && <dd>PDF (text layer / render used for OCR — no EXIF on file)</dd>}
        {r && typeof r === 'object' && 'error' in r && (
          <>
            <dt>Decode</dt>
            <dd className="meta-dl__warn">{String((r as { error: string }).error)}</dd>
          </>
        )}
        {r && typeof r === 'object' && 'note' in r && (r as { note?: string }).note && (
          <>
            <dt>Note</dt>
            <dd>{String((r as { note: string }).note)}</dd>
          </>
        )}
      </dl>
      {exif && Object.keys(exif).length > 0 && (
        <details className="meta-exif">
          <summary>EXIF / embedded tags</summary>
          <dl className="meta-dl meta-dl--exif">
            {Object.entries(exif).map(([k, v]) => (
              <div key={k} className="meta-exif__row">
                <dt title={k}>{k}</dt>
                <dd>{exifValue(v)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </section>
  )
}
