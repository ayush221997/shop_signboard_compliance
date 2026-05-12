/**
 * 90° rotation steps for preview + upload: canvas → PNG.
 */

/**
 * @param direction `cw` = 90° clockwise, `ccw` = 90° counter‑clockwise (view).
 */
export async function rotateImageBlob(
  blob: Blob,
  direction: 'cw' | 'ccw',
): Promise<Blob> {
  const bmp = await createImageBitmap(blob)
  const w = bmp.width
  const h = bmp.height
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    bmp.close()
    throw new Error('2D context unavailable')
  }
  try {
    canvas.width = h
    canvas.height = w
    if (direction === 'cw') {
      ctx.translate(h, 0)
      ctx.rotate(Math.PI / 2)
    } else {
      ctx.translate(0, w)
      ctx.rotate(-Math.PI / 2)
    }
    ctx.drawImage(bmp, 0, 0)
  } finally {
    bmp.close()
  }
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('encode failed'))),
      'image/png',
    )
  })
}
