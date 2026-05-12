export type Point = [number, number]

function gaussianElimination(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row
    }
    ;[M[col], M[maxRow]] = [M[maxRow], M[col]]
    const pivot = M[col][col]
    if (Math.abs(pivot) < 1e-10) continue
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const f = M[row][col] / pivot
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k]
    }
  }
  return M.map((row, i) => row[n] / row[i])
}

function computeHomography(src: Point[], dst: Point[]): number[][] {
  const A: number[][] = []
  const b: number[] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i]
    const [u, v] = dst[i]
    A.push([-x, -y, -1, 0, 0, 0, x * u, y * u])
    b.push(-u)
    A.push([0, 0, 0, -x, -y, -1, x * v, y * v])
    b.push(-v)
  }
  const h = gaussianElimination(A, b)
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0],
  ]
}

function applyH(H: number[][], x: number, y: number): Point {
  const w = H[2][0] * x + H[2][1] * y + H[2][2]
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ]
}

function edgeDist(a: Point, b: Point): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

/** Sort 4 points into [TL, TR, BR, BL] order. */
export function sortCorners(pts: Point[]): [Point, Point, Point, Point] {
  const byY = [...pts].sort((a, b) => a[1] - b[1])
  const top = byY.slice(0, 2).sort((a, b) => a[0] - b[0])
  const bot = byY.slice(2, 4).sort((a, b) => a[0] - b[0])
  return [top[0], top[1], bot[1], bot[0]]
}

function estimateOutputSize(corners: [Point, Point, Point, Point]): [number, number] {
  const [tl, tr, br, bl] = corners
  const w = Math.max(1, Math.round(Math.max(edgeDist(tl, tr), edgeDist(bl, br))))
  const h = Math.max(1, Math.round(Math.max(edgeDist(tl, bl), edgeDist(tr, br))))
  return [w, h]
}

/** Mobile Safari: full-frame getImageData + very large output canvases can fail silently. */
const MAX_SRC_PIXELS = 7_000_000
const MAX_OUT_SIDE = 3_200
const MAX_OUT_PIXELS = 4_000_000

function capOutputSize(w: number, h: number): [number, number] {
  let nw = Math.max(1, w)
  let nh = Math.max(1, h)
  const m = Math.max(nw, nh)
  if (m > MAX_OUT_SIDE) {
    const s = MAX_OUT_SIDE / m
    nw = Math.max(1, Math.round(nw * s))
    nh = Math.max(1, Math.round(nh * s))
  }
  if (nw * nh > MAX_OUT_PIXELS) {
    const t = Math.sqrt(MAX_OUT_PIXELS / (nw * nh))
    nw = Math.max(1, Math.round(nw * t))
    nh = Math.max(1, Math.round(nh * t))
  }
  return [nw, nh]
}

/**
 * Perspective-warp `img` so that `corners` (TL,TR,BR,BL in natural px)
 * maps to an axis-aligned rectangle. Returns a new canvas.
 * Downscales internally on large phone photos so mobile browsers can read pixels.
 */
export function warpPerspective(
  img: HTMLImageElement,
  corners: [Point, Point, Point, Point],
): HTMLCanvasElement {
  const natW = Math.max(1, img.naturalWidth)
  const natH = Math.max(1, img.naturalHeight)
  const srcArea = natW * natH
  const srcScale =
    srcArea <= MAX_SRC_PIXELS
      ? 1
      : Math.sqrt(MAX_SRC_PIXELS / srcArea)
  const srcW = Math.max(1, Math.round(natW * srcScale))
  const srcH = Math.max(1, Math.round(natH * srcScale))
  const cornersUse: [Point, Point, Point, Point] =
    srcScale >= 0.999999
      ? corners
      : [
          [corners[0][0] * srcScale, corners[0][1] * srcScale],
          [corners[1][0] * srcScale, corners[1][1] * srcScale],
          [corners[2][0] * srcScale, corners[2][1] * srcScale],
          [corners[3][0] * srcScale, corners[3][1] * srcScale],
        ]
  const sorted = sortCorners(cornersUse)
  const [w0, h0] = estimateOutputSize(sorted)
  const [outW, outH] = capOutputSize(w0, h0)

  const dst: Point[] = [
    [0, 0],
    [outW - 1, 0],
    [outW - 1, outH - 1],
    [0, outH - 1],
  ]
  // Inverse homography: dst pixel → src pixel
  const H = computeHomography(dst, sorted)

  const srcCanvas = document.createElement('canvas')
  srcCanvas.width = srcW
  srcCanvas.height = srcH
  const sctx = srcCanvas.getContext('2d')!
  sctx.imageSmoothingEnabled = true
  sctx.imageSmoothingQuality = 'high'
  sctx.drawImage(img, 0, 0, srcW, srcH)
  const srcData = sctx.getImageData(0, 0, srcW, srcH)

  const dstCanvas = document.createElement('canvas')
  dstCanvas.width = outW
  dstCanvas.height = outH
  const dstCtx = dstCanvas.getContext('2d')!
  const dstData = dstCtx.createImageData(outW, outH)

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const [sx, sy] = applyH(H, x, y)
      const sxi = Math.round(sx)
      const syi = Math.round(sy)
      if (sxi >= 0 && sxi < srcW && syi >= 0 && syi < srcH) {
        const si = (syi * srcW + sxi) * 4
        const di = (y * outW + x) * 4
        dstData.data[di] = srcData.data[si]
        dstData.data[di + 1] = srcData.data[si + 1]
        dstData.data[di + 2] = srcData.data[si + 2]
        dstData.data[di + 3] = srcData.data[si + 3]
      }
    }
  }

  dstCtx.putImageData(dstData, 0, 0)
  return dstCanvas
}
