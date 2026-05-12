"""
SignboardGeometryAgent
======================
Pixel-accurate language area measurement for signboard compliance (e.g. COMPLai).

Pipeline
--------
1. Perspective-correct the raw signboard photo into a canonical 1024×1024 canvas
   using OpenCV homography — eliminates camera-angle distortion.
2. Map every Vision API text block into the corrected canvas space.
3. For each language group, draw a binary mask over its blocks, then apply
   cv2.adaptiveThreshold to isolate actual ink pixels (eliminates padding bias).
4. cv2.countNonZero → real pixel weight (bold text counts more than thin text).
5. Derive per-language ratios and PASS/FAIL compliance.

Usage
-----
    agent  = SignboardGeometryAgent(primary_language="kn", compliance_threshold=0.60)
    result = agent.analyze(image_bgr, vision_response, board_corners=corners_array)
    debug  = agent.debug_visualization()          # BGR image for saving / display
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from compliance_rules import normalize_lang
from indic_script_infer import infer_indic_dominant_bcp47

logger = logging.getLogger(__name__)

# Auto corner: drop tiny OCR speckles; optional ignore top-of-frame strip (set >0 to enable)
_AUTO_CORNER_MIN_BLOCK_AREA_FRAC: float = 0.02
_AUTO_CORNER_TOP_STRIP_FRAC: float = 0.0
# Reject min-area-rect that is a hairline or covers almost no pixels (relative to image)
_AUTO_CORNER_MIN_DIM_FRAC: float = 0.03
_AUTO_CORNER_EDGE_AREA_MIN_FRAC: float = 0.10
_AUTO_CORNER_EDGE_AREA_MAX_FRAC: float = 0.95

# ── Canonical canvas dimensions (standardised across all images) ──────────────
TARGET_W = 1024
TARGET_H = 1024
TOTAL_PIXELS = TARGET_W * TARGET_H
# Threshed ink in a block above this % of the canonical board, with no OCR text → silent region
SILENT_INK_PCT_FLOOR: float = 5.0
# If threshed ink exceeds this and Vision reports no Kannada in the block, flag for human review
SILENT_INK_PCT_STRONG: float = 10.0


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class LanguageMetrics:
    code: str
    ink_pixels: int         # raw adaptive-threshold pixel count
    board_ratio: float      # ink_pixels / TOTAL_PIXELS  (vs full canvas)
    text_ratio: float       # ink_pixels / total_ink_pixels (vs all detected text)
    block_count: int


@dataclass
class GeometryResult:
    language_ratios: dict[str, float]            # {lang: text_ratio}
    language_details: dict[str, LanguageMetrics]
    compliance_status: str                        # "PASS" | "FAIL" | "UNCERTAIN"
    confidence_score: float                       # 0–1, penalised for blurry/sparse images
    dominant_language: str
    primary_language: str
    primary_ratio: float                          # primary lang text_ratio
    # Single language ≥ ~99% of threshed ink; used for optional monolingual pass in compliance_rules
    monolingual_single_lang: Optional[str] = None
    # Threshed ink as % of canonical board, one value per Vision page block (order matches page.blocks)
    per_block_threshed_board_pct: list[float] = field(default_factory=list)
    # Contour residuals not sufficiently overlapping OCR text regions (possible graphics/logos)
    detected_graphics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "language_ratios": self.language_ratios,
            "compliance_status": self.compliance_status,
            "confidence_score": self.confidence_score,
            "dominant_language": self.dominant_language,
            "primary_language": self.primary_language,
            "primary_ratio": self.primary_ratio,
            "monolingual_single_lang": self.monolingual_single_lang,
            "per_block_threshed_board_pct": self.per_block_threshed_board_pct,
            "detected_graphics": self.detected_graphics,
            "language_details": {
                lang: {
                    "ink_pixels": m.ink_pixels,
                    "board_ratio": m.board_ratio,
                    "text_ratio": m.text_ratio,
                    "block_count": m.block_count,
                }
                for lang, m in self.language_details.items()
            },
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

class SignboardGeometryAgent:
    """
    Pixel-accurate signboard language area measurement.

    Parameters
    ----------
    primary_language : str
        BCP-47 code of the regionally required language (default "kn" = Kannada).
    compliance_threshold : float
        Minimum ``text_ratio`` of ``primary_language`` for PASS (default 0.60 = 60 %).
    thresh_block_size : int
        Adaptive threshold neighbourhood size (must be odd, default 15).
    thresh_c : int
        Constant subtracted from the weighted mean in adaptive threshold (default 4).
    """

    _PALETTE = [
        (255, 80,  80),   # red-ish
        (80,  220, 80),   # green-ish
        (80,  80,  255),  # blue-ish
        (220, 220, 80),   # yellow-ish
        (220, 80,  220),  # magenta-ish
        (80,  220, 220),  # cyan-ish
        (255, 160, 60),   # orange-ish
        (160, 60,  255),  # violet-ish
    ]

    def __init__(
        self,
        primary_language: str = "kn",
        compliance_threshold: float = 0.60,
        thresh_block_size: int = 15,
        thresh_c: int = 4,
        ink_tag_confidence: float = 0.85,
    ) -> None:
        self.primary_language = primary_language
        self.compliance_threshold = compliance_threshold
        # Vision language_code on a block: ≥ this confidence attributes threshed ink to that
        # BCP-47 code; below this, the same label is still used to avoid a dominant "und" pool
        # (see _resolve_block_ink_lang).
        self._ink_tag_confidence = float(ink_tag_confidence)
        self._block_size = thresh_block_size | 1   # guarantee odd
        self._thresh_c = thresh_c

        # State kept for debug_visualization() after analyze()
        self._warped_bgr: Optional[np.ndarray] = None
        self._warped_gray: Optional[np.ndarray] = None
        self._homography: Optional[np.ndarray] = None
        self._language_groups: dict = {}

    # ── Corner helpers ────────────────────────────────────────────────────────

    @staticmethod
    def sort_corners(pts: np.ndarray) -> np.ndarray:
        """Sort 4 points into [TL, TR, BR, BL] order."""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        by_y = pts[pts[:, 1].argsort()]
        top = by_y[:2][by_y[:2, 0].argsort()]
        bot = by_y[2:][by_y[2:, 0].argsort()]
        return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

    @staticmethod
    def _aabb_block_area_cy(
        block,
    ) -> Optional[tuple[float, float]]:
        """(area, center_y) from Vision block AABB, or None."""
        verts = list(block.bounding_box.vertices)
        if len(verts) < 2:
            return None
        xs = [float(v.x) for v in verts]
        ys = [float(v.y) for v in verts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w < 1.0 or h < 1.0:
            return None
        return (w * h, float(sum(ys) / len(ys)))

    @staticmethod
    def _auto_corners_hull_from_points(pts_xy: np.ndarray) -> Optional[np.ndarray]:
        """Legacy: 4 "corners" as extremes on the convex hull (brittle; used as fallback)."""
        if pts_xy.shape[0] < 4:
            return None
        arr = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(arr)
        if hull is None or len(hull) < 4:
            return None
        h2 = hull.squeeze()
        if h2.ndim < 2 or len(h2) < 4:
            return None
        sums = h2[:, 0] + h2[:, 1]
        diffs = h2[:, 0] - h2[:, 1]
        tl = h2[sums.argmin()]
        br = h2[sums.argmax()]
        tr = h2[diffs.argmax()]
        bl = h2[diffs.argmin()]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _quad_area(quad: np.ndarray) -> float:
        q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        return float(abs(cv2.contourArea(q.reshape(-1, 1, 2))))

    @staticmethod
    def _aabb_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = (aw * ah) + (bw * bh) - inter
        return (inter / ua) if ua > 0.0 else 0.0

    @staticmethod
    def auto_corners(vision_response) -> Optional[np.ndarray]:
        """
        Estimate 4 signboard corners in **image pixel** space for perspective warp.

        Uses **per-block** filtering (removes very small text regions), optional drop of
        blocks in the top strip of the image (sky / out-of-FOV labels), then the
        **minimum-area rotated rectangle** (``minAreaRect``) around the remaining
        block vertices. That is more stable than taking four arbitrary hull vertices,
        which often mis-frames the board when text is L-shaped or when stray words sit
        far from the main sign.
        """
        pages = list(vision_response.full_text_annotation.pages)
        if not pages:
            return None
        page = pages[0]
        img_w = float(getattr(page, "width", 0) or 0)
        img_h = float(getattr(page, "height", 0) or 0)

        block_areas: list[float] = []
        for block in page.blocks:
            t = SignboardGeometryAgent._aabb_block_area_cy(block)
            if t is not None:
                block_areas.append(t[0])
        max_area = max(block_areas) if block_areas else 0.0
        min_area = max(_AUTO_CORNER_MIN_BLOCK_AREA_FRAC * max_area, 1.0) if max_area else 0.0
        top_y_cut = _AUTO_CORNER_TOP_STRIP_FRAC * img_h if img_h > 0.0 else -1.0

        all_pts: list[list[float]] = []
        for block in page.blocks:
            t = SignboardGeometryAgent._aabb_block_area_cy(block)
            if t is None:
                continue
            for v in block.bounding_box.vertices:
                all_pts.append([float(v.x), float(v.y)])

        kept_pts: list[list[float]] = []
        for block in page.blocks:
            t = SignboardGeometryAgent._aabb_block_area_cy(block)
            if t is None or t[0] < min_area:
                continue
            _area, cy = t
            if top_y_cut >= 0.0 and cy < top_y_cut:
                continue
            for v in block.bounding_box.vertices:
                kept_pts.append([float(v.x), float(v.y)])

        if len(all_pts) < 4:
            if len(all_pts) > 0:
                return SignboardGeometryAgent._auto_corners_hull_from_points(
                    np.asarray(all_pts, dtype=np.float32)
                )
            return None

        use_pts = kept_pts if len(kept_pts) >= 4 else all_pts
        pxy = np.asarray(use_pts, dtype=np.float32)

        rect = cv2.minAreaRect(pxy)
        w_rect, h_rect = float(rect[1][0]), float(rect[1][1])
        if w_rect < 1.0 or h_rect < 1.0:
            return SignboardGeometryAgent._auto_corners_hull_from_points(
                np.asarray(all_pts, dtype=np.float32)
            )
        if img_w > 0.0 and img_h > 0.0:
            short = min(w_rect, h_rect)
            if short < _AUTO_CORNER_MIN_DIM_FRAC * min(img_w, img_h):
                return SignboardGeometryAgent._auto_corners_hull_from_points(
                    np.asarray(all_pts, dtype=np.float32)
                )

        box = cv2.boxPoints(rect)
        if box is None or len(box) < 4:
            return SignboardGeometryAgent._auto_corners_hull_from_points(
                np.asarray(all_pts, dtype=np.float32)
            )
        return SignboardGeometryAgent.sort_corners(box.astype(np.float32))

    @staticmethod
    def auto_corners_from_image(image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Edge/contour-based board corner estimate from raster image alone.
        Prefers large quadrilaterals near signboard-like rectangular extents.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area = float(max(1, h * w))
        min_area = _AUTO_CORNER_EDGE_AREA_MIN_FRAC * img_area
        max_area = _AUTO_CORNER_EDGE_AREA_MAX_FRAC * img_area
        best_quad: Optional[np.ndarray] = None
        best_score = -1.0
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri <= 1.0:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if approx is None or len(approx) < 4:
                continue

            if len(approx) > 4:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.float32)
            else:
                box = approx.reshape(-1, 2).astype(np.float32)
            if box.shape[0] != 4:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.float32)
            if box.shape[0] != 4:
                continue

            q = SignboardGeometryAgent.sort_corners(box)
            a = SignboardGeometryAgent._quad_area(q)
            if a < min_area or a > max_area:
                continue

            # Prefer larger, less-skewed quads.
            x, y, bw, bh = cv2.boundingRect(q.astype(np.int32))
            ar = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            skew_penalty = 0.15 if ar > 8.0 else 0.0
            score = (a / img_area) - skew_penalty
            if score > best_score:
                best_score = score
                best_quad = q

        return best_quad

    @staticmethod
    def auto_corners_hybrid(image_bgr: np.ndarray, vision_response) -> Optional[np.ndarray]:
        """
        Hybrid corner estimator: fuse OCR-text geometry with image edges.
        Picks image quad when it broadly contains text-hull and has good support.
        """
        text_quad = SignboardGeometryAgent.auto_corners(vision_response)
        img_quad = SignboardGeometryAgent.auto_corners_from_image(image_bgr)

        if text_quad is None and img_quad is None:
            return None
        if text_quad is None:
            return img_quad
        if img_quad is None:
            return text_quad

        tx, ty, tw, th = cv2.boundingRect(text_quad.astype(np.int32))
        ix, iy, iw, ih = cv2.boundingRect(img_quad.astype(np.int32))
        iou = SignboardGeometryAgent._aabb_iou(
            (float(tx), float(ty), float(tw), float(th)),
            (float(ix), float(iy), float(iw), float(ih)),
        )
        t_area = SignboardGeometryAgent._quad_area(text_quad)
        i_area = SignboardGeometryAgent._quad_area(img_quad)

        # If image-based quad is much larger and still overlaps text significantly,
        # it is typically closer to real signboard boundaries than text hull.
        if iou >= 0.20 and i_area > (1.25 * t_area):
            return img_quad
        # If two methods agree enough, trust image boundary estimate.
        if iou >= 0.45:
            return img_quad
        return text_quad

    # ── Perspective correction ────────────────────────────────────────────────

    def _warp(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (warped_bgr [TARGET_W × TARGET_H], homography_matrix)."""
        dst = np.array(
            [[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(
            image, M, (TARGET_W, TARGET_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
        )
        return warped, M

    # ── Block → warped-space mapping ──────────────────────────────────────────

    @staticmethod
    def _map_poly(vertices, M: np.ndarray) -> np.ndarray:
        """Transform Vision API bounding_box vertices through the homography."""
        pts = np.asarray([[v.x, v.y] for v in vertices], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M)
        clipped = np.clip(warped.reshape(-1, 2), 0, [TARGET_W - 1, TARGET_H - 1])
        return clipped.astype(np.int32)

    @staticmethod
    def _vision_best_lang_with_conf(block) -> Tuple[str, float, Any]:
        """
        Return (normalised base code, confidence, raw best language entry).
        Empty list → ('und', 0.0, None).
        """
        prop = getattr(block, "property", None)
        langs = list(prop.detected_languages) if prop and prop.detected_languages else []
        if not langs:
            return ("und", 0.0, None)
        best = max(langs, key=lambda l: float(l.confidence or 0.0))
        code = (best.language_code or "").strip()
        conf = float(best.confidence or 0.0)
        b = normalize_lang(code) if code else "und"
        if not b or b == "und":
            return ("und", conf, best)
        return (b, conf, best)

    def _resolve_block_ink_lang(self, block) -> str:
        """
        Map each text block's threshed pixels to a language key.

        When Vision reports a non-und base code, all threshed pixels for that block's polygon
        are added to that language's ink total (OCR + geometry are aligned for that region).

        If Vision only returns ``und`` (common for Kannada/Indic in the wild), we infer
        a BCP-47 code from Unicode code points in the line text (see indic_script_infer).
        """
        b, _c, _ro = self._vision_best_lang_with_conf(block)
        if b and b != "und":
            return b
        plain = self._block_plain_text(block)
        inferred = infer_indic_dominant_bcp47(plain)
        if inferred:
            return normalize_lang(inferred)
        return "und"

    @staticmethod
    def _word_plain_text(word) -> str:
        return "".join(s.text for s in word.symbols)

    @staticmethod
    def _vision_best_lang_word_with_conf(word) -> Tuple[str, float, Any]:
        prop = getattr(word, "property", None)
        langs = list(prop.detected_languages) if prop and prop.detected_languages else []
        if not langs:
            return ("und", 0.0, None)
        best = max(langs, key=lambda l: float(l.confidence or 0.0))
        code = (best.language_code or "").strip()
        conf = float(best.confidence or 0.0)
        b = normalize_lang(code) if code else "und"
        if not b or b == "und":
            return ("und", conf, best)
        return (b, conf, best)

    def _resolve_word_ink_lang(self, word, block) -> str:
        b, _c, _o = self._vision_best_lang_word_with_conf(word)
        if b and b != "und":
            return b
        wplain = self._word_plain_text(word)
        inferred = infer_indic_dominant_bcp47(wplain)
        if inferred:
            return normalize_lang(inferred)
        return self._resolve_block_ink_lang(block)

    @staticmethod
    def _page_has_non_empty_words(vision_response) -> bool:
        for page in vision_response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        if SignboardGeometryAgent._word_plain_text(word).strip():
                            return True
        return False

    @staticmethod
    def _block_lang(block) -> str:
        """Legacy: best Vision language; prefer _resolve_block_ink_lang in analyze()."""
        b, _c, _o = SignboardGeometryAgent._vision_best_lang_with_conf(block)
        return b if b != "und" else "und"

    def _group_blocks(self, vision_response, M: np.ndarray) -> dict:
        """Return {lang_code: {"polygons": [np.int32 arrays], "count": int}} (matches ink)."""
        groups: dict = {}
        for page in vision_response.full_text_annotation.pages:
            for block in page.blocks:
                if not self._block_plain_text(block).strip():
                    continue
                lang = self._resolve_block_ink_lang(block)
                poly = self._map_poly(block.bounding_box.vertices, M)
                groups.setdefault(lang, {"polygons": [], "count": 0})
                groups[lang]["polygons"].append(poly)
                groups[lang]["count"] += 1
        return groups

    def _group_ink_by_words(self, vision_response, M: np.ndarray) -> dict:
        """
        One polygon per **word** with language from word property → script → block.
        Sums threshed ink in each box, reducing whole-block mis-attribution on multilingual lines.
        """
        groups: dict = {}
        for page in vision_response.full_text_annotation.pages:
            for block in page.blocks:
                if not self._block_plain_text(block).strip():
                    continue
                for para in block.paragraphs:
                    for word in para.words:
                        wplain = self._word_plain_text(word)
                        if not wplain.strip():
                            continue
                        lang = self._resolve_word_ink_lang(word, block)
                        bb = word.bounding_box
                        if not bb or not list(bb.vertices):
                            continue
                        poly = self._map_poly(bb.vertices, M)
                        groups.setdefault(lang, {"polygons": [], "count": 0})
                        groups[lang]["polygons"].append(poly)
                        groups[lang]["count"] += 1
        return groups

    # ── Threshed pixel counting ───────────────────────────────────────────────

    def _ink_pixels_in_poly(self, gray: np.ndarray, polygon: np.ndarray) -> int:
        """
        Count ink pixels inside ``polygon`` on the grayscale warped image.

        1. Fill polygon → binary mask.
        2. Crop bounding rect for speed.
        3. Adaptive Gaussian threshold (BINARY_INV: dark ink on light bg).
        4. AND with polygon mask → countNonZero.

        Bold / thick glyphs yield more pixels than thin ones — no bounding-box
        padding is counted.  Uneven lighting handled by adaptive (local) method.
        """
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        x, y, bw, bh = cv2.boundingRect(polygon)
        x, y = max(0, x), max(0, y)
        x2 = min(gray.shape[1], x + bw)
        y2 = min(gray.shape[0], y + bh)
        if x2 <= x or y2 <= y:
            return 0

        roi_gray = gray[y:y2, x:x2]
        roi_mask = mask[y:y2, x:x2]

        threshed = cv2.adaptiveThreshold(
            roi_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,   # ink = dark foreground
            blockSize=self._block_size,
            C=self._thresh_c,
        )
        threshed = cv2.bitwise_and(threshed, threshed, mask=roi_mask)
        return int(cv2.countNonZero(threshed))

    @staticmethod
    def _rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        ua = float(aw * ah + bw * bh - inter)
        return (inter / ua) if ua > 0 else 0.0

    @staticmethod
    def _rect_inter_over_candidate(cand: tuple[int, int, int, int], text: tuple[int, int, int, int]) -> float:
        cx, cy, cw, ch = cand
        tx, ty, tw, th = text
        cx2, cy2 = cx + cw, cy + ch
        tx2, ty2 = tx + tw, ty + th
        ix1, iy1 = max(cx, tx), max(cy, ty)
        ix2, iy2 = min(cx2, tx2), min(cy2, ty2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        cand_area = float(max(1, cw * ch))
        return inter / cand_area

    def _detect_graphic_residuals(
        self,
        warped_gray: np.ndarray,
        vision_response,
        M: np.ndarray,
        overlap_thr: float = 0.20,
    ) -> list[dict[str, Any]]:
        """
        Find residual contour regions from global threshed ink that do not overlap text boxes
        by more than `overlap_thr` of candidate area.
        """
        th = cv2.adaptiveThreshold(
            warped_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._block_size,
            self._thresh_c,
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

        # Build a precise text mask from OCR polygons (prefer words over whole blocks).
        text_boxes: list[tuple[int, int, int, int]] = []
        text_mask = np.zeros_like(th, dtype=np.uint8)
        for page in vision_response.full_text_annotation.pages:
            for block in page.blocks:
                txt = self._block_plain_text(block)
                if not txt.strip():
                    continue
                had_words = False
                for para in block.paragraphs:
                    for word in para.words:
                        bb = getattr(word, "bounding_box", None)
                        if not bb or not list(getattr(bb, "vertices", []) or []):
                            continue
                        wtxt = self._word_plain_text(word)
                        if not wtxt.strip():
                            continue
                        had_words = True
                        wpoly = self._map_poly(bb.vertices, M)
                        cv2.fillPoly(text_mask, [wpoly], 255)
                        x, y, w, h = cv2.boundingRect(wpoly)
                        if w > 1 and h > 1:
                            text_boxes.append((int(x), int(y), int(w), int(h)))
                if not had_words:
                    poly = self._map_poly(block.bounding_box.vertices, M)
                    cv2.fillPoly(text_mask, [poly], 255)
                    x, y, w, h = cv2.boundingRect(poly)
                    if w > 1 and h > 1:
                        text_boxes.append((int(x), int(y), int(w), int(h)))

        # Expand text regions a bit to absorb anti-aliased edges and tight neighboring glyphs.
        tk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, tk, iterations=1)

        # Residual binary = threshed foreground minus OCR text mask.
        residual = cv2.bitwise_and(th, cv2.bitwise_not(text_mask))
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, k, iterations=1)
        residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(residual, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: list[dict[str, Any]] = []
        seen: list[tuple[int, int, int, int]] = []
        min_area = int(TOTAL_PIXELS * 0.0002)  # ~0.02% of board (capture smaller logos)
        max_area = int(TOTAL_PIXELS * 0.40)    # avoid giant board-level blobs
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 2 or h <= 2:
                continue
            area = int(w * h)
            if area < min_area or area > max_area:
                continue

            cand = (int(x), int(y), int(w), int(h))
            max_inter = 0.0
            for tb in text_boxes:
                max_inter = max(max_inter, self._rect_inter_over_candidate(cand, tb))
                if max_inter > overlap_thr:
                    break
            if max_inter > overlap_thr:
                continue

            if any(self._rect_iou(cand, s) > 0.7 for s in seen):
                continue
            seen.append(cand)

            roi = residual[y:y + h, x:x + w]
            ink_px = int(cv2.countNonZero(roi))
            if ink_px <= 0:
                continue
            # Ignore extremely sparse noise components.
            fill = ink_px / float(max(1, area))
            if fill < 0.05:
                continue
            out.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "type": "unknown_graphic",
                    "threshed_ink_share": round(float(ink_px) / float(TOTAL_PIXELS), 5),
                    "bbox_area_share": round(float(area) / float(TOTAL_PIXELS), 5),
                }
            )
        out.sort(key=lambda g: float(g.get("threshed_ink_share", 0.0)), reverse=True)
        return out

    # ── Main analysis ─────────────────────────────────────────────────────────

    def analyze(
        self,
        image_bgr: np.ndarray,
        vision_response,
        board_corners: Optional[np.ndarray] = None,
    ) -> GeometryResult:
        """
        Parameters
        ----------
        image_bgr : np.ndarray
            BGR image decoded with cv2 (or cv2.imdecode from bytes).
        vision_response :
            google.cloud.vision AnnotateImageResponse (must include
            DOCUMENT_TEXT_DETECTION).
        board_corners : np.ndarray, optional
            Float32 (4×2) array [TL, TR, BR, BL] in *original* image pixel
            coordinates.  If None, corners are auto-detected from text blocks.

        Returns
        -------
        GeometryResult
        """
        # ── 1. Resolve board corners ─────────────────────────────────────────
        if board_corners is not None:
            corners = self.sort_corners(board_corners)
        else:
            corners = self.auto_corners_hybrid(image_bgr, vision_response)
            if corners is None:
                logger.warning("No corners detected; using full image as board.")
                h, w = image_bgr.shape[:2]
                corners = np.array(
                    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float32,
                )

        # ── 2. Perspective correction → canonical 1024×1024 ───────────────────
        warped, M = self._warp(image_bgr, corners)
        self._warped_bgr  = warped
        self._warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self._homography  = M

        # ── 3. Per-word (preferred) or per-block threshed regions in warped space ─
        if self._page_has_non_empty_words(vision_response):
            self._language_groups = self._group_ink_by_words(vision_response, M)
        else:
            self._language_groups = self._group_blocks(vision_response, M)

        # ── 4. Count threshed ink pixels per language ────────────────────────
        raw_ink: dict[str, int] = {}
        block_counts: dict[str, int] = {}
        for lang, data in self._language_groups.items():
            raw_ink[lang] = sum(
                self._ink_pixels_in_poly(self._warped_gray, poly)
                for poly in data["polygons"]
            )
            block_counts[lang] = data["count"]

        total_ink = max(1, sum(raw_ink.values()))  # avoid /0

        # ── 5. Build metrics ─────────────────────────────────────────────────
        details: dict[str, LanguageMetrics] = {
            lang: LanguageMetrics(
                code=lang,
                ink_pixels=ink,
                board_ratio=round(ink / TOTAL_PIXELS, 4),
                text_ratio=round(ink / total_ink, 4),
                block_count=block_counts[lang],
            )
            for lang, ink in raw_ink.items()
        }

        # language_ratios uses text_ratio — share of the detected ink, not board
        language_ratios = {lang: m.text_ratio for lang, m in details.items()}
        primary_ratio   = language_ratios.get(self.primary_language, 0.0)
        # Ties: prefer a named language over "und" so later checks do not read "und" as dominant
        if language_ratios:
            mval = max(language_ratios.values())
            at_max = [k for k, v in language_ratios.items() if abs(v - mval) < 1e-6]
            named = [k for k in at_max if k != "und"]
            dominant = (named[0] if named else (at_max[0] if at_max else "und"))
        else:
            dominant = "und"
        # ≈100% of threshed ink in a single *named* script; allow ≤1% total in other named scripts
        monolingual_single: Optional[str] = None
        non_und = {k: v for k, v in language_ratios.items() if k != "und"}
        if non_und:
            top, tv = max(non_und.items(), key=lambda x: x[1])
            other_named = sum(v for k, v in non_und.items() if k != top)
            if other_named < 0.01 and tv >= 0.99:
                monolingual_single = top

        # Confidence: penalise if total ink is below a reasonable floor
        # (blurry photo, mostly-blank image, failed OCR, etc.)
        expected_floor = TOTAL_PIXELS * 0.005   # at least 0.5 % of canvas should be ink
        confidence = round(min(1.0, total_ink / expected_floor), 3)

        if confidence < 0.30:
            status = "UNCERTAIN"
        else:
            status = "PASS" if primary_ratio >= self.compliance_threshold else "FAIL"

        _pages0 = list(vision_response.full_text_annotation.pages)
        _page_blocks = list(_pages0[0].blocks) if _pages0 else []
        per_block_pcts = (
            self.per_block_threshed_pcts_including_silent(_page_blocks) if _page_blocks else []
        )
        detected_graphics = self._detect_graphic_residuals(self._warped_gray, vision_response, M)

        return GeometryResult(
            language_ratios=language_ratios,
            language_details=details,
            compliance_status=status,
            confidence_score=confidence,
            dominant_language=dominant,
            primary_language=self.primary_language,
            primary_ratio=round(primary_ratio, 4),
            monolingual_single_lang=monolingual_single,
            per_block_threshed_board_pct=per_block_pcts,
            detected_graphics=detected_graphics,
        )

    @staticmethod
    def _block_plain_text(block) -> str:
        lines: list[str] = []
        for para in block.paragraphs:
            wlist: list[str] = []
            for word in para.words:
                wlist.append("".join(s.text for s in word.symbols))
            if wlist:
                lines.append(" ".join(wlist))
        return "\n".join(lines)

    def per_block_threshed_board_pct(self, vision_response) -> list[float]:
        """
        After ``analyze()``, threshed ink as % of the 1024×1024 canonical board for
        each non-empty block (same order as a typical main.py block loop).
        """
        if self._warped_gray is None or self._homography is None:
            raise RuntimeError("Call analyze() before per_block_threshed_board_pct()")
        M = self._homography
        pcts: list[float] = []
        for page in vision_response.full_text_annotation.pages:
            for block in page.blocks:
                if not self._block_plain_text(block).strip():
                    continue
                poly = self._map_poly(block.bounding_box.vertices, M)
                ink = self._ink_pixels_in_poly(self._warped_gray, poly)
                pcts.append(100.0 * ink / TOTAL_PIXELS)
        return pcts

    def per_block_threshed_pcts_including_silent(self, page_blocks: list) -> list[float]:
        """
        Threshed ink as % of the canonical board, one value per ``page_blocks`` entry
        in Vision order — includes blocks where OCR text is empty (e.g. low-contrast sign).
        """
        if self._warped_gray is None or self._homography is None:
            raise RuntimeError("Call analyze() before per_block_threshed_pcts_including_silent()")
        M = self._homography
        pcts: list[float] = []
        for block in page_blocks:
            bb = block.bounding_box
            if not bb or not list(getattr(bb, "vertices", []) or []):
                pcts.append(0.0)
                continue
            poly = self._map_poly(block.bounding_box.vertices, M)
            ink = self._ink_pixels_in_poly(self._warped_gray, poly)
            pcts.append(100.0 * ink / TOTAL_PIXELS)
        return pcts

    # ── Debug visualisation ───────────────────────────────────────────────────

    def debug_visualization(self) -> Optional[np.ndarray]:
        """
        Build a BGR debug image of the perspective-corrected canvas showing:

        • Per-language polygon outlines and semi-transparent fills.
        • Adaptive-threshold ink pixels coloured per language.
        • Language code labels at block centroids.

        Returns None if ``analyze()`` has not been called yet.
        """
        if self._warped_bgr is None or self._warped_gray is None:
            logger.warning("debug_visualization() called before analyze().")
            return None

        canvas  = self._warped_bgr.copy()
        overlay = canvas.copy()
        ink_vis = np.zeros_like(canvas)

        for idx, (lang, data) in enumerate(self._language_groups.items()):
            colour = self._PALETTE[idx % len(self._PALETTE)]

            for poly in data["polygons"]:
                # Semi-transparent polygon fill
                cv2.fillPoly(overlay, [poly], colour)
                # Solid polygon outline on canvas
                cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], True, colour, 2,
                              lineType=cv2.LINE_AA)

                # Per-polygon threshed ink pixels, coloured by language
                x, y, bw, bh = cv2.boundingRect(poly)
                x, y = max(0, x), max(0, y)
                x2 = min(self._warped_gray.shape[1], x + bw)
                y2 = min(self._warped_gray.shape[0], y + bh)
                if x2 <= x or y2 <= y:
                    continue

                pm = np.zeros(self._warped_gray.shape, dtype=np.uint8)
                cv2.fillPoly(pm, [poly], 255)
                th = cv2.adaptiveThreshold(
                    self._warped_gray[y:y2, x:x2],
                    255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, self._block_size, self._thresh_c,
                )
                th_masked = cv2.bitwise_and(th, th, mask=pm[y:y2, x:x2])
                ink_here = th_masked > 0
                for c, val in enumerate(colour):
                    ink_vis[y:y2, x:x2, c] = np.where(
                        ink_here, val, ink_vis[y:y2, x:x2, c]
                    )

            # Language label at centroid of first polygon
            fp = data["polygons"][0]
            cx, cy = int(fp[:, 0].mean()), int(fp[:, 1].mean())
            cv2.putText(
                canvas, lang,
                (max(0, cx - 15), max(16, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA,
            )

        # Blend semi-transparent polygon fill
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Overlay ink pixels
        ink_mask = ink_vis.sum(axis=2) > 0
        canvas[ink_mask] = (
            canvas[ink_mask].astype(np.float32) * 0.35
            + ink_vis[ink_mask].astype(np.float32) * 0.65
        ).clip(0, 255).astype(np.uint8)

        return canvas


# ── Standalone helper ─────────────────────────────────────────────────────────

def save_debug_mask(agent: SignboardGeometryAgent, path: str = "debug_mask.png") -> bool:
    """Save the debug visualisation to disk.  Returns True on success."""
    img = agent.debug_visualization()
    if img is None:
        logger.error("No debug image available — call analyze() first.")
        return False
    ok = cv2.imwrite(path, img)
    if ok:
        logger.info("Debug mask saved to %s", path)
    return ok
