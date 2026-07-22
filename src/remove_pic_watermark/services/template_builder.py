"""Build watermark templates from a user-selected ROI.

Design notes (aligned with common inpainting practice: IOPaint / LaMa workflows):
- Coarse masks that fully cover the watermark work better than incomplete thin edges.
- Semi-transparent overlays are extracted via tophat + pale/bright color cues.
- If extraction is too sparse, fall back to a filled envelope of the ROI so LaMa still gets a solid region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..models import BBox


@dataclass
class TemplateBuildResult:
    template_mask: np.ndarray
    sample_crop_bgr: np.ndarray
    preview_overlay_bgr: np.ndarray
    bbox: BBox
    image_size: tuple[int, int]  # width, height
    stats: dict[str, Any]
    suggested_detector: dict[str, Any]


def _clamp_bbox(x: int, y: int, w: int, h: int, width: int, height: int) -> BBox:
    x1 = max(0, min(x, width - 1))
    y1 = max(0, min(y, height - 1))
    x2 = max(x1 + 1, min(x + w, width))
    y2 = max(y1 + 1, min(y + h, height))
    return BBox(x1, y1, x2 - x1, y2 - y1)


def extract_mask_from_crop(
    crop_bgr: np.ndarray,
    *,
    dilate: int = 1,
    min_fill_ratio: float = 0.04,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract binary template mask from a crop (same HxW, no trim).

    For pale / semi-transparent text (e.g. white \"Monica\" on light flowers),
    **do not use GrabCut** — FG/BG colors are nearly identical and GrabCut
    collapses to empty or full blobs. Use multi-scale morphological tophat /
    blackhat residual instead (standard approach for translucent overlays).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop is empty")

    mask, method, fill_ratio = _smart_extract(crop_bgr, min_fill_ratio=min_fill_ratio)
    mask, fill_ratio = _finalize_template_mask(mask, dilate=dilate)
    stats = {
        "fill_ratio": round(fill_ratio, 4),
        "method": method,
        "template_size": [int(mask.shape[1]), int(mask.shape[0])],
    }
    return mask, stats


def extract_mask_with_points(
    crop_bgr: np.ndarray,
    points_xy: list[tuple[float, float]] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    dilate: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Point-guided extract for pale watermarks (no GrabCut / no EdgeSAM).

    labels: 1 = foreground stroke, 0 = background.
    Pipeline: multi-scale residual map → looser threshold → geodesic
    reconstruction from FG seeds under residual → punch BG seeds.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop is empty")
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    lbs = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(pts) == 0 or not np.any(lbs == 1):
        raise ValueError("请至少在水印笔画上点一个前景点（绿点）")

    fg_pts = [(float(x), float(y)) for (x, y), lab in zip(pts, lbs) if int(lab) == 1]
    bg_pts = [(float(x), float(y)) for (x, y), lab in zip(pts, lbs) if int(lab) == 0]

    mask, acc = _letter_residual_mask(crop_bgr, percentile=78.0)
    # Reconstruct under a looser residual using FG seeds (grow along strokes)
    thr_loose = max(5, int(np.percentile(acc, 70)))
    _, loose = cv2.threshold(acc.astype(np.uint8), thr_loose, 255, cv2.THRESH_BINARY)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    h, w = mask.shape[:2]
    marker = np.zeros((h, w), dtype=np.uint8)
    for x, y in fg_pts:
        cv2.circle(marker, (int(round(x)), int(round(y))), 2, 255, -1)
    marker = cv2.bitwise_and(marker, loose)
    if not marker.any():
        # Seeds missed residual — still force small disks at FG and OR with letter mask
        for x, y in fg_pts:
            cv2.circle(marker, (int(round(x)), int(round(y))), 3, 255, -1)
        grown = cv2.bitwise_or(mask, marker)
    else:
        grown = _geodesic_dilate(marker, loose, max_iter=100)

    for x, y in bg_pts:
        cv2.circle(grown, (int(round(x)), int(round(y))), 5, 0, -1)

    grown = cv2.morphologyEx(
        grown, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    )
    grown = _keep_primary_watermark_components(grown)
    # Merge auto letter mask near FG band so we don't only keep seed islands
    grown = _keep_primary_watermark_components(cv2.bitwise_or(grown, mask))

    fill = float(np.count_nonzero(grown) / max(1, grown.size))
    method = "points_residual"
    if fill < 0.04:
        # Even looser thr
        thr2 = max(4, int(np.percentile(acc, 62)))
        _, loose2 = cv2.threshold(acc.astype(np.uint8), thr2, 255, cv2.THRESH_BINARY)
        marker2 = np.zeros((h, w), dtype=np.uint8)
        for x, y in fg_pts:
            cv2.circle(marker2, (int(round(x)), int(round(y))), 3, 255, -1)
        grown = _geodesic_dilate(cv2.bitwise_and(marker2, loose2), loose2, max_iter=120)
        for x, y in bg_pts:
            cv2.circle(grown, (int(round(x)), int(round(y))), 5, 0, -1)
        grown = _keep_primary_watermark_components(grown)
        method = "points_residual_loose"
        fill = float(np.count_nonzero(grown) / max(1, grown.size))

    grown, fill = _finalize_template_mask(grown, dilate=dilate)
    if fill < 0.02:
        raise RuntimeError(
            "点选结果过空。请把绿点点在半透明笔画上（不要点花瓣），并少点红点。"
        )
    if fill > 0.75:
        # Too fat — use stricter letter mask only near FG
        strict, _ = _letter_residual_mask(crop_bgr, percentile=85.0)
        marker = np.zeros_like(strict)
        for x, y in fg_pts:
            cv2.circle(marker, (int(round(x)), int(round(y))), 2, 255, -1)
        grown = _geodesic_dilate(cv2.bitwise_and(marker, strict), strict, max_iter=80)
        grown = _keep_primary_watermark_components(grown)
        grown, fill = _finalize_template_mask(grown, dilate=0)
        method = "points_residual_strict"

    stats = {
        "fill_ratio": round(fill, 4),
        "method": method,
        "template_size": [int(grown.shape[1]), int(grown.shape[0])],
        "n_fg": int(np.sum(lbs == 1)),
        "n_bg": int(np.sum(lbs == 0)),
        "score": round(1.0 - abs(fill - 0.22), 4),
    }
    return grown, stats


def _smart_extract(
    crop_bgr: np.ndarray, *, min_fill_ratio: float = 0.04
) -> tuple[np.ndarray, str, float]:
    """Multi-scale tophat residual for pale text; never GrabCut on similar colors."""
    mask, _acc = _letter_residual_mask(crop_bgr, percentile=78.0)
    fill = float(np.count_nonzero(mask) / max(1, mask.size))
    method = "multiscale_tophat"

    if fill < min_fill_ratio:
        # Looser
        mask, _ = _letter_residual_mask(crop_bgr, percentile=72.0)
        fill = float(np.count_nonzero(mask) / max(1, mask.size))
        method = "multiscale_tophat_loose"
    if fill > 0.55:
        mask, _ = _letter_residual_mask(crop_bgr, percentile=84.0)
        fill = float(np.count_nonzero(mask) / max(1, mask.size))
        method = "multiscale_tophat_strict"
    if fill < min_fill_ratio:
        refined, refined_method = _refine_overfilled_mask(crop_bgr)
        refined = _keep_primary_watermark_components(refined)
        rf = float(np.count_nonzero(refined) / max(1, refined.size))
        if rf >= min_fill_ratio:
            return refined, refined_method, rf
        feat = _extract_watermark_mask(crop_bgr)
        ff = float(np.count_nonzero(feat) / max(1, feat.size))
        if min_fill_ratio <= ff <= 0.65:
            return feat, "feature", ff
        env = _envelope_mask(crop_bgr.shape[:2])
        return env, "envelope_fallback", float(np.count_nonzero(env) / max(1, env.size))
    return mask, method, fill


def _letter_residual_mask(
    crop_bgr: np.ndarray, *, percentile: float = 78.0
) -> tuple[np.ndarray, np.ndarray]:
    """Binary letter-like mask + float residual map (tophat/blackhat multi-scale).

    Works for bright or dark semi-transparent logos without color similarity
    assumptions required by GrabCut.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    acc = np.zeros(gray.shape, dtype=np.float32)
    for k in (9, 15, 21, 31, 45):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        acc = np.maximum(
            acc, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker).astype(np.float32)
        )
        acc = np.maximum(
            acc, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker).astype(np.float32)
        )
    # Local median residual also helps on textured flowers
    for k in (11, 21):
        med = cv2.medianBlur(gray, k)
        acc = np.maximum(acc, cv2.absdiff(gray, med).astype(np.float32))

    thr = max(6, int(np.percentile(acc, float(percentile))))
    _, mask = cv2.threshold(acc.astype(np.uint8), thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Connect letters along reading direction
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    )
    mask = _keep_primary_watermark_components(mask)
    return mask, acc


def _residual_seed_map(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias used by tests / callers."""
    return _letter_residual_mask(crop_bgr, percentile=80.0)


def _geodesic_dilate(
    marker: np.ndarray, mask: np.ndarray, *, max_iter: int = 80
) -> np.ndarray:
    """Morphological reconstruction: grow marker under mask."""
    ker = np.ones((3, 3), np.uint8)
    cur = (marker > 0).astype(np.uint8) * 255
    limit = (mask > 0).astype(np.uint8) * 255
    for _ in range(max_iter):
        nxt = cv2.bitwise_and(cv2.dilate(cur, ker), limit)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def _finalize_template_mask(
    mask: np.ndarray, *, dilate: int = 1
) -> tuple[np.ndarray, float]:
    """Optional modest dilate + component filter; return mask and fill ratio."""
    mask = (mask > 10).astype(np.uint8) * 255
    if dilate > 0:
        d = min(int(dilate), 2)
        kernel_size = max(1, d * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = _keep_primary_watermark_components(mask)
    fill_ratio = float(np.count_nonzero(mask) / max(1, mask.size))
    return mask, fill_ratio


def crop_oriented_roi(
    image_bgr: np.ndarray,
    bbox: BBox,
    angle_deg: float = 0.0,
) -> tuple[np.ndarray, BBox]:
    """Crop the content under a (possibly rotated) ROI as an upright patch.

    ``bbox`` is the local axis-aligned rect in image pixels (same as canvas
    ``_roi_rect``). ``angle_deg`` is the canvas rotation around the rect center
    (matches ImageCanvas / Qt). When angle is ~0, this is a normal slice.
    When rotated, warp so the OBB content becomes a w×h upright crop — avoids
    axis-aligned crop cutting off diagonal text (e.g. half of \"Monica\").
    """
    height, width = image_bgr.shape[:2]
    box = _clamp_bbox(bbox.x, bbox.y, bbox.width, bbox.height, width, height)
    ang = float(angle_deg or 0.0)
    if abs(ang) < 0.5:
        crop = image_bgr[box.y : box.bottom, box.x : box.right].copy()
        return crop, box

    cx = box.x + box.width / 2.0
    cy = box.y + box.height / 2.0
    out_w = max(8, int(round(box.width)))
    out_h = max(8, int(round(box.height)))
    # Derotate image so the oriented box becomes axis-aligned, then crop w×h
    # at the mapped center. Sign matches Qt/canvas: positive angle = CCW of rect.
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m[0, 2] += (out_w / 2.0) - cx
    m[1, 2] += (out_h / 2.0) - cy
    crop = cv2.warpAffine(
        image_bgr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, box


def build_solid_roi_template(
    image_bgr: np.ndarray,
    bbox: BBox,
    *,
    dilate: int = 0,
    angle_deg: float = 0.0,
) -> TemplateBuildResult:
    """User ROI as a **solid filled** mask (no feature extract / AI matting).

    Used when creating styles: position + region are what the user drew.
    Batch「固定位置」only needs roi_norm; nearby/full-frame can still re-extract
    a finer recognition template later in the detail page.
    """
    height, width = image_bgr.shape[:2]
    crop, box = crop_oriented_roi(image_bgr, bbox, angle_deg=angle_deg)
    if crop.size == 0:
        raise ValueError("ROI is empty")

    mask = np.full((crop.shape[0], crop.shape[1]), 255, dtype=np.uint8)
    if dilate > 0:
        k = max(1, int(dilate) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    preview = _preview_overlay(crop, mask)
    detector = suggest_detector_params(box, width, height, crop.shape[1])
    # Pin-friendly: keep bbox solid as default output for matching too
    detector["output_mask_mode"] = "bbox"

    return TemplateBuildResult(
        template_mask=mask,
        sample_crop_bgr=crop,
        preview_overlay_bgr=preview,
        bbox=box,
        image_size=(width, height),
        stats={
            "fill_ratio": 1.0,
            "method": "solid_roi",
            "template_size": [int(crop.shape[1]), int(crop.shape[0])],
            "roi": box.to_list(),
            "angle_deg": round(float(angle_deg or 0.0), 2),
        },
        suggested_detector=detector,
    )


def build_template_from_roi(
    image_bgr: np.ndarray,
    bbox: BBox,
    *,
    dilate: int = 3,
    min_fill_ratio: float = 0.04,
    angle_deg: float = 0.0,
    solid: bool = False,
) -> TemplateBuildResult:
    """Build template from ROI.

    - solid=True (default for new styles): filled rectangle = user box.
    - solid=False: feature extract (detail-page reextract / legacy).
    """
    if solid:
        return build_solid_roi_template(
            image_bgr, bbox, dilate=max(0, int(dilate) - 1), angle_deg=angle_deg
        )

    height, width = image_bgr.shape[:2]
    crop, box = crop_oriented_roi(image_bgr, bbox, angle_deg=angle_deg)
    if crop.size == 0:
        raise ValueError("ROI is empty")

    mask, stats0 = extract_mask_from_crop(crop, dilate=dilate, min_fill_ratio=min_fill_ratio)
    fill_ratio = float(stats0.get("fill_ratio") or 0.0)
    method = str(stats0.get("method") or "feature")

    # Keep crop and mask the same size as the user selection (no mask-based trim)
    if mask.shape[:2] != crop.shape[:2]:
        mask = cv2.resize(
            mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        fill_ratio = float(np.count_nonzero(mask) / max(1, mask.size))

    preview = _preview_overlay(crop, mask)
    detector = suggest_detector_params(box, width, height, crop.shape[1])

    return TemplateBuildResult(
        template_mask=mask,
        sample_crop_bgr=crop,
        preview_overlay_bgr=preview,
        bbox=box,
        image_size=(width, height),
        stats={
            "fill_ratio": round(fill_ratio, 4),
            "method": method,
            "template_size": [int(crop.shape[1]), int(crop.shape[0])],
            "roi": box.to_list(),
            "angle_deg": round(float(angle_deg or 0.0), 2),
        },
        suggested_detector=detector,
    )


def suggest_detector_params(
    bbox: BBox,
    image_width: int,
    image_height: int,
    template_width: int,
) -> dict[str, Any]:
    """Defaults for a new style: full-frame search + solid bbox mask for LaMa.

    ``bbox`` / image size only seed ``reference_width``; we intentionally do not
    shrink search regions (position bias caused many misses on similar stamps).
    """
    _ = bbox  # ROI size is captured via template_width / sample crop, not search box
    return {
        "reference_width": int(image_width),
        # Wider scale pyramid → same stamp at different resolutions
        "scale_factors": [0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7],
        "min_confidence": 0.28,
        "feature_mode": "fused",
        "feature_adaptive": True,
        # Solid rectangle over the hit — LaMa covers the rest (not stamp contour)
        "output_mask_mode": "bbox",
        "footprint_close_kernel": 21,
        "feature_kernel": 31,
        "feature_threshold": 12,
        "candidate_limit": 32,
        "min_hough_circles": 0,
        "min_watermark_color_ratio": 0.0,
        "min_bottom_ratio": 0.0,
        "prefer_larger_score_margin": 0.0,
        "hough_param2": 18,
        "dilate": 4,
        "mask_expand_ratio": 0.12,
        "mask_expand_top_ratio": 0.12,
        "mask_expand_bottom_ratio": 0.12,
        "mask_expand_x_ratio": 0.12,
        "edge_expand_ratio": 0.08,
        "refine_enabled": True,
        "min_match_side": 48,
        "min_side_ratio": 0.04,
        "max_side_ratio": 0.48,
        "min_scale_vs_expected": 0.55,
        "max_scale_vs_expected": 1.55,
        "min_residual_density": 0.06,
        "multi_instance": False,
        "multi_score_ratio": 0.78,
        "search_regions": [
            {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}
        ],
        "template_pixel_width": int(template_width),
    }


def _extract_watermark_mask(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Semi-transparent pale overlays (common watermarks)
    pale = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
    # Dark ink / stamp — keep moderate darkness only (avoid whole black garment)
    dark = cv2.inRange(hsv, (0, 0, 15), (180, 140, 95))
    # Bright text
    bright = cv2.inRange(gray, 185, 255)

    # High-frequency residual vs local background (good for translucent logos)
    # Slightly lower thresholds so thin / pale letter strokes are not dropped.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, tophat_mask = cv2.threshold(tophat, 14, 255, cv2.THRESH_BINARY)
    _, blackhat_mask = cv2.threshold(blackhat, 14, 255, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_16S, ksize=3))
    _, lap_mask = cv2.threshold(lap, 14, 255, cv2.THRESH_BINARY)

    # Adaptive residual: pale text often sits just above local median
    med = cv2.medianBlur(gray, 15)
    diff = cv2.absdiff(gray, med)
    thr = max(10, int(np.percentile(diff, 78)))
    _, adaptive_res = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    residual = tophat_mask | blackhat_mask | lap_mask | adaptive_res
    combined = pale | bright | residual
    combined = combined | (dark & residual)

    # Suppress highly saturated photo content (clothes, sky) when not stamp-like
    high_sat = saturation > 100
    high_detail = value > 40
    photo_like = high_sat & high_detail
    combined[photo_like & (pale == 0)] = 0

    near_black = value < 28
    combined[near_black & (pale == 0) & (bright == 0)] = 0

    # Mild open (noise), then horizontal-biased close to reconnect text strokes
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
    # Extra close along text reading direction helps "Monica" stay one band
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, h_close)

    cleaned = _keep_primary_watermark_components(combined)
    if np.count_nonzero(cleaned) == 0:
        return combined
    return cleaned


def _keep_primary_watermark_components(mask: np.ndarray) -> np.ndarray:
    """Drop small satellite blobs (leaf tips, fabric noise) around the main stamp.

    Keeps the largest component always; keeps secondary ones only if they are
    large enough relative to the main blob and not far from its bounding box.
    Text letters that split into several blobs are re-joined when near the main mass.
    """
    binary = (mask > 10).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    comps: list[tuple[int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comps.append((area, label))
    comps.sort(reverse=True)
    main_area, main_id = comps[0]
    if main_area < 8:
        return binary

    mx = int(stats[main_id, cv2.CC_STAT_LEFT])
    my = int(stats[main_id, cv2.CC_STAT_TOP])
    mw = int(stats[main_id, cv2.CC_STAT_WIDTH])
    mh = int(stats[main_id, cv2.CC_STAT_HEIGHT])
    # Wide pad: multi-letter logos often break into separate components
    pad_x = max(8, int(round(mw * 0.85)))
    pad_y = max(6, int(round(mh * 0.75)))
    x1 = mx - pad_x
    y1 = my - pad_y
    x2 = mx + mw + pad_x
    y2 = my + mh + pad_y

    keep = {main_id}
    # Lower floor so thin letter fragments (i, c dots) survive when near the word
    area_floor = max(12, int(main_area * 0.04), int(binary.size * 0.0015))
    for area, label in comps[1:]:
        if area < area_floor:
            continue
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        bw = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        bh = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        aspect = max(bw, bh) / float(min(bw, bh))
        fill = area / float(bw * bh)
        # Only drop compact solid noise dots far smaller than main text
        if area < main_area * 0.08 and aspect < 1.2 and fill > 0.7:
            continue
        keep.add(label)

    cleaned = np.zeros_like(binary)
    for label in keep:
        cleaned[labels == label] = 255

    if np.count_nonzero(cleaned) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, hk)
    return cleaned


def _refine_overfilled_mask(crop_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Second-pass residual extraction when the first pass paints most of the ROI."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, 21)
    diff = cv2.absdiff(gray, med)
    thr, residual = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 18:
        _, residual = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    residual = cv2.morphologyEx(
        residual, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    # Drop very dark garment pixels that still leak through
    value = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    residual[value < 25] = 0
    return residual, "residual_refine"


def _envelope_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    pad_x = max(1, int(round(width * 0.04)))
    pad_y = max(1, int(round(height * 0.04)))
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(1, width // 2 - pad_x), max(1, height // 2 - pad_y)),
        0,
        0,
        360,
        255,
        -1,
    )
    # also fill a soft rectangle so text bars are covered
    cv2.rectangle(
        mask,
        (pad_x, int(height * 0.25)),
        (width - 1 - pad_x, int(height * 0.75)),
        255,
        -1,
    )
    return mask


def _trim_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = 2
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1] - 1, x2 + pad)
    y2 = min(mask.shape[0] - 1, y2 + pad)
    trimmed = mask[y1 : y2 + 1, x1 : x2 + 1].copy()
    return trimmed, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _preview_overlay(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = crop_bgr.copy()
    red = np.zeros_like(crop_bgr)
    red[:, :, 2] = 255
    hit = mask > 0
    if np.any(hit):
        overlay[hit] = cv2.addWeighted(crop_bgr, 0.4, red, 0.6, 0)[hit]
    return overlay
