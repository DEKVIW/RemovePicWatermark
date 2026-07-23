"""Template-matching utilities: multi-peak extraction and NMS.

Used by TemplateStampDetector for full-frame repeating watermarks
(e.g. tiled text logos). Pure functions — no I/O, easy to unit-test.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def extract_peaks(
    score_map: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    suppress_width: int,
    suppress_height: int,
) -> list[tuple[float, int, int]]:
    """Greedy peaks on a matchTemplate score map.

    Returns list of (score, x, y) in score-map coordinates (top-left of match).
    After each peak, a neighborhood of size suppress_w × suppress_h is zeroed
    so nearby duplicates of the same instance are suppressed.
    """
    if score_map.size == 0 or max_peaks <= 0:
        return []

    work = score_map.astype(np.float32, copy=True)
    h, w = work.shape[:2]
    half_w = max(1, suppress_width // 2)
    half_h = max(1, suppress_height // 2)
    peaks: list[tuple[float, int, int]] = []

    for _ in range(max_peaks):
        _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(work)
        score = float(max_v)
        if score < threshold:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        peaks.append((score, x, y))
        x0 = max(0, x - half_w)
        x1 = min(w, x + half_w + 1)
        y0 = max(0, y - half_h)
        y1 = min(h, y + half_h + 1)
        work[y0:y1, x0:x1] = -1.0

    return peaks


def box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU for axis-aligned boxes as (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def nms_by_iou(
    candidates: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.35,
    max_keep: int = 64,
) -> list[dict[str, Any]]:
    """Greedy NMS on candidate dicts with keys score, location (x,y), size (w,h)."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in ordered:
        if len(kept) >= max_keep:
            break
        x, y = cand["location"]
        w, h = cand["size"]
        box = (int(x), int(y), int(w), int(h))
        if any(
            box_iou(box, (k["location"][0], k["location"][1], k["size"][0], k["size"][1]))
            >= iou_threshold
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


def match_template_peaks(
    image_edges: np.ndarray,
    template_edges: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    multi_instance: bool,
) -> list[tuple[float, tuple[int, int], tuple[int, int]]]:
    """Run TM_CCOEFF_NORMED and return peaks as (score, location, size).

    When multi_instance is False, returns at most one best peak (legacy behaviour).
    """
    th, tw = template_edges.shape[:2]
    if tw < 8 or th < 8:
        return []
    if tw > image_edges.shape[1] or th > image_edges.shape[0]:
        return []
    if np.count_nonzero(template_edges) == 0:
        return []

    score_map = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    if not multi_instance:
        _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(score_map)
        if float(max_v) < threshold:
            return []
        return [(float(max_v), (int(max_loc[0]), int(max_loc[1])), (tw, th))]

    # Suppress radius ≈ half template so adjacent tiles can still be found
    peaks = extract_peaks(
        score_map,
        threshold=threshold,
        max_peaks=max_peaks,
        suppress_width=max(4, tw // 2),
        suppress_height=max(4, th // 2),
    )
    return [(score, (x, y), (tw, th)) for score, x, y in peaks]


def refine_match_local(
    image_edges: np.ndarray,
    template_edges: np.ndarray,
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    scale: float,
    pad_ratio: float = 0.35,
    scale_multipliers: tuple[float, ...] = (0.90, 0.95, 1.0, 1.05, 1.10),
    min_score: float = 0.20,
    min_side: int = 24,
) -> tuple[float, tuple[int, int], tuple[int, int], float] | None:
    """Re-match around a coarse hit with nearby scales for tighter localization.

    Returns (score, absolute_location, size, scale) or None if refinement fails.
    """
    img_h, img_w = image_edges.shape[:2]
    x, y = int(location[0]), int(location[1])
    w, h = int(size[0]), int(size[1])
    if w < 8 or h < 8:
        return None

    pad = int(round(max(w, h) * float(pad_ratio)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)
    roi = image_edges[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    tpl_h0, tpl_w0 = template_edges.shape[:2]
    best: tuple[float, tuple[int, int], tuple[int, int], float] | None = None

    for mult in scale_multipliers:
        s = float(scale) * float(mult)
        tw = max(1, int(round(tpl_w0 * s)))
        th = max(1, int(round(tpl_h0 * s)))
        if tw < min_side or th < min_side:
            continue
        if tw > roi.shape[1] or th > roi.shape[0]:
            continue
        resized = cv2.resize(template_edges, (tw, th), interpolation=cv2.INTER_AREA)
        if np.count_nonzero(resized) == 0:
            continue
        score_map = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(score_map)
        score = float(max_v)
        if score < min_score:
            continue
        abs_loc = (x0 + int(max_loc[0]), y0 + int(max_loc[1]))
        if best is None or score > best[0]:
            best = (score, abs_loc, (tw, th), s)

    return best
