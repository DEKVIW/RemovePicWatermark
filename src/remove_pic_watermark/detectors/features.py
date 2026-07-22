"""Feature maps for watermark template matching.

Stage-1 enhancement for semi-transparent / pale overlays:
- tophat: bright residual (light text on mid tones)
- blackhat: dark residual (dark stamps)
- laplacian: local contrast edges
- fused: OR of the above (default for full-frame search)
- adaptive: fused + percentile-based thresholds per image

All public functions are pure (numpy/cv2 only) for easy unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# Supported feature_mode values (also accepted aliases)
FEATURE_MODES = frozenset(
    {
        "tophat",
        "blackhat",
        "laplacian",
        "canny",
        "fused",
        "adaptive",
    }
)


@dataclass(frozen=True)
class FeatureParams:
    mode: str = "fused"
    kernel: int = 31
    threshold: int = 12  # absolute floor; adaptive may lower further
    adaptive: bool = False

    @classmethod
    def from_detector_fields(
        cls,
        *,
        feature_mode: str = "fused",
        feature_kernel: int = 31,
        feature_threshold: int = 12,
        feature_adaptive: bool | None = None,
    ) -> "FeatureParams":
        mode = (feature_mode or "fused").strip().lower()
        if mode not in FEATURE_MODES:
            mode = "fused"
        # "adaptive" mode always enables adaptive thresholds
        adaptive = bool(feature_adaptive) if feature_adaptive is not None else (mode == "adaptive")
        if mode == "adaptive":
            mode = "fused"
        return cls(
            mode=mode,
            kernel=int(feature_kernel),
            threshold=int(feature_threshold),
            adaptive=adaptive,
        )


def odd_kernel(size: int) -> int:
    size = max(3, int(size))
    return size if size % 2 == 1 else size + 1


def adaptive_threshold(response: np.ndarray, floor: int, *, percentile: float = 92.0) -> int:
    """Pick a binary threshold from response statistics, never below floor*0.4."""
    if response.size == 0:
        return max(1, floor)
    # ignore zeros for sparse residual maps
    nz = response[response > 0]
    if nz.size < 32:
        return max(1, floor)
    p = float(np.percentile(nz, percentile))
    # Soft floor: allow lower than configured for very pale watermarks
    soft_floor = max(3, int(round(floor * 0.45)))
    thr = int(round(p * 0.55))
    return int(np.clip(thr, soft_floor, max(soft_floor, floor * 2)))


def _morph_kernel(size: int) -> np.ndarray:
    k = odd_kernel(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def tophat_response(gray: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Bright residual — good for white/pale semi-transparent text."""
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _morph_kernel(kernel))


def blackhat_response(gray: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Dark residual — good for dark ink stamps."""
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _morph_kernel(kernel))


def laplacian_response(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(lap)


def _binarize(response: np.ndarray, threshold: int, *, adaptive: bool) -> np.ndarray:
    thr = adaptive_threshold(response, threshold) if adaptive else max(1, threshold)
    _, binary = cv2.threshold(response, thr, 255, cv2.THRESH_BINARY)
    return binary


def compute_feature_edges(
    image_bgr: np.ndarray,
    params: FeatureParams | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Build a binary edge/feature map used as matchTemplate target.

    Returns uint8 image, 0/255.
    """
    if params is None:
        params = FeatureParams.from_detector_fields(**kwargs) if kwargs else FeatureParams()

    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mode = params.mode
    adaptive = params.adaptive
    thr = params.threshold
    kernel = params.kernel

    if mode == "laplacian":
        resp = laplacian_response(gray)
        binary = _binarize(resp, thr, adaptive=adaptive)
        return binary

    if mode == "tophat":
        resp = tophat_response(gray, kernel)
        binary = _binarize(resp, thr, adaptive=adaptive)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # keep some structure for matching
        edges = cv2.Canny(binary, 20, 80)
        return cv2.bitwise_or(binary, edges)

    if mode == "blackhat":
        resp = blackhat_response(gray, kernel)
        binary = _binarize(resp, thr, adaptive=adaptive)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return binary

    if mode == "canny":
        return cv2.Canny(gray, max(10, thr), max(30, thr * 3))

    # fused (default): tophat | blackhat | laplacian — covers pale text and dark marks
    th = tophat_response(gray, kernel)
    bh = blackhat_response(gray, kernel)
    lap = laplacian_response(gray)

    th_bin = _binarize(th, thr, adaptive=adaptive)
    bh_bin = _binarize(bh, thr, adaptive=adaptive)
    # laplacian usually needs a slightly higher floor to avoid noise
    lap_thr = thr if not adaptive else adaptive_threshold(lap, max(thr, 10), percentile=94.0)
    _, lap_bin = cv2.threshold(lap, lap_thr, 255, cv2.THRESH_BINARY)

    fused = cv2.bitwise_or(th_bin, bh_bin)
    fused = cv2.bitwise_or(fused, lap_bin)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # thin edges help template correlation; keep residual mass for pale text
    edges = cv2.Canny(fused, 20, 80)
    return cv2.bitwise_or(fused, edges)


def build_template_feature_edges(
    sample_bgr: np.ndarray | None,
    mask_gray: np.ndarray,
    params: FeatureParams | None = None,
) -> np.ndarray:
    """Edges for the template side of matchTemplate.

    Always prefer **sample_crop residual features** (same domain as the search
    image). Using only the binary mask outline (old path when fill_ratio < 0.45)
    made scores collapse to ~0.15 after users recreated sparse stamp masks —
    GUI jobs then reported 0 hits while tests on older templates still worked.
    """
    params = params or FeatureParams()
    mask_bin = (mask_gray > 10).astype(np.uint8) * 255

    if sample_bgr is not None and sample_bgr.size > 0:
        content = compute_feature_edges(sample_bgr, params)
        if content.shape[:2] == mask_gray.shape[:2] and int(np.count_nonzero(mask_bin)) > 0:
            # Keep stamp region; if mask is sparse, dilate slightly so strokes survive
            if float(np.count_nonzero(mask_bin) / max(1, mask_bin.size)) < 0.35:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_use = cv2.dilate(mask_bin, k, iterations=1)
            else:
                mask_use = mask_bin
            content = cv2.bitwise_and(content, content, mask=mask_use)
        if int(np.count_nonzero(content)) >= 40:
            return content
        # Unmasked sample features still beat pure silhouette for NCC
        if int(np.count_nonzero(content)) >= 20:
            return content
        raw = compute_feature_edges(sample_bgr, params)
        if int(np.count_nonzero(raw)) >= 40:
            return raw

    # Fallback only when sample_crop is missing: outline of binary mask
    outline = cv2.Canny(mask_gray, 20, 80)
    if int(np.count_nonzero(outline)) >= 20:
        return outline
    return mask_gray
