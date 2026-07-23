"""Auto-scan detector: residual feature → connected components.

No neural weights, no style profile. Product label: 「自动扫描」.
Complements template matching for pale tiled text when no style is available,
or unions with styles in detect_mode=both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..masking import dilate_mask
from ..models import BBox, Detection
from .features import FeatureParams, compute_feature_edges


@dataclass
class ResidualAiDetector:
    """Unsupervised watermark-like region proposal from fused residuals.

    Product label: 「自动扫描」 (always available, no weights).
    YOLO is a separate optional detector in the same scan chain.
    """

    label: str = "ai_residual"
    feature_mode: str = "fused"
    feature_threshold: int = 10
    feature_adaptive: bool = True
    feature_kernel: int = 31
    min_area: int = 80
    max_area_ratio: float = 0.12
    min_aspect: float = 0.15
    max_aspect: float = 8.0
    max_instances: int = 96
    dilate: int = 3
    close_kernel: int = 5
    min_fill: float = 0.08
    # Drop components that hug the full image border (often photo edges / frames)
    reject_border_full: bool = True
    border_margin_ratio: float = 0.02

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "ResidualAiDetector":
        cfg = dict(config or {})
        return cls(
            label=str(cfg.get("label", "ai_residual")),
            feature_mode=str(cfg.get("feature_mode", "fused")),
            feature_threshold=int(cfg.get("feature_threshold", 10)),
            feature_adaptive=bool(cfg.get("feature_adaptive", True)),
            feature_kernel=int(cfg.get("feature_kernel", 31)),
            min_area=int(cfg.get("min_area", 80)),
            max_area_ratio=float(cfg.get("max_area_ratio", 0.12)),
            min_aspect=float(cfg.get("min_aspect", 0.15)),
            max_aspect=float(cfg.get("max_aspect", 8.0)),
            max_instances=int(cfg.get("max_instances", 96)),
            dilate=int(cfg.get("dilate", 3)),
            close_kernel=int(cfg.get("close_kernel", 5)),
            min_fill=float(cfg.get("min_fill", 0.08)),
            reject_border_full=bool(cfg.get("reject_border_full", True)),
            border_margin_ratio=float(cfg.get("border_margin_ratio", 0.02)),
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        params = FeatureParams.from_detector_fields(
            feature_mode=self.feature_mode,
            feature_kernel=self.feature_kernel,
            feature_threshold=self.feature_threshold,
            feature_adaptive=self.feature_adaptive,
        )
        edges = compute_feature_edges(image, params)
        # Connect broken glyph strokes
        k = self.close_kernel if self.close_kernel % 2 == 1 else self.close_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k), max(3, k)))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        num, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        max_area = int(height * width * self.max_area_ratio)
        margin_x = max(2, int(width * self.border_margin_ratio))
        margin_y = max(2, int(height * self.border_margin_ratio))
        candidates: list[tuple[float, int, int, int, int, float]] = []
        for i in range(1, num):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area or area > max_area:
                continue
            if w < 6 or h < 6:
                continue
            aspect = w / max(1, h)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            if self.reject_border_full and self._is_border_frame(x, y, w, h, width, height, margin_x, margin_y):
                continue
            # score: residual density in box
            patch = edges[y : y + h, x : x + w]
            fill = float(np.count_nonzero(patch) / max(1, patch.size))
            if fill < self.min_fill:
                continue
            score = min(0.99, 0.35 + fill * 0.6)
            candidates.append((score, x, y, w, h, fill))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[: self.max_instances]

        detections: list[Detection] = []
        for index, (score, x, y, w, h, fill) in enumerate(candidates):
            bbox = BBox(x, y, w, h).clamp(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            # Solid bbox for LaMa — residual only *locates*; contour not used as mask
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
            mask = dilate_mask(mask, self.dilate)
            detections.append(
                Detection(
                    label=self.label,
                    bbox=bbox,
                    confidence=float(score),
                    mask=mask,
                    metadata={
                        "detector": "residual_ai",
                        "fill": round(fill, 4),
                        "mask_mode": "bbox_fill",
                        "instance_index": index,
                    },
                )
            )
        return detections

    @staticmethod
    def _is_border_frame(
        x: int,
        y: int,
        w: int,
        h: int,
        width: int,
        height: int,
        margin_x: int,
        margin_y: int,
    ) -> bool:
        """True if box spans nearly full width or height while hugging edges (photo frame)."""
        spans_w = w >= int(width * 0.85)
        spans_h = h >= int(height * 0.85)
        touches_lr = x <= margin_x and (x + w) >= width - margin_x
        touches_tb = y <= margin_y and (y + h) >= height - margin_y
        if spans_w and touches_tb and h < int(height * 0.2):
            return True  # top/bottom strip frame
        if spans_h and touches_lr and w < int(width * 0.2):
            return True  # left/right strip frame
        if spans_w and spans_h:
            return True  # almost whole image
        return False
