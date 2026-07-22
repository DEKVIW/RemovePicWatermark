from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..masking import dilate_mask
from ..models import BBox, Detection


@dataclass
class FixedBoxDetector:
    """Place a mask at a fixed normalized box (optional template shape)."""

    label: str
    box: dict[str, float]
    mask_mode: str = "rectangle"
    bright_threshold: int = 190
    low_saturation_threshold: int = 130
    min_mask_ratio: float = 0.0
    min_span_ratio: float = 0.0
    dilate: int = 0
    fallback_to_rectangle: bool = True
    # Optional HxW uint8 mask from style template; resized into the box
    template_mask: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FixedBoxDetector":
        template_mask = config.get("template_mask")
        if template_mask is not None:
            template_mask = np.asarray(template_mask)
            if template_mask.ndim == 3:
                template_mask = template_mask[:, :, 0]
        return cls(
            label=config["label"],
            box=config["box"],
            mask_mode=config.get("mask_mode", "rectangle"),
            bright_threshold=int(config.get("bright_threshold", 190)),
            low_saturation_threshold=int(config.get("low_saturation_threshold", 130)),
            min_mask_ratio=float(config.get("min_mask_ratio", 0.0)),
            min_span_ratio=float(config.get("min_span_ratio", 0.0)),
            dilate=int(config.get("dilate", 0)),
            fallback_to_rectangle=bool(config.get("fallback_to_rectangle", True)),
            template_mask=template_mask,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        bbox = self._absolute_box(width, height).clamp(width, height)
        if bbox.width == 0 or bbox.height == 0:
            return []

        mask = np.zeros((height, width), dtype=np.uint8)
        roi = image[bbox.y : bbox.bottom, bbox.x : bbox.right]

        if self.template_mask is not None and self.template_mask.size > 0:
            tm = self.template_mask
            if tm.ndim == 3:
                tm = tm[:, :, 0]
            placed = cv2.resize(
                tm.astype(np.uint8),
                (bbox.width, bbox.height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = (placed > 127).astype(
                np.uint8
            ) * 255
            mode_meta = "template"
        elif self.mask_mode == "bright_text":
            roi_mask = self._bright_text_mask(roi)
            if not self._is_plausible_text_mask(roi_mask):
                if not self.fallback_to_rectangle:
                    return []
                roi_mask[:, :] = 255
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = roi_mask
            mode_meta = "bright_text"
        else:
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
            mode_meta = "rectangle"

        mask = dilate_mask(mask, self.dilate)
        return [
            Detection(
                label=self.label,
                bbox=bbox,
                confidence=1.0,
                mask=mask,
                metadata={"detector": "fixed_box", "mask_mode": mode_meta, "pin": True},
            )
        ]

    def _absolute_box(self, width: int, height: int) -> BBox:
        left = int(round(width * self.box["left"]))
        top = int(round(height * self.box["top"]))
        right = int(round(width * self.box["right"]))
        bottom = int(round(height * self.box["bottom"]))
        return BBox(left, top, right - left, bottom - top)

    def _bright_text_mask(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, self.bright_threshold, 255)
        low_sat = cv2.inRange(hsv[:, :, 1], 0, self.low_saturation_threshold)
        return cv2.bitwise_and(bright, low_sat)

    def _is_plausible_text_mask(self, mask: np.ndarray) -> bool:
        mask_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        if total_pixels == 0 or mask_pixels / total_pixels < self.min_mask_ratio:
            return False

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return False
        span_ratio = (int(xs.max()) - int(xs.min()) + 1) / mask.shape[1]
        return span_ratio >= self.min_span_ratio
