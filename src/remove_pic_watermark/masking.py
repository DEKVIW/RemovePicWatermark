from __future__ import annotations

import cv2
import numpy as np

from .models import Detection


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel_size = max(1, pixels * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def combine_masks(detections: list[Detection], shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=np.uint8)
    for detection in detections:
        combined = cv2.bitwise_or(combined, detection.mask)
    return combined


def draw_debug_overlay(image: np.ndarray, mask: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Red = actual inpaint mask pixels; yellow outline = box geometry.

    For OBB detections with a polygon stored in metadata, draw the rotated
    quad (not only the axis-aligned outer bbox) so tilted masks are visible.
    """
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image, 0.45, red, 0.55, 0)[mask_bool]

    for detection in detections:
        bbox = detection.bbox
        meta = detection.metadata or {}
        poly = meta.get("obb_poly")
        drew_poly = False
        if poly is not None:
            try:
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) >= 3:
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                    drew_poly = True
            except Exception:  # noqa: BLE001
                drew_poly = False
        if not drew_poly:
            cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.right, bbox.bottom), (0, 255, 255), 2)
        ang = float(meta.get("angle_deg") or 0.0)
        if abs(ang) >= 3.0:
            label = f"{detection.label} {detection.confidence:.2f} r{ang:.0f}"
        else:
            label = f"{detection.label} {detection.confidence:.2f}"
        y = max(22, bbox.y - 8)
        cv2.putText(overlay, label, (bbox.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label, (bbox.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay
