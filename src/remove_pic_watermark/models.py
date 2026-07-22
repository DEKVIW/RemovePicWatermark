from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def clamp(self, image_width: int, image_height: int) -> "BBox":
        x1 = max(0, min(self.x, image_width))
        y1 = max(0, min(self.y, image_height))
        x2 = max(0, min(self.right, image_width))
        y2 = max(0, min(self.bottom, image_height))
        return BBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


@dataclass
class Detection:
    label: str
    bbox: BBox
    confidence: float
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox.to_list(),
            "confidence": round(float(self.confidence), 4),
            "metadata": _json_safe(self.metadata),
        }


def _json_safe(value: Any) -> Any:
    """Convert numpy / nested values so job report.json can always dump."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    # Paths, enums, etc.
    try:
        if hasattr(value, "item") and callable(value.item):
            return value.item()
    except Exception:  # noqa: BLE001
        pass
    return str(value)
