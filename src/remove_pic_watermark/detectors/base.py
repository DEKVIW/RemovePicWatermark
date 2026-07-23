"""Detector protocol shared by style matchers and optional AI backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..models import Detection


@runtime_checkable
class WatermarkDetector(Protocol):
    """Any object with detect(image_bgr) -> list[Detection]."""

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover - protocol
        ...


def detector_label(detector: Any) -> str:
    return str(getattr(detector, "label", detector.__class__.__name__))
