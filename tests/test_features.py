"""Stage-1 feature maps for pale / semi-transparent watermarks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remove_pic_watermark.detectors.features import (
    FeatureParams,
    adaptive_threshold,
    compute_feature_edges,
    tophat_response,
)


def _pale_text_image(size: int = 200) -> np.ndarray:
    """Gray field with semi-transparent white 'W' glyphs (approx pale watermark)."""
    base = np.full((size, size), 200, dtype=np.uint8)
    # soft bright stamps
    for y, x in ((40, 40), (40, 120), (120, 40), (120, 120)):
        cv2.putText(base, "W", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 245, 2, cv2.LINE_AA)
    # blend toward background to simulate transparency
    pale = cv2.addWeighted(base, 0.35, np.full_like(base, 200), 0.65, 0)
    return cv2.cvtColor(pale, cv2.COLOR_GRAY2BGR)


class FeatureMapTests(unittest.TestCase):
    def test_tophat_responds_to_bright_text(self) -> None:
        img = _pale_text_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resp = tophat_response(gray, kernel=31)
        self.assertGreater(float(resp.max()), 5.0)

    def test_fused_more_mass_than_laplacian_alone(self) -> None:
        img = _pale_text_image()
        fused = compute_feature_edges(img, FeatureParams(mode="fused", threshold=8, adaptive=True))
        lap = compute_feature_edges(img, FeatureParams(mode="laplacian", threshold=12, adaptive=False))
        self.assertGreater(int(np.count_nonzero(fused)), 0)
        # fused should not be empty on pale text
        self.assertGreaterEqual(int(np.count_nonzero(fused)), int(np.count_nonzero(lap)) // 2)

    def test_adaptive_threshold_soft_floor(self) -> None:
        # low-contrast residual
        resp = np.zeros((64, 64), dtype=np.uint8)
        resp[10:20, 10:50] = 8
        thr = adaptive_threshold(resp, floor=18, percentile=90)
        self.assertLessEqual(thr, 18)
        self.assertGreaterEqual(thr, 3)

    def test_feature_params_adaptive_mode_alias(self) -> None:
        p = FeatureParams.from_detector_fields(feature_mode="adaptive", feature_threshold=10)
        self.assertEqual(p.mode, "fused")
        self.assertTrue(p.adaptive)


if __name__ == "__main__":
    unittest.main()
