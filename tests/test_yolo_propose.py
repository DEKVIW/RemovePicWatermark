"""YOLO propose → template confirm: never emits raw YOLO without style gates."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remove_pic_watermark.detectors.template_stamp import TemplateStampDetector
from remove_pic_watermark.image_io import write_image
from remove_pic_watermark.models import BBox


class _FakeYolo:
    def __init__(self, boxes: list[BBox]) -> None:
        self.boxes = boxes

    def propose_boxes(self, image: np.ndarray) -> list[dict]:
        return [{"bbox": b, "confidence": 0.9, "class_id": 0} for b in self.boxes]


class YoloProposeConfirmTests(unittest.TestCase):
    def test_primary_hit_ignores_yolo(self) -> None:
        """When template already finds the stamp, YOLO path must not run / not regress."""
        tile = np.zeros((48, 48), dtype=np.uint8)
        cv2.rectangle(tile, (4, 10), (44, 38), 255, 2)
        cv2.putText(tile, "W", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        image = np.full((200, 200, 3), 180, dtype=np.uint8)
        image[40:88, 40:88] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl = root / "template_mask.png"
            sample = root / "sample_crop.png"
            write_image(tpl, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            write_image(sample, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            det = TemplateStampDetector(
                label="t",
                template_path=tpl,
                reference_width=200,
                scale_factors=[1.0],
                min_confidence=0.25,
                feature_mode="laplacian",
                feature_threshold=8,
                multi_instance=False,
                dilate=0,
                mask_expand_ratio=0.0,
                yolo_propose_enabled=True,
                secondary_search_enabled=True,
            )
            # Deliberately wrong YOLO box far from stamp
            det.attach_yolo_proposer(_FakeYolo([BBox(150, 150, 40, 40)]))
            hits = det.detect(image)
            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0].metadata.get("search_pass"), "primary")
            # Hit should be near the real stamp, not the fake YOLO corner
            self.assertLess(hits[0].bbox.x, 100)

    def test_no_yolo_same_as_unattached(self) -> None:
        tile = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(tile, (20, 20), 12, 255, -1)
        image = np.full((160, 160, 3), 40, dtype=np.uint8)
        cv2.circle(image, (50, 100), 14, (220, 220, 220), -1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl = root / "template_mask.png"
            write_image(tpl, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            det = TemplateStampDetector(
                label="t",
                template_path=tpl,
                reference_width=160,
                scale_factors=[1.0],
                min_confidence=0.15,
                feature_mode="tophat",
                feature_threshold=8,
                output_mask_mode="bbox",
                dilate=0,
                mask_expand_ratio=0.0,
                min_structure_alignment=0.0,
                min_residual_density=0.0,
                min_orb_matches=0,
                dual_feature_search=False,
                secondary_search_enabled=False,
                yolo_propose_enabled=True,
            )
            a = det.detect(image)
            det.attach_yolo_proposer(None)
            b = det.detect(image)
            self.assertEqual(len(a), len(b))


if __name__ == "__main__":
    unittest.main()
