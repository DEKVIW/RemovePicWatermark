"""Multi-instance template matching (step 0 of detection optimisations)."""

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

from remove_pic_watermark.detectors.matching import box_iou, extract_peaks, nms_by_iou
from remove_pic_watermark.detectors.template_stamp import TemplateStampDetector
from remove_pic_watermark.image_io import write_image
from remove_pic_watermark.masking import combine_masks
from remove_pic_watermark.profiles.models import MatchStrategy
from remove_pic_watermark.services.profile_service import ProfileService, RoiNorm
from remove_pic_watermark.workspace import Workspace


def _stamp_pattern(size: int = 48) -> np.ndarray:
    """High-contrast glyph used as repeating watermark unit."""
    tile = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(tile, (4, 10), (size - 4, size - 10), 255, 2)
    cv2.putText(tile, "W", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    return tile


def _tiled_image(
    rows: int = 3,
    cols: int = 4,
    tile_size: int = 48,
    gap: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return BGR image with white stamps on gray, the stamp mask, and centers."""
    tile = _stamp_pattern(tile_size)
    h = rows * tile_size + (rows + 1) * gap
    w = cols * tile_size + (cols + 1) * gap
    gray = np.full((h, w), 180, dtype=np.uint8)
    positions: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            y = gap + r * (tile_size + gap)
            x = gap + c * (tile_size + gap)
            gray[y : y + tile_size, x : x + tile_size] = np.maximum(
                gray[y : y + tile_size, x : x + tile_size], tile
            )
            positions.append((x, y))
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr, tile, positions


class MatchingUtilsTests(unittest.TestCase):
    def test_extract_peaks_finds_separated_maxima(self) -> None:
        score = np.zeros((40, 40), dtype=np.float32)
        score[5, 5] = 0.9
        score[5, 25] = 0.85
        score[25, 5] = 0.8
        peaks = extract_peaks(
            score, threshold=0.5, max_peaks=10, suppress_width=8, suppress_height=8
        )
        self.assertEqual(len(peaks), 3)
        self.assertAlmostEqual(peaks[0][0], 0.9, places=3)

    def test_nms_keeps_non_overlapping(self) -> None:
        cands = [
            {"score": 0.9, "location": (0, 0), "size": (20, 20)},
            {"score": 0.8, "location": (5, 5), "size": (20, 20)},  # overlap
            {"score": 0.7, "location": (50, 50), "size": (20, 20)},
        ]
        kept = nms_by_iou(cands, iou_threshold=0.3, max_keep=10)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["location"], (0, 0))
        self.assertEqual(kept[1]["location"], (50, 50))

    def test_box_iou_identity(self) -> None:
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)


class MultiInstanceDetectorTests(unittest.TestCase):
    def test_multi_instance_finds_multiple_tiles(self) -> None:
        image, tile, positions = _tiled_image(rows=3, cols=4)
        expected = len(positions)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "tiled.png"
            tpl_path = root / "template_mask.png"
            sample_path = root / "sample_crop.png"
            write_image(img_path, image)
            # template as white-on-black mask + sample crop for edges
            write_image(tpl_path, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            write_image(sample_path, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))

            detector = TemplateStampDetector(
                label="tile",
                template_path=tpl_path,
                reference_width=image.shape[1],
                scale_factors=[1.0],
                min_confidence=0.35,
                feature_mode="laplacian",
                feature_threshold=8,
                multi_instance=True,
                max_instances=32,
                candidate_limit=64,
                nms_iou=0.25,
                multi_score_ratio=0.55,
                dilate=0,
                mask_expand_ratio=0.0,
            )
            dets = detector.detect(image)
            combined = combine_masks(dets, image.shape[:2])
            # Should recover most tiles (allow a few misses on synthetic edges)
            self.assertGreaterEqual(len(dets), expected - 2, msg=f"got {len(dets)} expected ~{expected}")
            self.assertGreater(int(np.count_nonzero(combined)), tile.size)

    def test_single_instance_returns_at_most_one(self) -> None:
        image, tile, _positions = _tiled_image(rows=2, cols=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl_path = root / "template_mask.png"
            sample_path = root / "sample_crop.png"
            write_image(tpl_path, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            write_image(sample_path, cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
            detector = TemplateStampDetector(
                label="tile",
                template_path=tpl_path,
                reference_width=image.shape[1],
                scale_factors=[1.0],
                min_confidence=0.35,
                feature_mode="laplacian",
                feature_threshold=8,
                multi_instance=False,
                dilate=0,
            )
            dets = detector.detect(image)
            self.assertLessEqual(len(dets), 1)

    def test_search_strategy_roi_is_single_instance_full_frame(self) -> None:
        """ROI 建档 + 全图查找：全帧搜索，但默认只取最佳一处（单 logo）。"""
        image, tile, _ = _tiled_image(rows=2, cols=2, tile_size=40, gap=20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = Workspace(root / "workspace")
            ws.ensure()
            img_path = root / "sample.png"
            write_image(img_path, image)
            h, w = image.shape[:2]
            gap, ts = 20, 40
            roi = RoiNorm(
                left=gap / w,
                top=gap / h,
                right=(gap + ts) / w,
                bottom=(gap + ts) / h,
            )
            svc = ProfileService(ws)
            profile, _build, directory = svc.create_from_roi(
                name="single_logo",
                image_path=img_path,
                roi=roi,
                match_strategy=MatchStrategy.SEARCH,
                profile_id="single_logo",
            )
            # Full-frame search, but ROI styles stay single-best to avoid flood FPs
            self.assertFalse(profile.detector.get("multi_instance"))
            self.assertEqual(
                profile.detector.get("search_regions"),
                [{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}],
            )
            config = {
                "label": profile.id,
                "template_path": str(directory / profile.template_file),
                **profile.detector,
            }
            det = TemplateStampDetector.from_config(config, directory / "profile.json")
            self.assertFalse(det.multi_instance)
            dets = det.detect(image)
            self.assertLessEqual(len(dets), 1)
            self.assertGreaterEqual(len(dets), 1)

    def test_paint_strategy_enables_multi_on_profile(self) -> None:
        """涂抹建档默认多实例（满屏重复单元）。"""
        image, tile, positions = _tiled_image(rows=2, cols=3, tile_size=40, gap=20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = Workspace(root / "workspace")
            ws.ensure()
            img_path = root / "sample.png"
            write_image(img_path, image)
            # Paint only the first tile
            h, w = image.shape[:2]
            gap, ts = 20, 40
            paint = np.zeros((h, w), dtype=np.uint8)
            paint[gap : gap + ts, gap : gap + ts] = 255
            svc = ProfileService(ws)
            profile, directory = svc.create_from_paint_mask(
                name="multi_unit",
                image_path=img_path,
                mask_gray=paint,
                match_strategy=MatchStrategy.SEARCH,
            )
            self.assertTrue(profile.detector.get("multi_instance"))
            self.assertEqual(
                profile.detector.get("search_regions"),
                [{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}],
            )
            config = {
                "label": profile.id,
                "template_path": str(directory / profile.template_file),
                **profile.detector,
            }
            det = TemplateStampDetector.from_config(config, directory / "profile.json")
            self.assertTrue(det.multi_instance)
            dets = det.detect(image)
            self.assertGreaterEqual(len(dets), 2)


if __name__ == "__main__":
    unittest.main()
