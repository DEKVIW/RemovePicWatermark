from __future__ import annotations

import json
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

from remove_pic_watermark.config import resolve_config_path
from remove_pic_watermark.image_io import iter_image_files, write_image
from remove_pic_watermark.models import BBox
from remove_pic_watermark.pipeline import run_detection_batch, run_opencv_preview
from remove_pic_watermark.services.profile_service import ProfileService, RoiNorm
from remove_pic_watermark.services.template_builder import (
    build_template_from_roi,
    crop_oriented_roi,
    extract_mask_from_crop,
    extract_mask_with_points,
)
from remove_pic_watermark.workspace import Workspace, slugify


class PathHandlingTests(unittest.TestCase):
    def test_resolves_template_relative_to_config_before_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs" / "custom.json"
            template_path = root / "configs" / "templates" / "stamp.png"
            template_path.parent.mkdir(parents=True)
            template_path.write_bytes(b"template")

            resolved = resolve_config_path("templates/stamp.png", config_path)

            self.assertEqual(resolved, template_path.resolve())

    def test_iter_images_skips_generated_dirs_and_debug_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "keep.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            write_image(root / "debug_keep_out.jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            write_image(root / "masks" / "skip.png", np.zeros((8, 8, 3), dtype=np.uint8))

            images = [path.relative_to(root).as_posix() for path in iter_image_files(root)]

            self.assertEqual(images, ["keep.jpg"])

    def test_slugify(self) -> None:
        self.assertEqual(slugify("NICEDAY Stamp"), "niceday_stamp")


class PipelineTests(unittest.TestCase):
    def test_detection_batch_creates_empty_mask_when_no_detector_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.jpg"
            write_image(image_path, np.zeros((24, 24, 3), dtype=np.uint8))
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"fixed_watermarks": [], "template_watermarks": []}),
                encoding="utf-8",
            )

            report = run_detection_batch(
                image_path,
                root / "masks",
                root / "debug",
                root / "report.json",
                config_path,
                use_profiles=False,
            )

            self.assertEqual(len(report), 1)
            self.assertEqual(report[0]["detections"], [])
            mask = cv2.imread(str(root / "masks" / "input.png"), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_run_opencv_preview_raises_for_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(FileNotFoundError):
                run_opencv_preview(root / "missing", root / "masks", root / "output")

    def test_run_opencv_preview_copies_images_with_empty_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.jpg"
            mask_path = root / "masks" / "input.png"
            image = np.full((24, 24, 3), 120, dtype=np.uint8)
            write_image(image_path, image)
            write_image(mask_path, np.zeros((24, 24), dtype=np.uint8))

            results = run_opencv_preview(image_path, root / "masks", root / "output")

            output_path = root / "output" / "input.jpg"
            self.assertEqual(results[0]["action"], "copied")
            self.assertEqual(image_path.read_bytes(), output_path.read_bytes())


class TemplateAndProfileTests(unittest.TestCase):
    def test_build_template_from_bright_roi(self) -> None:
        image = np.full((200, 300, 3), 40, dtype=np.uint8)
        image[120:170, 30:110] = 220
        result = build_template_from_roi(image, BBox(20, 110, 100, 70), dilate=1)
        self.assertGreater(result.stats["fill_ratio"], 0.0)
        self.assertEqual(result.template_mask.ndim, 2)
        self.assertIn("search_regions", result.suggested_detector)

    def test_oriented_crop_uses_rotation(self) -> None:
        image = np.full((200, 300, 3), 50, dtype=np.uint8)
        image[90:110, 80:220] = 230
        crop0, _ = crop_oriented_roi(image, BBox(100, 70, 100, 60), 0.0)
        crop45, _ = crop_oriented_roi(image, BBox(100, 70, 100, 60), 45.0)
        self.assertEqual(crop0.shape[:2], (60, 100))
        self.assertEqual(crop45.shape[:2], (60, 100))
        # Rotated crop should differ from axis-aligned slice for angled content
        self.assertFalse(np.array_equal(crop0, crop45))

    def test_extract_mask_with_points(self) -> None:
        crop = np.full((80, 160, 3), 60, dtype=np.uint8)
        crop[30:50, 20:140] = 210
        pts = [(40, 40), (80, 40), (120, 40), (5, 5), (150, 5)]
        labs = [1, 1, 1, 0, 0]
        mask, stats = extract_mask_with_points(crop, pts, labs, dilate=1)
        self.assertEqual(mask.shape, (80, 160))
        self.assertGreater(stats["fill_ratio"], 0.02)
        self.assertLess(stats["fill_ratio"], 0.7)
        self.assertTrue(
            str(stats["method"]).startswith("points_residual")
            or stats["method"] in {"points_grabcut", "points_residual_fallback"},
            stats["method"],
        )

    def test_create_profile_from_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = Workspace(root / "workspace").ensure()
            image_path = root / "sample.jpg"
            image = np.full((240, 320, 3), 30, dtype=np.uint8)
            image[160:210, 20:100] = 230
            write_image(image_path, image)

            service = ProfileService(workspace)
            profile, build, directory = service.create_from_roi(
                name="Test Stamp",
                image_path=image_path,
                roi=RoiNorm(0.05, 0.65, 0.35, 0.9),
                description="unit test",
            )

            self.assertTrue((directory / "profile.json").exists())
            self.assertTrue((directory / "template_mask.png").exists())
            self.assertEqual(profile.kind.value, "template")
            self.assertGreater(build.stats["fill_ratio"], 0.0)
            loaded = service.get(profile.id)
            self.assertTrue(loaded.enabled)


if __name__ == "__main__":
    unittest.main()
