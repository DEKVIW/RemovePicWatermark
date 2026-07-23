"""Stage 2/3: residual AI + detect orchestrator + job detect_mode."""

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

from remove_pic_watermark.detectors.orchestrator import DetectOrchestrator, normalize_detect_mode
from remove_pic_watermark.detectors.residual_ai import ResidualAiDetector
from remove_pic_watermark.detectors.template_stamp import TemplateStampDetector
from remove_pic_watermark.detectors.yolo_watermark import (
    YoloWatermarkDetector,
    ensure_yolo_dir,
    normalize_yolo_device,
    probe_yolo,
    resolve_yolo_weights,
)
from remove_pic_watermark.image_io import write_image
from remove_pic_watermark.masking import combine_masks
from remove_pic_watermark.models import BBox
from remove_pic_watermark.services.job_service import JobService, JobSpec
from remove_pic_watermark.workspace import Workspace


def _pale_tiled(size: int = 240, n: int = 3) -> np.ndarray:
    base = np.full((size, size), 195, dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            x = 20 + c * 70
            y = 30 + r * 70
            cv2.putText(base, "W", (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 240, 2, cv2.LINE_AA)
    pale = cv2.addWeighted(base, 0.4, np.full_like(base, 195), 0.6, 0)
    return cv2.cvtColor(pale, cv2.COLOR_GRAY2BGR)


class OrchestratorTests(unittest.TestCase):
    def test_normalize_detect_mode(self) -> None:
        self.assertEqual(normalize_detect_mode("both"), "both")
        self.assertEqual(normalize_detect_mode("AI"), "ai")
        self.assertEqual(normalize_detect_mode(None), "styles")

    def test_residual_finds_regions(self) -> None:
        img = _pale_tiled()
        det = ResidualAiDetector.from_config({"max_instances": 32, "min_area": 40})
        hits = det.detect(img)
        self.assertGreaterEqual(len(hits), 2)
        mask = combine_masks(hits, img.shape[:2])
        self.assertGreater(int(np.count_nonzero(mask)), 100)

    def test_orchestrator_ai_only(self) -> None:
        img = _pale_tiled()
        ai = ResidualAiDetector.from_config({"label": "ai"})
        orch = DetectOrchestrator(style_detectors=[], ai_detectors=[ai], mode="ai")
        hits = orch.detect(img)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(orch.describe()["mode"], "ai")

    def test_orchestrator_both_cascade_skips_ai_when_style_hits(self) -> None:
        """样式+模型: style hit must not union AI paint masks."""
        from remove_pic_watermark.models import Detection

        class StyleHit:
            def detect(self, image):
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[10:40, 10:50] = 255
                return [
                    Detection(
                        label="style",
                        bbox=BBox(10, 10, 40, 30),  # area 1200 on 100x100
                        confidence=0.55,
                        mask=mask,
                        metadata={"detector": "template_stamp"},
                    )
                ]

        class NoisyYolo:
            last_raw_proposals = 3
            last_emitted = 1

            def detect(self, image):
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[0:80, 0:80] = 255
                return [
                    Detection(
                        label="ai_yolo",
                        bbox=BBox(0, 0, 80, 80),
                        confidence=0.9,
                        mask=mask,
                        metadata={"detector": "yolo_watermark"},
                    )
                ]

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        orch = DetectOrchestrator(
            style_detectors=[StyleHit()],
            ai_detectors=[NoisyYolo()],
            mode="both",
            style_size_hints=[(40, 30)],
            style_accept_min_confidence=0.38,
            style_max_image_area_ratio=0.12,
        )
        hits = orch.detect(img)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].label, "style")
        self.assertEqual(orch.last_run_stats.get("cascade_path"), "style_hit_skip_ai")
        self.assertTrue(orch.describe().get("cascade"))

    def test_orchestrator_both_cascade_ai_fill_drops_low_conf(self) -> None:
        from remove_pic_watermark.models import Detection

        class EmptyStyle:
            def detect(self, image):
                return []

        class FakeYolo:
            last_raw_proposals = 2
            last_emitted = 2

            def detect(self, image):
                h, w = image.shape[:2]
                z = np.zeros((h, w), dtype=np.uint8)
                return [
                    Detection(
                        "ai_yolo",
                        BBox(10, 10, 100, 70),
                        0.12,
                        z,
                        {"detector": "yolo_watermark"},
                    ),
                    Detection(
                        "ai_yolo",
                        BBox(50, 50, 100, 70),
                        0.88,
                        z,
                        {"detector": "yolo_watermark"},
                    ),
                ]

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        orch = DetectOrchestrator(
            style_detectors=[EmptyStyle()],
            ai_detectors=[FakeYolo()],
            mode="both",
            style_size_hints=[(80, 60)],
        )
        hits = orch.detect(img)
        self.assertEqual(orch.last_run_stats.get("cascade_path"), "style_miss_ai_fill")
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(float(hits[0].confidence), 0.88, places=5)

    def test_orchestrator_both_weak_style_allows_ai_fill(self) -> None:
        """Low-confidence style TM should not block cascade AI fill."""
        from remove_pic_watermark.models import Detection

        class WeakStyle:
            def detect(self, image):
                h, w = image.shape[:2]
                z = np.zeros((h, w), dtype=np.uint8)
                z[0:200, 0:200] = 255  # huge weak box
                return [
                    Detection(
                        "style",
                        BBox(0, 0, 200, 200),
                        0.30,  # below style_accept_min_confidence 0.38
                        z,
                        {"detector": "template_stamp"},
                    )
                ]

        class FakeYolo:
            last_raw_proposals = 1
            last_emitted = 1

            def detect(self, image):
                z = np.zeros(image.shape[:2], dtype=np.uint8)
                return [
                    Detection(
                        "ai_yolo",
                        BBox(50, 50, 100, 70),
                        0.90,
                        z,
                        {"detector": "yolo_watermark"},
                    )
                ]

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        orch = DetectOrchestrator(
            style_detectors=[WeakStyle()],
            ai_detectors=[FakeYolo()],
            mode="both",
            style_size_hints=[(80, 60)],
        )
        hits = orch.detect(img)
        self.assertEqual(orch.last_run_stats.get("cascade_path"), "style_weak_ai_fill")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].label, "ai_yolo")

    def test_job_ai_only_without_profiles(self) -> None:
        img = _pale_tiled()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = Workspace(root / "workspace")
            ws.ensure()
            src = root / "in.png"
            write_image(src, img)
            service = JobService(ws)
            result = service.run(
                JobSpec(
                    input_path=src,
                    profile_ids=[],
                    backend="opencv",
                    detect_mode="ai",
                    copy_inputs=True,
                )
            )
            self.assertGreaterEqual(result.summary.get("detected", 0), 1)
            self.assertEqual(result.summary.get("detect_mode"), "ai")
            # mask non-empty
            mask_path = Path(result.images[0]["mask"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            self.assertGreater(int(np.count_nonzero(mask)), 0)

    def test_template_refine_intersects_residual(self) -> None:
        """Legacy contour mode: template ∩ residual trims empty padding; sparse falls back."""
        # Synthetic detector without loading a real template file
        det = object.__new__(TemplateStampDetector)
        from remove_pic_watermark.detectors.features import FeatureParams

        det._feature_params = FeatureParams(
            mode="tophat", kernel=15, threshold=8, adaptive=False
        )

        h, w = 64, 64
        # Flat mid-tone + a small bright "glyph" so residual is local, not full-box
        image = np.full((h, w, 3), 160, dtype=np.uint8)
        image[28:36, 28:36] = 250
        bbox = BBox(16, 16, 32, 32)
        # Template patch covers whole box (over-large envelope)
        fat_patch = np.full((32, 32), 255, dtype=np.uint8)
        refined, mode = det._refine_patch_with_residual(image, bbox, fat_patch)
        self.assertIn(mode, {"template_x_residual", "template_fallback"})
        if mode == "template_x_residual":
            self.assertLess(int(np.count_nonzero(refined)), int(np.count_nonzero(fat_patch)))
            self.assertGreater(int(np.count_nonzero(refined)), 0)
        else:
            # Soft residual may still be sparse on synthetic noise-free flats
            self.assertEqual(int(np.count_nonzero(refined)), int(np.count_nonzero(fat_patch)))

        # Empty residual path: uniform crop → fallback to template
        flat = np.full((h, w, 3), 128, dtype=np.uint8)
        kept, mode2 = det._refine_patch_with_residual(flat, bbox, fat_patch)
        self.assertEqual(mode2, "template_fallback")
        self.assertEqual(int(np.count_nonzero(kept)), int(np.count_nonzero(fat_patch)))

    def test_template_bbox_mode_fills_rectangle(self) -> None:
        """Product default: solid bbox mask, not stamp contour."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Circular stamp template
            tpl = np.zeros((40, 40), dtype=np.uint8)
            cv2.circle(tpl, (20, 20), 12, 255, -1)
            tpl_path = root / "template_mask.png"
            cv2.imwrite(str(tpl_path), tpl)
            # Image with similar bright disk bottom-left
            img = np.full((200, 200, 3), 40, dtype=np.uint8)
            cv2.circle(img, (50, 150), 14, (220, 220, 220), -1)
            det = TemplateStampDetector(
                label="t",
                template_path=tpl_path,
                reference_width=200,
                scale_factors=[1.0],
                min_confidence=0.15,
                feature_mode="tophat",
                feature_threshold=8,
                output_mask_mode="bbox",
                dilate=0,
                mask_expand_ratio=0.0,
                search_regions=[{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}],
            )
            hits = det.detect(img)
            self.assertGreaterEqual(len(hits), 1)
            hit = hits[0]
            self.assertEqual(hit.metadata.get("mask_refine"), "bbox_fill")
            self.assertEqual(hit.metadata.get("output_mask_mode"), "bbox")
            # Mask should be a solid rectangle ≈ bbox area (not sparse circle)
            box_area = hit.bbox.width * hit.bbox.height
            nz = int(np.count_nonzero(hit.mask))
            self.assertGreaterEqual(nz, int(box_area * 0.9))

    def test_yolo_resolve_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            yolo_dir = ensure_yolo_dir(models)
            self.assertTrue((yolo_dir / "README.txt").is_file())
            self.assertIsNone(resolve_yolo_weights(models))

            # Named weight discovery
            fake = yolo_dir / "watermark.pt"
            fake.write_bytes(b"not-a-real-model")
            self.assertEqual(resolve_yolo_weights(models), fake)

            probe = probe_yolo(models, try_load=False)
            # Without ultralytics → missing_ultralytics; with it → ready (weights exist)
            self.assertIn(probe.status, {"ready", "missing_ultralytics"})
            if probe.status == "ready":
                self.assertTrue(probe.ready)
            else:
                self.assertEqual(probe.weights, fake)

            # Custom filename still discovered
            fake.unlink()
            custom = yolo_dir / "my_wm.pt"
            custom.write_bytes(b"x")
            self.assertEqual(resolve_yolo_weights(models), custom)

    def test_normalize_yolo_device(self) -> None:
        self.assertEqual(normalize_yolo_device("cpu"), "cpu")
        self.assertEqual(normalize_yolo_device("cuda"), "0")
        self.assertEqual(normalize_yolo_device("gpu"), "0")
        self.assertEqual(normalize_yolo_device("cuda:1"), "1")
        self.assertIn(normalize_yolo_device("auto"), {"cpu", "0"})

    def test_yolo_bbox_residual_refine(self) -> None:
        det = object.__new__(YoloWatermarkDetector)
        det.refine_with_residual = True
        det.residual_keep_ratio = 0.12
        h, w = 64, 64
        image = np.full((h, w, 3), 160, dtype=np.uint8)
        image[28:36, 28:36] = 250
        bbox = BBox(16, 16, 32, 32)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
        refined, mode = det._refine_bbox_mask(image, bbox, mask)
        self.assertIn(mode, {"bbox_x_residual", "bbox_fill"})
        self.assertGreater(int(np.count_nonzero(refined)), 0)

    def test_residual_rejects_full_frame(self) -> None:
        # Large border-like component should be dropped
        img = np.full((200, 200, 3), 180, dtype=np.uint8)
        # bright border frame
        img[0:8, :] = 250
        img[-8:, :] = 250
        img[:, 0:8] = 250
        img[:, -8:] = 250
        det = ResidualAiDetector.from_config(
            {"min_area": 40, "max_area_ratio": 0.5, "reject_border_full": True}
        )
        hits = det.detect(img)
        for hit in hits:
            self.assertLess(hit.bbox.width * hit.bbox.height, 200 * 200 * 0.8)


if __name__ == "__main__":
    unittest.main()
