"""YOLO dataset service (no GUI)."""

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

from remove_pic_watermark.services.yolo_dataset import BoxNorm, YoloDatasetService
from remove_pic_watermark.workspace import Workspace


class YoloDatasetTests(unittest.TestCase):
    def test_box_roundtrip_and_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp) / "workspace").ensure()
            svc = YoloDatasetService(ws).ensure()
            # make two fake images
            for name in ("a.jpg", "b.jpg"):
                img = np.full((100, 120, 3), 80, dtype=np.uint8)
                cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
                path = Path(tmp) / name
                cv2.imencode(".jpg", img)[1].tofile(str(path))
            n = svc.import_paths([Path(tmp) / "a.jpg", Path(tmp) / "b.jpg"], copy=True)
            self.assertEqual(n, 2)
            self.assertEqual(svc.count(), 2)
            boxes = [BoxNorm.from_xyxy_norm(0.1, 0.1, 0.4, 0.4)]
            svc.set_boxes(0, boxes)
            self.assertEqual(svc.labeled_count(), 1)
            self.assertFalse(svc.dataset_uses_obb())
            # reload
            svc2 = YoloDatasetService(ws).ensure()
            items = svc2.load_manifest()
            self.assertEqual(len(items), 2)
            self.assertEqual(len(items[0].boxes), 1)
            yaml_path = svc2.write_data_yaml()
            self.assertTrue(yaml_path.is_file())
            text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("watermark", text)
            self.assertNotIn("task: obb", text)

    def test_obb_label_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp) / "workspace").ensure()
            svc = YoloDatasetService(ws).ensure()
            img = np.full((100, 120, 3), 80, dtype=np.uint8)
            path = Path(tmp) / "c.jpg"
            cv2.imencode(".jpg", img)[1].tofile(str(path))
            svc.import_paths([path], copy=True)
            boxes = [BoxNorm.from_oriented_norm(0.5, 0.5, 0.3, 0.1, angle_deg=-25.0)]
            self.assertTrue(boxes[0].is_obb)
            svc.set_boxes(0, boxes)
            self.assertTrue(svc.dataset_uses_obb())
            # Ultralytics OBB on disk: cls + 4 corners (9 columns)
            label = svc.labels_dir / "c.txt"
            line = label.read_text(encoding="utf-8").strip().split()
            self.assertEqual(len(line), 9)
            yaml_path = svc.write_data_yaml()
            self.assertIn("task: obb", yaml_path.read_text(encoding="utf-8"))
            # reload angles (polygon → minAreaRect; allow a few degrees tolerance)
            svc3 = YoloDatasetService(ws).ensure()
            items = svc3.load_manifest()
            self.assertAlmostEqual(items[0].boxes[0].angle_deg, -25.0, delta=5.0)

    def test_deploy_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp) / "workspace").ensure()
            svc = YoloDatasetService(ws).ensure()
            fake = Path(tmp) / "best.pt"
            fake.write_bytes(b"fake-weights-content-ok")
            dest = svc.deploy_weights(fake)
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.name, "watermark.pt")
            # second deploy creates backup
            fake2 = Path(tmp) / "best2.pt"
            fake2.write_bytes(b"second")
            svc.deploy_weights(fake2)
            self.assertTrue((ws.yolo_dir / "watermark.prev.pt").is_file())


if __name__ == "__main__":
    unittest.main()
