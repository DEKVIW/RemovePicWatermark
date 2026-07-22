"""YOLO train dataset: image queue, YOLO-format labels, deploy weights."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..image_io import iter_image_files
from ..workspace import Workspace


# Snap near-zero angles so horizontal labels stay plain detect format
_OBB_ANGLE_EPS = 3.0


@dataclass
class BoxNorm:
    """YOLO box: normalized center x,y and w,h in [0,1].

    Optional ``angle_deg`` (degrees, image y-down) enables YOLO-OBB labels.
    Near-zero angles are treated as axis-aligned detect boxes.
    """

    cx: float
    cy: float
    w: float
    h: float
    class_id: int = 0
    angle_deg: float = 0.0

    @property
    def is_obb(self) -> bool:
        return abs(float(self.angle_deg)) >= _OBB_ANGLE_EPS

    def corner_points_norm(self) -> list[tuple[float, float]]:
        """Four corners (normalized) of the oriented box, for Ultralytics OBB labels.

        Ultralytics OBB datasets use polygon format (not xywhr on disk)::
            cls x1 y1 x2 y2 x3 y3 x4 y4
        """
        import math

        rad = math.radians(float(self.angle_deg))
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = float(self.w) / 2.0, float(self.h) / 2.0
        corners: list[tuple[float, float]] = []
        for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            x = float(self.cx) + cos_a * dx - sin_a * dy
            y = float(self.cy) + sin_a * dx + cos_a * dy
            corners.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
        return corners

    def to_yolo_line(self, *, as_obb: bool | None = None) -> str:
        """Write one label line.

        * detect: ``cls cx cy w h`` (5 numbers after class → 5 columns total with class? 
          actually class + 4 = 5 values; Ultralytics says 5 columns = cls+xywh)
        * OBB: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (9 columns) — Ultralytics polygon OBB
        """
        use_obb = self.is_obb if as_obb is None else bool(as_obb)
        if use_obb:
            pts = self.corner_points_norm()
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            return f"{int(self.class_id)} {flat}"
        return f"{int(self.class_id)} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_xyxy_norm(cls, x1: float, y1: float, x2: float, y2: float, class_id: int = 0) -> "BoxNorm":
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return cls(cx=x1 + w / 2, cy=y1 + h / 2, w=w, h=h, class_id=class_id, angle_deg=0.0)

    @classmethod
    def from_oriented_norm(
        cls,
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle_deg: float = 0.0,
        class_id: int = 0,
    ) -> "BoxNorm":
        ang = float(angle_deg)
        if abs(ang) < _OBB_ANGLE_EPS:
            ang = 0.0
        return cls(
            cx=float(cx),
            cy=float(cy),
            w=max(0.0, float(w)),
            h=max(0.0, float(h)),
            class_id=int(class_id),
            angle_deg=ang,
        )

    @classmethod
    def from_roi_norm(cls, left: float, top: float, right: float, bottom: float, class_id: int = 0) -> "BoxNorm":
        return cls.from_xyxy_norm(left, top, right, bottom, class_id=class_id)

    def to_xyxy_norm(self) -> tuple[float, float, float, float]:
        """Axis-aligned outer bounds (for legacy UI / AABB fallback)."""
        if not self.is_obb:
            x1 = self.cx - self.w / 2
            y1 = self.cy - self.h / 2
            x2 = self.cx + self.w / 2
            y2 = self.cy + self.h / 2
            return (x1, y1, x2, y2)
        import math

        rad = math.radians(float(self.angle_deg))
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        # AABB of rotated rect
        aw = self.w * cos_a + self.h * sin_a
        ah = self.w * sin_a + self.h * cos_a
        return (self.cx - aw / 2, self.cy - ah / 2, self.cx + aw / 2, self.cy + ah / 2)

    def to_oriented_norm(self) -> tuple[float, float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h, float(self.angle_deg))


@dataclass
class DatasetItem:
    """One image in the annotation queue (path relative to images/ or absolute)."""

    image_path: Path
    stem: str
    boxes: list[BoxNorm] = field(default_factory=list)

    @property
    def labeled(self) -> bool:
        return len(self.boxes) > 0


class YoloDatasetService:
    """Manage workspace/yolo_train/dataset for GUI annotation + training."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.yolo_train_dir
        self.dataset_dir = self.root / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.runs_dir = self.root / "runs"
        self._items: list[DatasetItem] = []

    def ensure(self) -> "YoloDatasetService":
        for p in (self.root, self.dataset_dir, self.images_dir, self.labels_dir, self.runs_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.workspace.yolo_dir.mkdir(parents=True, exist_ok=True)
        return self

    def load_manifest(self) -> list[DatasetItem]:
        self.ensure()
        self._items = []
        if self.manifest_path.is_file():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for row in data.get("items") or []:
                    rel = str(row.get("image") or "")
                    path = self.images_dir / Path(rel).name
                    if not path.is_file():
                        # try relative as stored
                        cand = self.dataset_dir / rel
                        if cand.is_file():
                            path = cand
                    if not path.is_file():
                        continue
                    boxes = self._load_label_file(self._label_path_for(path))
                    self._items.append(DatasetItem(image_path=path, stem=path.stem, boxes=boxes))
            except Exception:  # noqa: BLE001
                self._items = []
        # Also pick up any images without manifest
        known = {i.image_path.resolve() for i in self._items}
        for path in sorted(self.images_dir.iterdir()) if self.images_dir.is_dir() else []:
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            if path.resolve() in known:
                continue
            boxes = self._load_label_file(self._label_path_for(path))
            self._items.append(DatasetItem(image_path=path, stem=path.stem, boxes=boxes))
        self._save_manifest()
        return list(self._items)

    def items(self) -> list[DatasetItem]:
        return list(self._items)

    def count(self) -> int:
        return len(self._items)

    def labeled_count(self) -> int:
        return sum(1 for i in self._items if i.labeled)

    def import_paths(self, paths: list[Path], *, copy: bool = True) -> int:
        """Add images into the queue. Returns number newly added."""
        self.ensure()
        existing = {i.image_path.resolve() for i in self._items}
        existing_stems = {i.stem for i in self._items}
        added = 0
        for src in paths:
            src = Path(src)
            if not src.is_file():
                continue
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            dest_name = src.name
            dest = self.images_dir / dest_name
            # avoid stem clash
            if dest.exists() and dest.resolve() != src.resolve():
                n = 2
                while (self.images_dir / f"{src.stem}_{n}{src.suffix}").exists():
                    n += 1
                dest = self.images_dir / f"{src.stem}_{n}{src.suffix}"
            if copy:
                if not dest.exists() or dest.resolve() != src.resolve():
                    shutil.copy2(src, dest)
            else:
                dest = src
            if dest.resolve() in existing:
                continue
            if dest.stem in existing_stems and dest.resolve() not in existing:
                # still add if path different
                pass
            boxes = self._load_label_file(self._label_path_for(dest))
            self._items.append(DatasetItem(image_path=dest, stem=dest.stem, boxes=boxes))
            existing.add(dest.resolve())
            existing_stems.add(dest.stem)
            added += 1
        self._save_manifest()
        return added

    def import_folder(self, folder: Path, *, recursive: bool = False) -> int:
        folder = Path(folder)
        if not folder.is_dir():
            return 0
        files = list(iter_image_files(folder, recursive=recursive))
        return self.import_paths(files, copy=True)

    def set_boxes(self, index: int, boxes: list[BoxNorm]) -> None:
        if index < 0 or index >= len(self._items):
            return
        item = self._items[index]
        item.boxes = list(boxes)
        # Write with OBB lines if any box is rotated (per-file format)
        as_obb = any(b.is_obb for b in item.boxes)
        self._write_label_file(
            self._label_path_for(item.image_path), item.boxes, as_obb=as_obb
        )
        self._save_manifest()

    def dataset_uses_obb(self) -> bool:
        """True if any labeled box has a meaningful rotation (train as YOLO-OBB)."""
        for item in self._items:
            for b in item.boxes:
                if b.is_obb:
                    return True
        return False

    def page_slice(self, page: int, page_size: int) -> list[tuple[int, DatasetItem]]:
        """Return (global_index, item) for a 0-based page."""
        if page_size <= 0:
            page_size = 4
        start = max(0, page * page_size)
        end = min(len(self._items), start + page_size)
        return [(i, self._items[i]) for i in range(start, end)]

    def page_count(self, page_size: int) -> int:
        if page_size <= 0 or not self._items:
            return 1
        return max(1, (len(self._items) + page_size - 1) // page_size)

    def write_data_yaml(self) -> Path:
        """Write Ultralytics data.yaml pointing at this dataset.

        When any box is rotated, rewrite **all** labels as OBB (xywhr) so the
        whole set trains with ``task=obb``. Pure axis-aligned sets stay detect.
        """
        self.ensure()
        use_obb = self.dataset_uses_obb()
        labeled = [i for i in self._items if i.labeled]
        train_list = self.dataset_dir / "train.txt"
        lines = []
        for item in labeled:
            lines.append(str(item.image_path.resolve()).replace("\\", "/"))
            self._write_label_file(
                self._label_path_for(item.image_path),
                item.boxes,
                as_obb=use_obb,
            )
        train_list.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        yaml_path = self.dataset_dir / "data.yaml"
        content = (
            f"path: {self.dataset_dir.resolve().as_posix()}\n"
            f"train: {train_list.resolve().as_posix()}\n"
            f"val: {train_list.resolve().as_posix()}\n"
            "names:\n"
            "  0: watermark\n"
        )
        if use_obb:
            # Hint for our train worker (ultralytics also infers from label cols)
            content += "task: obb\n"
        yaml_path.write_text(content, encoding="utf-8")
        # Sidecar so train worker can pick base weights without re-scanning labels
        meta = {"task": "obb" if use_obb else "detect", "n_labeled": len(labeled)}
        (self.dataset_dir / "train_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Drop stale cache (old 6-col xywhr would poison the next train)
        self.clear_label_cache()
        return yaml_path

    def deploy_weights(self, best_pt: Path) -> Path:
        """Copy trained weights to workspace/models/yolo/watermark.pt (backup previous)."""
        self.workspace.yolo_dir.mkdir(parents=True, exist_ok=True)
        dest = self.workspace.yolo_dir / "watermark.pt"
        best_pt = Path(best_pt)
        if not best_pt.is_file():
            raise FileNotFoundError(f"训练权重不存在: {best_pt}")
        if dest.is_file():
            backup = self.workspace.yolo_dir / "watermark.prev.pt"
            try:
                shutil.copy2(dest, backup)
            except OSError:
                pass
        shutil.copy2(best_pt, dest)
        return dest

    def clear_dataset(self, *, delete_files: bool = True) -> int:
        """Clear the annotation queue so a new import batch won't mix with old data.

        Removes images/labels under dataset/ and resets manifest.
        Does **not** delete trained weights under models/yolo/.
        Returns number of queue items cleared.
        """
        self.ensure()
        n = len(self._items)
        if delete_files:
            for item in list(self._items):
                try:
                    path = Path(item.image_path)
                    # only delete files we own under images_dir
                    if path.is_file() and path.resolve().parent == self.images_dir.resolve():
                        path.unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    label = self._label_path_for(item.image_path)
                    if label.is_file():
                        label.unlink(missing_ok=True)
                except OSError:
                    pass
            # wipe leftovers (orphans not in manifest)
            for folder in (self.images_dir, self.labels_dir):
                if not folder.is_dir():
                    continue
                for p in list(folder.iterdir()):
                    if p.is_file():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            for extra in ("train.txt", "data.yaml"):
                cand = self.dataset_dir / extra
                if cand.is_file():
                    try:
                        cand.unlink()
                    except OSError:
                        pass
        self._items = []
        self._save_manifest()
        return n

    def _label_path_for(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _load_label_file(self, path: Path) -> list[BoxNorm]:
        if not path.is_file():
            return []
        import math

        boxes: list[BoxNorm] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
                # 4 corners (8 numbers) = Ultralytics OBB polygon
                if len(vals) >= 8:
                    poly = vals[:8]
                    box = self._box_from_polygon_norm(poly, class_id=cid)
                    if box is not None:
                        boxes.append(box)
                    continue
                # Legacy mistaken format: cls cx cy w h angle_rad (6 cols) — still load
                if len(vals) == 5:
                    cx, cy, w, h, raw = vals
                    if abs(raw) <= math.pi + 0.01:
                        ang = math.degrees(raw)
                    else:
                        ang = raw
                    # Heuristic: if "w,h" look like sizes (<1.5) and angle small, treat as xywhr
                    boxes.append(
                        BoxNorm(cx=cx, cy=cy, w=w, h=h, class_id=cid, angle_deg=ang)
                    )
                    continue
                # detect: cls cx cy w h
                if len(vals) == 4:
                    cx, cy, w, h = vals
                    boxes.append(
                        BoxNorm(cx=cx, cy=cy, w=w, h=h, class_id=cid, angle_deg=0.0)
                    )
        except Exception:  # noqa: BLE001
            return []
        return boxes

    @staticmethod
    def _box_from_polygon_norm(
        coords: list[float], *, class_id: int = 0
    ) -> "BoxNorm | None":
        """Convert normalized 4-corner polygon to BoxNorm (cx,cy,w,h,angle)."""
        import math

        import numpy as np

        if len(coords) < 8:
            return None
        pts = np.array(coords[:8], dtype=np.float32).reshape(4, 2)
        try:
            import cv2

            (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
            rw, rh = float(max(1e-6, rw)), float(max(1e-6, rh))
            ang = float(angle)
            if rw < rh:
                rw, rh = rh, rw
                ang = ang + 90.0
            while ang <= -90.0:
                ang += 180.0
            while ang > 90.0:
                ang -= 180.0
            if abs(ang) < _OBB_ANGLE_EPS:
                ang = 0.0
            return BoxNorm(
                cx=float(cx),
                cy=float(cy),
                w=rw,
                h=rh,
                class_id=int(class_id),
                angle_deg=ang,
            )
        except Exception:  # noqa: BLE001
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            return BoxNorm.from_xyxy_norm(x1, y1, x2, y2, class_id=class_id)

    def _write_label_file(
        self, path: Path, boxes: list[BoxNorm], *, as_obb: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # as_obb=True → entire file is OBB polygon (9 cols) for YOLO-OBB train
        # as_obb=False → detect cxcywh (5 cols)
        text = "\n".join(b.to_yolo_line(as_obb=as_obb) for b in boxes)
        if text:
            text += "\n"
        path.write_text(text, encoding="utf-8")

    def clear_label_cache(self) -> None:
        """Remove ultralytics labels.cache so format changes are re-scanned."""
        for name in ("labels.cache", "train.cache", "val.cache"):
            cand = self.dataset_dir / name
            if cand.is_file():
                try:
                    cand.unlink()
                except OSError:
                    pass
            # also under labels/
            cand2 = self.labels_dir / name
            if cand2.is_file():
                try:
                    cand2.unlink()
                except OSError:
                    pass

    def _save_manifest(self) -> None:
        self.ensure()
        rows = []
        for item in self._items:
            try:
                rel = item.image_path.name
            except Exception:  # noqa: BLE001
                rel = str(item.image_path)
            rows.append({"image": rel, "stem": item.stem, "n_boxes": len(item.boxes)})
        payload = {"version": 1, "items": rows}
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
