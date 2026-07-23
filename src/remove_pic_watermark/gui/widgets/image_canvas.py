"""Image canvas: rect ROI (resize / move / rotate), multi-box labels, paint, pan/zoom.

UX (Figma / PowerPoint / PS):
- Rect tool: drag to create; after release, corner/edge handles resize,
  interior drag (SizeAll) moves, top handle rotates.
- Multi-box mode (YOLO train): create/resize/move/rotate per box.
  Train export: oriented boxes (OBB) keep angle; near-axis strokes snap to 0°.
- Paint tool: free brush + eraser (Alt); right drag pan; Ctrl+wheel zoom.
  Train page converts paint strokes to multi-boxes on stroke end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...services.profile_service import RoiNorm


class CanvasTool(str, Enum):
    RECT = "rect"
    PAINT = "paint"
    ERASE = "erase"
    PAN = "pan"


@dataclass
class _AnnotSnapshot:
    """In-memory annotation state for undo/redo (not written to disk)."""

    multi_boxes: list[QRectF] = field(default_factory=list)
    multi_box_angles: list[float] = field(default_factory=list)
    active_multi_index: int = -1
    roi_rect: QRectF = field(default_factory=QRectF)
    roi_angle: float = 0.0
    roi_norm: RoiNorm | None = None
    paint_mask: QImage | None = None


class ImageCanvas(QWidget):
    roi_changed = Signal(object)  # RoiNorm | None
    mask_changed = Signal()
    image_loaded = Signal(str)
    view_changed = Signal()
    tool_hint = Signal(str)
    empty_clicked = Signal()  # click on empty placeholder → open / import
    image_clicked = Signal(int, int)  # image-pixel (x, y) left-click (color-pick mode)
    # Point-prompt mode (EdgeSAM): (x, y, label) label 1=fg 0=bg
    prompt_point_clicked = Signal(int, int, int)
    multi_boxes_changed = Signal()  # multi-box annotation list updated
    history_changed = Signal()  # undo/redo availability changed
    tool_changed = Signal(object)  # CanvasTool — for syncing tool rails
    # Fired when the user starts drawing/editing on this canvas (for multi-tile focus)
    edit_started = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imageCanvas")
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        # Click-to-sample mode (legacy color pick; rarely used)
        self._color_pick_mode = False
        # EdgeSAM / PS-style point prompts: list of (x, y, label) image pixels
        self._prompt_points: list[tuple[int, int, int]] = []
        self._prompt_point_mode = False  # left=fg, right=bg when True

        self._source: QImage | None = None
        self._source_path: Path | None = None
        self._placeholder = "拖入或打开图片"

        self._tool = CanvasTool.RECT
        self._zoom = 1.0
        self._fit_scale = 1.0
        self._pan = QPointF(0.0, 0.0)
        # Training / multi-instance annotation: local rect + rotation (degrees).
        # YOLO export uses AABB of the rotated rect; the UI keeps the true OBB.
        self._multi_box_mode = False
        self._multi_boxes: list[QRectF] = []
        self._multi_box_angles: list[float] = []
        self._active_multi_index: int = -1  # selected multi-box for edit (-1 = none)

        # interaction mode
        # pan | create_rect | rotate | resize | move | paint | erase
        self._drag: str | None = None
        self._resize_mode: str | None = None  # n,s,e,w,ne,nw,se,sw
        self._press_pos = QPoint()
        self._last_pos = QPoint()
        self._rubber = QRect()
        self._hover_pos: QPoint | None = None  # widget coords for brush ring
        self._resize_origin_rect = QRectF()  # image-space rect at resize/move start
        self._move_start_img = QPointF()  # image-space mouse at move start
        # Last brush sample in image space — for continuous stroke interpolation
        self._last_brush_img: QPointF | None = None

        # rect selection in image pixel coords (axis-aligned before rotation)
        self._roi_rect = QRectF()  # image space, unrotated AABB
        self._roi_angle = 0.0  # degrees
        self._roi_norm: RoiNorm | None = None

        # paint mask: full image size, white = painted
        self._paint_mask: QImage | None = None
        self._brush_radius = 18  # widget pixels (on-screen stamp radius)

        # annotation undo/redo (RAM only)
        self._undo_stack: list[_AnnotSnapshot] = []
        self._redo_stack: list[_AnnotSnapshot] = []
        self._history_limit = 30
        self._history_suspended = False

        self._apply_tool_cursor()
        # Undo/redo shortcuts live on MainWindow (ApplicationShortcut) so they
        # work even when focus is not on the canvas.

    # ---- public API ----

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def roi_norm(self) -> RoiNorm | None:
        return self._roi_norm

    @property
    def roi_angle_deg(self) -> float:
        return self._roi_angle

    @property
    def tool(self) -> CanvasTool:
        return self._tool

    def set_tool(self, tool: CanvasTool | str) -> None:
        new_tool = CanvasTool(tool)
        if self._tool == new_tool:
            return
        self._tool = new_tool
        self._drag = None
        # Leaving rect edit: drop blue selection chrome (boxes stay as green)
        if self._multi_box_mode and self._tool != CanvasTool.RECT:
            self._active_multi_index = -1
            self._clear_selection_state()
        self._apply_tool_cursor()
        # Short status only — detailed usage lives in tooltips on the toolbar
        if self._tool == CanvasTool.PAINT:
            self.tool_hint.emit("涂抹")
        elif self._tool == CanvasTool.ERASE:
            self.tool_hint.emit("擦除")
        elif self._tool == CanvasTool.PAN:
            self.tool_hint.emit("平移")
        elif self._multi_box_mode:
            self.tool_hint.emit("多框标注")
        else:
            self.tool_hint.emit("矩形框选")
        self.tool_changed.emit(self._tool)
        self.update()

    def set_multi_box_mode(self, enabled: bool) -> None:
        """When True, each finished rect is appended; used for YOLO multi-instance labels."""
        self._multi_box_mode = bool(enabled)
        if self._multi_box_mode:
            # Keep current tool (train page may switch to paint); only reset angle.
            self._roi_angle = 0.0
        else:
            self._active_multi_index = -1
        self.update()

    def multi_box_mode(self) -> bool:
        return self._multi_box_mode

    def set_multi_boxes_norm(self, boxes: list[tuple[float, float, float, float]]) -> None:
        """Set boxes as list of (left, top, right, bottom) normalized 0–1 (axis-aligned)."""
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        self._clear_selection_state()
        if self._source is None or self._source.isNull():
            self.update()
            return
        w = float(self._source.width())
        h = float(self._source.height())
        for left, top, right, bottom in boxes:
            x1 = max(0.0, min(w, float(left) * w))
            y1 = max(0.0, min(h, float(top) * h))
            x2 = max(0.0, min(w, float(right) * w))
            y2 = max(0.0, min(h, float(bottom) * h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                self._multi_boxes.append(QRectF(x1, y1, x2 - x1, y2 - y1))
                self._multi_box_angles.append(0.0)
        if self._multi_boxes:
            self._select_multi_box(len(self._multi_boxes) - 1)
        self.update()
        self.multi_boxes_changed.emit()

    def multi_boxes_norm(self) -> list[tuple[float, float, float, float]]:
        """Return (left, top, right, bottom) normalized 0–1 for each box.

        Rotated boxes are exported as their axis-aligned bounding box so YOLO
        labels still cover the watermark; the on-screen OBB keeps its angle.
        """
        self._flush_active_multi_to_list()
        if self._source is None or self._source.isNull():
            return []
        w = float(max(1, self._source.width()))
        h = float(max(1, self._source.height()))
        out: list[tuple[float, float, float, float]] = []
        for i, r in enumerate(self._multi_boxes):
            ang = self._multi_box_angles[i] if i < len(self._multi_box_angles) else 0.0
            x1, y1, x2, y2 = self._rect_angle_to_aabb(r, ang)
            out.append((x1 / w, y1 / h, x2 / w, y2 / h))
        return out

    def clear_multi_boxes(self) -> None:
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        self._clear_selection_state()
        self.update()
        self.multi_boxes_changed.emit()

    def pop_last_multi_box(self) -> bool:
        """Delete active multi-box, or the last one if none selected."""
        self._flush_active_multi_to_list()
        if not self._multi_boxes:
            return False
        self.push_annotation_history()
        idx = self._active_multi_index if 0 <= self._active_multi_index < len(self._multi_boxes) else -1
        if idx < 0:
            idx = len(self._multi_boxes) - 1
        self._multi_boxes.pop(idx)
        if idx < len(self._multi_box_angles):
            self._multi_box_angles.pop(idx)
        self._active_multi_index = -1
        self._clear_selection_state()
        if self._multi_boxes:
            self._select_multi_box(min(idx, len(self._multi_boxes) - 1))
        self.update()
        self.multi_boxes_changed.emit()
        return True

    def paint_mask_component_obbs(self) -> list[tuple[QRectF, float]]:
        """Oriented min-area rects of painted blobs: (local QRectF, angle_deg).

        Angle matches canvas rotation (degrees, image coords, y-down).
        Local rect is axis-aligned around the blob center; size ≈ stroke extent
        along the long/short axes so diagonal watermarks stay tight.
        Does not clear the paint layer.
        """
        import numpy as np

        mask = self.export_paint_mask_numpy()
        if mask is None or self._source is None:
            return []
        try:
            import cv2
        except ImportError:
            ys, xs = np.where(mask > 0)
            if len(xs) < 1:
                return []
            r = QRectF(
                float(xs.min()),
                float(ys.min()),
                float(max(1, xs.max() - xs.min() + 1)),
                float(max(1, ys.max() - ys.min() + 1)),
            )
            return [(r, 0.0)]

        binary = (mask > 0).astype(np.uint8)
        if binary.max() == 0:
            return []
        # Slight dilate so thin brush dabs merge into one box instead of vanishing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out: list[tuple[QRectF, float]] = []
        h_img, w_img = mask.shape[:2]
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < 4 or bw < 1 or bh < 1:
                continue
            ys, xs = np.where(labels == i)
            if len(xs) < 2:
                r = QRectF(float(x), float(y), float(max(1, bw)), float(max(1, bh)))
                out.append((r, 0.0))
                continue
            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            rect, angle = self._min_area_rect_to_local(pts)
            if rect is None:
                r = QRectF(float(x), float(y), float(max(1, bw)), float(max(1, bh)))
                out.append((r, 0.0))
            else:
                # Clamp center inside image; keep size
                cx = min(max(rect.center().x(), 0.0), float(w_img - 1))
                cy = min(max(rect.center().y(), 0.0), float(h_img - 1))
                rw = max(2.0, rect.width())
                rh = max(2.0, rect.height())
                out.append((QRectF(cx - rw / 2.0, cy - rh / 2.0, rw, rh), float(angle)))
        return out

    @staticmethod
    def _min_area_rect_to_local(pts) -> tuple[QRectF | None, float]:
        """OpenCV minAreaRect → (local unrotated QRectF, Qt-compatible angle deg)."""
        import cv2
        import numpy as np

        if pts is None or len(pts) < 2:
            return None, 0.0
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 2:
            x1, y1 = float(pts[0, 0]), float(pts[0, 1])
            x2, y2 = float(pts[1, 0]), float(pts[1, 1])
            return (
                QRectF(min(x1, x2), min(y1, y2), max(2.0, abs(x2 - x1)), max(2.0, abs(y2 - y1))),
                0.0,
            )
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
        # OpenCV: angle in [-90, 0); width is side with that angle.
        # Normalize so width is the longer side and angle is in (-90, 90].
        rw = float(max(1.0, rw))
        rh = float(max(1.0, rh))
        ang = float(angle)
        if rw < rh:
            rw, rh = rh, rw
            ang = ang + 90.0
        # Snap near-axis boxes to 0° so pure horizontal strokes stay detect-friendly
        while ang <= -90.0:
            ang += 180.0
        while ang > 90.0:
            ang -= 180.0
        if abs(ang) < 3.0:
            ang = 0.0
        rect = QRectF(float(cx) - rw / 2.0, float(cy) - rh / 2.0, rw, rh)
        return rect, ang

    def paint_mask_component_rects(self) -> list[QRectF]:
        """Image-space AABBs of painted blobs (legacy helper)."""
        return [r for r, _a in self.paint_mask_component_obbs()]

    def has_paint_mask(self) -> bool:
        return self._paint_mask is not None and not self._paint_mask_empty()

    def commit_paint_mask_as_multi_boxes(self, *, clear_paint: bool = True) -> int:
        """Convert painted blobs into multi-boxes (oriented min-area rects).

        Train UX: stroke disappears, green/selected box appears with matching
        size and tilt so the user can verify instances without manual rotate.
        """
        candidates = self.paint_mask_component_obbs()
        if not candidates:
            return 0
        for r, ang in candidates:
            self._append_multi_box(r, float(ang))
        if clear_paint and self._paint_mask is not None:
            self._paint_mask.fill(Qt.transparent)
        # Select last box so user sees it immediately (handles if RECT tool)
        self._select_multi_box(len(self._multi_boxes) - 1)
        self.multi_boxes_changed.emit()
        self.update()
        return len(candidates)

    def multi_boxes_oriented(
        self,
    ) -> list[tuple[float, float, float, float, float]]:
        """Return (cx, cy, w, h, angle_deg) in image pixels for each multi-box.

        w/h are the local (unrotated) rect size — tight OBB, not AABB.
        """
        self._flush_active_multi_to_list()
        out: list[tuple[float, float, float, float, float]] = []
        for i, r in enumerate(self._multi_boxes):
            if r.isNull() or r.width() < 1 or r.height() < 1:
                continue
            ang = self._multi_angle_at(i)
            out.append(
                (
                    float(r.center().x()),
                    float(r.center().y()),
                    float(r.width()),
                    float(r.height()),
                    float(ang),
                )
            )
        return out

    def multi_boxes_oriented_norm(
        self,
    ) -> list[tuple[float, float, float, float, float]]:
        """Normalized (cx, cy, w, h, angle_deg) for YOLO OBB / detect export."""
        if self._source is None or self._source.isNull():
            return []
        w = float(max(1, self._source.width()))
        h = float(max(1, self._source.height()))
        out: list[tuple[float, float, float, float, float]] = []
        for cx, cy, bw, bh, ang in self.multi_boxes_oriented():
            out.append((cx / w, cy / h, bw / w, bh / h, ang))
        return out

    def set_multi_boxes_oriented_norm(
        self,
        boxes: list[tuple[float, float, float, float, float]],
    ) -> None:
        """Load (cx, cy, w, h, angle_deg) normalized boxes (train reload)."""
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        self._clear_selection_state()
        if self._source is None or self._source.isNull():
            self.update()
            return
        iw = float(self._source.width())
        ih = float(self._source.height())
        for row in boxes:
            if len(row) < 4:
                continue
            cx, cy, bw, bh = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            ang = float(row[4]) if len(row) > 4 else 0.0
            pw = max(2.0, bw * iw)
            ph = max(2.0, bh * ih)
            px = cx * iw - pw / 2.0
            py = cy * ih - ph / 2.0
            self._multi_boxes.append(QRectF(px, py, pw, ph))
            self._multi_box_angles.append(ang)
        if self._multi_boxes:
            self._select_multi_box(len(self._multi_boxes) - 1)
        self.update()
        self.multi_boxes_changed.emit()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(2, min(80, int(radius)))
        # Cursor diameter must track brush size
        if self._tool == CanvasTool.PAINT and self.has_image():
            self._apply_tool_cursor()
        self.update()

    def brush_radius(self) -> int:
        return self._brush_radius

    def can_undo_annotation(self) -> bool:
        return bool(self._undo_stack)

    def can_redo_annotation(self) -> bool:
        return bool(self._redo_stack)

    def clear_annotation_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.history_changed.emit()

    def push_annotation_history(self) -> None:
        """Snapshot current annotation state before a mutating edit."""
        if self._history_suspended or not self.has_image():
            return
        snap = self._capture_annotation()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.history_changed.emit()

    def undo_annotation(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture_annotation())
        snap = self._undo_stack.pop()
        self._restore_annotation(snap)
        self.history_changed.emit()
        return True

    def redo_annotation(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture_annotation())
        snap = self._redo_stack.pop()
        self._restore_annotation(snap)
        self.history_changed.emit()
        return True

    def _capture_annotation(self) -> _AnnotSnapshot:
        self._flush_active_multi_to_list()
        mask_copy = None
        if self._paint_mask is not None and not self._paint_mask.isNull():
            mask_copy = self._paint_mask.copy()
        return _AnnotSnapshot(
            multi_boxes=[QRectF(r) for r in self._multi_boxes],
            multi_box_angles=list(self._multi_box_angles),
            active_multi_index=self._active_multi_index,
            roi_rect=QRectF(self._roi_rect),
            roi_angle=float(self._roi_angle),
            roi_norm=self._roi_norm,
            paint_mask=mask_copy,
        )

    def _restore_annotation(self, snap: _AnnotSnapshot) -> None:
        self._history_suspended = True
        try:
            self._multi_boxes = [QRectF(r) for r in snap.multi_boxes]
            self._multi_box_angles = list(snap.multi_box_angles)
            self._active_multi_index = snap.active_multi_index
            self._roi_rect = QRectF(snap.roi_rect)
            self._roi_angle = float(snap.roi_angle)
            self._roi_norm = snap.roi_norm
            if snap.paint_mask is not None and not snap.paint_mask.isNull():
                self._paint_mask = snap.paint_mask.copy()
            elif self._source is not None:
                self._paint_mask = QImage(self._source.size(), QImage.Format_ARGB32_Premultiplied)
                self._paint_mask.fill(Qt.transparent)
            self.update()
            self.roi_changed.emit(self._roi_norm)
            self.multi_boxes_changed.emit()
            self.mask_changed.emit()
        finally:
            self._history_suspended = False

    def has_image(self) -> bool:
        return self._source is not None and not self._source.isNull()

    def has_selection(self) -> bool:
        if self._paint_mask is not None and not self._paint_mask_empty():
            return True
        if self._multi_box_mode and self._multi_boxes:
            return True
        if not self._roi_rect.isNull() or self._roi_norm is not None:
            return True
        return False

    def source_qimage(self) -> QImage | None:
        """Deep copy of the current base image (for in-memory undo / export)."""
        if self._source is None or self._source.isNull():
            return None
        return self._source.copy()

    def load_qimage(
        self,
        image: QImage,
        *,
        logical_path: Path | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a QImage as the canvas base (clears paint mask)."""
        if image.isNull():
            raise ValueError("无效图片")
        self._source = image.copy()
        if logical_path is not None:
            self._source_path = logical_path
        self._clear_selection_state()
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        self.clear_annotation_history()
        self._paint_mask = QImage(self._source.size(), QImage.Format_ARGB32_Premultiplied)
        self._paint_mask.fill(Qt.transparent)
        if not keep_view:
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
        self._recompute_fit()
        self._apply_tool_cursor()
        self.update()
        self.image_loaded.emit(str(logical_path) if logical_path else "")
        self.roi_changed.emit(None)
        self.mask_changed.emit()
        self.multi_boxes_changed.emit()

    def load_path(self, path: Path, *, keep_view: bool = False) -> None:
        image = QImage(str(path))
        if image.isNull():
            from ...image_io import read_image

            bgr = read_image(path)
            rgb = bgr[:, :, ::-1].copy()
            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            if image.isNull():
                raise ValueError(f"无法打开图片: {path}")
        self.load_qimage(image, logical_path=path, keep_view=keep_view)

    def clear_roi(self) -> None:
        self.push_annotation_history()
        self._clear_selection_state()
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        if self._paint_mask is not None:
            self._paint_mask.fill(Qt.transparent)
        self.update()
        self.roi_changed.emit(None)
        self.mask_changed.emit()
        self.multi_boxes_changed.emit()

    def clear_image(self) -> None:
        self._source = None
        self._source_path = None
        self._paint_mask = None
        self._clear_selection_state()
        self._multi_boxes = []
        self._multi_box_angles = []
        self._active_multi_index = -1
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._fit_scale = 1.0
        self.setCursor(Qt.ArrowCursor)
        self.update()
        self.roi_changed.emit(None)
        self.mask_changed.emit()
        self.multi_boxes_changed.emit()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._recompute_fit()
        self.update()
        self.view_changed.emit()

    def zoom_in(self, factor: float = 1.15) -> None:
        if not self.has_image():
            return
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_at(center, factor)

    def zoom_out(self, factor: float = 1.15) -> None:
        if not self.has_image():
            return
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_at(center, 1.0 / max(1.01, factor))

    def zoom_actual(self) -> None:
        """1:1 — display scale ≈ 1.0 (source pixel = screen pixel)."""
        if not self.has_image() or self._fit_scale <= 0:
            return
        self._zoom = 1.0 / max(1e-6, self._fit_scale)
        self._pan = QPointF(0.0, 0.0)
        self.update()
        self.view_changed.emit()

    def export_paint_mask_numpy(self):
        """Binary mask HxW uint8 (0/255) from paint layer.

        PySide6 returns a memoryview from QImage.bits() (no setsize like PyQt5);
        use sizeInBytes / as_numpy-safe copy.
        """
        import numpy as np

        if self._paint_mask is None or self._source is None:
            return None
        rgba = self._paint_mask.convertToFormat(QImage.Format_RGBA8888)
        w, h = rgba.width(), rgba.height()
        if w <= 0 or h <= 0:
            return None
        byte_count = int(rgba.sizeInBytes())
        expected = h * w * 4
        # Prefer constBits/bits as memoryview (PySide6) or buffer protocol
        try:
            ptr = rgba.constBits()
        except Exception:  # noqa: BLE001
            ptr = rgba.bits()
        if hasattr(ptr, "setsize"):
            # PyQt5 sip.voidptr
            ptr.setsize(byte_count)
            buf = ptr
        else:
            # PySide6 memoryview — already sized; may be larger if padded rows
            buf = ptr
        arr = np.frombuffer(buf, dtype=np.uint8, count=byte_count)
        # Handle possible bytesPerLine padding
        bpl = int(rgba.bytesPerLine())
        if bpl == w * 4 and byte_count >= expected:
            rgba_arr = arr[:expected].reshape((h, w, 4)).copy()
        else:
            # row-padded
            rgba_arr = np.empty((h, w, 4), dtype=np.uint8)
            for y in range(h):
                row = arr[y * bpl : y * bpl + w * 4]
                rgba_arr[y] = row.reshape((w, 4))
        alpha = rgba_arr[:, :, 3]
        # Premultiplied paint stamps can leave low alpha at edges; accept weak dabs
        mask = np.where(alpha > 5, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(mask)) < 1:
            return None
        return mask

    def export_roi_solid_mask_numpy(self):
        """Solid filled ROI (supports rotation) as HxW uint8 mask, or None."""
        import numpy as np

        if self._source is None or self._source.isNull() or self._roi_rect.isNull():
            return None
        h, w = int(self._source.height()), int(self._source.width())
        if h <= 0 or w <= 0:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        if not self._fill_poly_mask(mask, self._roi_image_polygon()):
            # Axis-aligned fallback from roi_norm
            rn = self._roi_norm
            if rn is None:
                return None
            x1 = int(max(0, min(w - 1, round(rn.left * w))))
            y1 = int(max(0, min(h - 1, round(rn.top * h))))
            x2 = int(max(0, min(w, round(rn.right * w))))
            y2 = int(max(0, min(h, round(rn.bottom * h))))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                return None
            mask[y1:y2, x1:x2] = 255
        if int(np.count_nonzero(mask)) < 8:
            return None
        return mask

    def export_multi_boxes_solid_mask_numpy(self):
        """Solid fill of all multi-boxes (and active editing rect) → HxW uint8 mask."""
        import numpy as np

        if self._source is None or self._source.isNull():
            return None
        if not self._multi_box_mode and self._roi_rect.isNull():
            return None
        h, w = int(self._source.height()), int(self._source.width())
        if h <= 0 or w <= 0:
            return None
        self._flush_active_multi_to_list()
        mask = np.zeros((h, w), dtype=np.uint8)
        filled = False
        for i, r in enumerate(self._multi_boxes):
            if r.isNull() or r.width() < 1 or r.height() < 1:
                continue
            ang = self._multi_angle_at(i)
            poly = self._image_polygon_for_rect(r, ang)
            if self._fill_poly_mask(mask, poly):
                filled = True
        # Active rubber/single ROI not yet committed (or single-ROI mode)
        if not self._roi_rect.isNull():
            if self._fill_poly_mask(mask, self._roi_image_polygon()):
                filled = True
        if not filled or int(np.count_nonzero(mask)) < 8:
            return None
        return mask

    def _image_polygon_for_rect(self, rect: QRectF, angle_deg: float):
        """Image-space polygon for a local rect + rotation (same math as ROI)."""
        from PySide6.QtGui import QPolygonF

        if rect.isNull():
            return QPolygonF()
        cx, cy = rect.center().x(), rect.center().y()
        corners = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom()),
        ]
        if abs(angle_deg) < 0.5:
            return QPolygonF(corners)
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(angle_deg)
        t.translate(-cx, -cy)
        return QPolygonF([t.map(p) for p in corners])

    def _fill_poly_mask(self, mask, poly) -> bool:
        """Fill polygon into HxW uint8 mask (in-place). Returns True if filled."""
        import numpy as np

        if poly is None or poly.count() < 3:
            return False
        h, w = mask.shape[:2]
        pts = []
        for i in range(poly.count()):
            p = poly.at(i)
            pts.append([float(p.x()), float(p.y())])
        arr = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        arr[:, 0, 0] = np.clip(arr[:, 0, 0], 0, w - 1e-3)
        arr[:, 0, 1] = np.clip(arr[:, 0, 1], 0, h - 1e-3)
        try:
            import cv2

            cv2.fillPoly(mask, [arr.astype(np.int32)], 255)
            return True
        except Exception:  # noqa: BLE001
            return False

    def export_inpaint_mask_numpy(self):
        """Paint ∪ solid rect(s) → binary mask for refine/inpaint.

        Multi-box mode: all boxes are filled (not only the active blue ROI).
        """
        import numpy as np

        paint = self.export_paint_mask_numpy()
        if self._multi_box_mode:
            boxes = self.export_multi_boxes_solid_mask_numpy()
            # multi path already includes active ROI; avoid double-count single export
            roi = boxes
        else:
            roi = self.export_roi_solid_mask_numpy()
        if paint is None and roi is None:
            return None
        if paint is None:
            return roi
        if roi is None:
            return paint
        if paint.shape != roi.shape:
            return paint
        return np.maximum(paint, roi)

    def delete_selection_or_box(self) -> bool:
        """Delete key semantics: remove one multi-box, else clear paint/ROI on this canvas."""
        if self._multi_box_mode and (
            self._multi_boxes or not self._roi_rect.isNull()
        ):
            if self.pop_last_multi_box():
                return True
        if self.has_selection():
            self.clear_roi()
            return True
        return False

    def set_paint_mask_from_numpy(self, mask_gray) -> bool:
        """Load a binary HxW mask into the paint layer (for template editing)."""
        import numpy as np

        if self._source is None or self._source.isNull():
            return False
        arr = np.asarray(mask_gray)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        h, w = int(self._source.height()), int(self._source.width())
        if arr.shape[0] != h or arr.shape[1] != w:
            try:
                import cv2

                arr = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:  # noqa: BLE001
                return False
        on = arr > 10
        # Premultiplied ARGB (little-endian B,G,R,A): red≈239, a=160 → premult rgb
        a = 160
        r_p, g_p, b_p = (239 * a) // 255, (68 * a) // 255, (68 * a) // 255
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[on, 0] = b_p
        rgba[on, 1] = g_p
        rgba[on, 2] = r_p
        rgba[on, 3] = a
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_ARGB32_Premultiplied).copy()
        self._paint_mask = qimg
        self.mask_changed.emit()
        self.update()
        return True

    # ---- events ----

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._recompute_fit()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if not self.has_image():
            return
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.12 if delta > 0 else 1 / 1.12
            self._zoom_at(event.position(), factor)
            event.accept()
            return
        super().wheelEvent(event)

    def set_color_pick_mode(self, enabled: bool) -> None:
        """When True, left-click samples image pixels (no paint/rect)."""
        self._color_pick_mode = bool(enabled)
        if enabled:
            self._prompt_point_mode = False
        if self._color_pick_mode or self._prompt_point_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._apply_tool_cursor()

    def color_pick_mode(self) -> bool:
        return bool(self._color_pick_mode)

    def set_prompt_point_mode(self, enabled: bool) -> None:
        """When True, left-click = FG, right-click = BG (EdgeSAM prompts)."""
        self._prompt_point_mode = bool(enabled)
        if enabled:
            self._color_pick_mode = False
        if self._prompt_point_mode or self._color_pick_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._apply_tool_cursor()
        self.update()

    def prompt_point_mode(self) -> bool:
        return bool(self._prompt_point_mode)

    def prompt_points(self) -> list[tuple[int, int, int]]:
        return list(self._prompt_points)

    def clear_prompt_points(self) -> None:
        self._prompt_points.clear()
        self.update()

    def add_prompt_point(self, x: int, y: int, label: int) -> None:
        self._prompt_points.append((int(x), int(y), int(label)))
        self.update()

    def sample_bgr_at(self, x: int, y: int) -> tuple[int, int, int] | None:
        """Return BGR (OpenCV order) at integer image pixel, or None if OOB."""
        if self._source is None or self._source.isNull():
            return None
        w, h = self._source.width(), self._source.height()
        if not (0 <= x < w and 0 <= y < h):
            return None
        # QImage pixel is ARGB; convert to BGR for OpenCV pipelines
        c = self._source.pixelColor(x, y)
        return (int(c.blue()), int(c.green()), int(c.red()))

    def source_bgr_numpy(self):
        """Full source as BGR uint8 HxWx3 (copy), or None."""
        import numpy as np

        if self._source is None or self._source.isNull():
            return None
        img = self._source.convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        bpl = int(img.bytesPerLine())
        try:
            ptr = img.constBits()
        except Exception:  # noqa: BLE001
            ptr = img.bits()
        if hasattr(ptr, "setsize"):
            ptr.setsize(img.sizeInBytes())
        buf = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
        if bpl == w * 3:
            rgb = buf[: h * w * 3].reshape((h, w, 3)).copy()
        else:
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            for row in range(h):
                rgb[row] = buf[row * bpl : row * bpl + w * 3].reshape((w, 3))
        # RGB → BGR
        return rgb[:, :, ::-1].copy()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.has_image():
            if event.button() == Qt.LeftButton:
                self.empty_clicked.emit()
                event.accept()
                return
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()
        button = event.button()

        # Color pick (template editor): sample pixel, do not start paint/rect
        if button == Qt.LeftButton and self._color_pick_mode:
            img_pt = self._widget_to_image(pos)
            x, y = int(img_pt.x()), int(img_pt.y())
            if self._source is not None:
                x = max(0, min(self._source.width() - 1, x))
                y = max(0, min(self._source.height() - 1, y))
                self.image_clicked.emit(x, y)
            event.accept()
            return

        # EdgeSAM point prompts: left = foreground, right = background
        if self._prompt_point_mode and button in (Qt.LeftButton, Qt.RightButton):
            img_pt = self._widget_to_image(pos)
            x, y = int(round(img_pt.x())), int(round(img_pt.y()))
            if self._source is not None:
                x = max(0, min(self._source.width() - 1, x))
                y = max(0, min(self._source.height() - 1, y))
                label = 1 if button == Qt.LeftButton else 0
                self.add_prompt_point(x, y, label)
                self.prompt_point_clicked.emit(x, y, label)
            event.accept()
            return

        # Middle button always pans (PS-style)
        if button == Qt.MiddleButton:
            self._drag = "pan"
            self._last_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            self.grabMouse()
            event.accept()
            return

        # Explicit pan tool: left-drag pans
        if button == Qt.LeftButton and self._tool == CanvasTool.PAN:
            self._drag = "pan"
            self._last_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            self.grabMouse()
            event.accept()
            return

        # Right-drag pans for non-pan tools (except while creating rect)
        if button == Qt.RightButton and self._tool != CanvasTool.PAN:
            self._drag = "pan"
            self._last_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            self.grabMouse()
            event.accept()
            return

        if button == Qt.LeftButton and self._tool == CanvasTool.RECT:
            # Multi-box: hit-test handles / body / rotate of any existing box first
            if self._multi_box_mode:
                hit = self._hit_multi_box_interaction(pos)
                if hit is not None:
                    kind, idx, edge = hit
                    self._select_multi_box(idx)
                    self.edit_started.emit()
                    if kind == "rotate":
                        self.push_annotation_history()
                        self._drag = "rotate"
                        self._last_pos = pos
                        self.grabMouse()
                        event.accept()
                        return
                    if kind == "resize" and edge is not None:
                        self.push_annotation_history()
                        self._drag = "resize"
                        self._resize_mode = edge
                        self._resize_origin_rect = QRectF(self._roi_rect)
                        self._press_pos = pos
                        self._last_pos = pos
                        self.grabMouse()
                        event.accept()
                        return
                    if kind == "move":
                        self.push_annotation_history()
                        self._drag = "move"
                        self._resize_origin_rect = QRectF(self._roi_rect)
                        self._move_start_img = self._widget_to_image(pos)
                        self._press_pos = pos
                        self._last_pos = pos
                        self.setCursor(Qt.SizeAllCursor)
                        self.grabMouse()
                        event.accept()
                        return
                # empty area → new box (snapshot first so Ctrl+Z removes it)
                self.edit_started.emit()
                self.push_annotation_history()
                self._active_multi_index = -1
                self._clear_selection_state()
                self._drag = "create_rect"
                self._press_pos = pos
                self._last_pos = pos
                self._rubber = QRect(pos, pos)
                self._roi_angle = 0.0
                self.grabMouse()
                event.accept()
                return

            # Single ROI (profiles / refine single-box)
            if not self._roi_rect.isNull():
                if self._hit_rotate_handle(pos):
                    self.edit_started.emit()
                    self.push_annotation_history()
                    self._drag = "rotate"
                    self._last_pos = pos
                    self.grabMouse()
                    event.accept()
                    return
                edge = self._hit_resize_handle(pos)
                if edge is not None:
                    self.edit_started.emit()
                    self.push_annotation_history()
                    self._drag = "resize"
                    self._resize_mode = edge
                    self._resize_origin_rect = QRectF(self._roi_rect)
                    self._press_pos = pos
                    self._last_pos = pos
                    self.grabMouse()
                    event.accept()
                    return
                if self._hit_roi_body(pos):
                    self.edit_started.emit()
                    self.push_annotation_history()
                    self._drag = "move"
                    self._resize_origin_rect = QRectF(self._roi_rect)
                    self._move_start_img = self._widget_to_image(pos)
                    self._press_pos = pos
                    self._last_pos = pos
                    self.setCursor(Qt.SizeAllCursor)
                    self.grabMouse()
                    event.accept()
                    return
            self.edit_started.emit()
            self.push_annotation_history()
            self._drag = "create_rect"
            self._press_pos = pos
            self._last_pos = pos
            self._rubber = QRect(pos, pos)
            self._roi_angle = 0.0
            self.grabMouse()
            event.accept()
            return

        if button == Qt.LeftButton and self._tool in {CanvasTool.PAINT, CanvasTool.ERASE}:
            # Erase tool always erases; paint tool still supports Alt-erase (PS)
            self.edit_started.emit()
            erase = self._tool == CanvasTool.ERASE or bool(event.modifiers() & Qt.AltModifier)
            self.push_annotation_history()
            self._drag = "erase" if erase else "paint"
            self._hover_pos = pos
            self._last_brush_img = None  # start of a new stroke
            self.setCursor(self._make_brush_cursor(erase=erase))
            self._stroke_brush(pos, erase=erase)
            self.grabMouse()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        self._hover_pos = pos
        if self._drag == "pan":
            delta = pos - self._last_pos
            self._last_pos = pos
            self._pan += QPointF(float(delta.x()), float(delta.y()))
            self.update()
            event.accept()
            return
        if self._drag == "create_rect":
            self._rubber = QRect(self._press_pos, pos).normalized()
            self.update()
            event.accept()
            return
        if self._drag == "rotate":
            self._update_rotation_from_mouse(pos)
            self.update()
            event.accept()
            return
        if self._drag == "resize" and self._resize_mode:
            self._apply_resize_from_mouse(pos)
            self.update()
            event.accept()
            return
        if self._drag == "move":
            self._apply_move_from_mouse(pos)
            self.update()
            event.accept()
            return
        if self._drag in {"paint", "erase"}:
            erase = (
                self._tool == CanvasTool.ERASE
                or self._drag == "erase"
                or bool(event.modifiers() & Qt.AltModifier)
            )
            self._drag = "erase" if erase else "paint"
            self._stroke_brush(pos, erase=erase)
            event.accept()
            return

        # hover cursor
        if self._tool == CanvasTool.PAN and self.has_image():
            self.setCursor(Qt.OpenHandCursor)
        elif self._tool == CanvasTool.RECT and self.has_image():
            if self._multi_box_mode:
                hit = self._hit_multi_box_interaction(pos)
                if hit is not None:
                    kind, _idx, edge = hit
                    if kind == "rotate":
                        self.setCursor(Qt.OpenHandCursor)
                    elif kind == "resize" and edge is not None:
                        self.setCursor(self._cursor_for_resize(edge))
                    else:
                        self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.CrossCursor)
            elif not self._roi_rect.isNull():
                if self._hit_rotate_handle(pos):
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    edge = self._hit_resize_handle(pos)
                    if edge is not None:
                        self.setCursor(self._cursor_for_resize(edge))
                    elif self._hit_roi_body(pos):
                        self.setCursor(Qt.SizeAllCursor)
                    else:
                        self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.CrossCursor)
        elif self._tool in {CanvasTool.PAINT, CanvasTool.ERASE} and self.has_image():
            erase = self._tool == CanvasTool.ERASE or bool(event.modifiers() & Qt.AltModifier)
            self.setCursor(self._make_brush_cursor(erase=erase))
            self.update()  # brush ring follows pointer
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        drag = self._drag
        self._drag = None
        self._last_brush_img = None
        if self.mouseGrabber() is self:
            self.releaseMouse()
        self._apply_tool_cursor()

        if drag in {"paint", "erase"}:
            self.mask_changed.emit()
            event.accept()
            return
        if drag == "pan":
            event.accept()
            return
        if drag == "create_rect" and event.button() == Qt.LeftButton:
            # History was pushed on press; commit or discard rubber band
            self._rubber = QRect(self._press_pos, event.position().toPoint()).normalized()
            if self._rubber.width() < 6 or self._rubber.height() < 6:
                if self._multi_box_mode:
                    self._clear_selection_state()
                    self._active_multi_index = -1
                else:
                    # cancel tiny drag — restore pre-press via undo stack top is wrong;
                    # just clear rubber; empty push already on stack (harmless no-op undo)
                    self._clear_selection_state()
            else:
                self._commit_rubber_to_roi()
                if self._multi_box_mode and not self._roi_rect.isNull():
                    # Keep selected with handles so user can resize/move/rotate immediately
                    self._append_multi_box(QRectF(self._roi_rect), 0.0)
                    self._active_multi_index = len(self._multi_boxes) - 1
                    self.multi_boxes_changed.emit()
                elif not self._multi_box_mode:
                    self.roi_changed.emit(self._roi_norm)
            self.update()
            event.accept()
            return
        if drag == "rotate":
            # History already pushed on press
            self._sync_roi_norm_from_rect()
            if self._multi_box_mode:
                # Keep local size + angle (do NOT expand to AABB — that made boxes
                # jump larger / squarer and snap horizontal). YOLO export still
                # uses AABB via multi_boxes_norm().
                self._flush_active_multi_to_list()
                self.multi_boxes_changed.emit()
            else:
                self.roi_changed.emit(self._roi_norm)
            self.update()
            event.accept()
            return
        if drag == "resize":
            self._resize_mode = None
            if self._roi_rect.width() < 4 or self._roi_rect.height() < 4:
                if self._multi_box_mode:
                    self._delete_active_multi_if_tiny()
                else:
                    # clear_roi also pushes — avoid double push: suspend then clear selection only
                    self._clear_selection_state()
                    self.roi_changed.emit(None)
            else:
                self._sync_roi_norm_from_rect()
                self._flush_active_multi_to_list()
                if self._multi_box_mode:
                    self.multi_boxes_changed.emit()
                else:
                    self.roi_changed.emit(self._roi_norm)
            self.update()
            event.accept()
            return
        if drag == "move":
            self._sync_roi_norm_from_rect()
            self._flush_active_multi_to_list()
            if self._multi_box_mode:
                self.multi_boxes_changed.emit()
            else:
                self.roi_changed.emit(self._roi_norm)
            self.update()
            event.accept()
            return
        if drag in {"paint", "erase"}:
            # Snapshot was taken on press (see mousePress); emit change only
            self.mask_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(11, 18, 32))

        if not self.has_image() or self._source is None:
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            painter.end()
            return

        scale = self._display_scale()
        origin = self._image_origin()
        img_w = self._source.width() * scale
        img_h = self._source.height() * scale
        target = QRectF(origin.x(), origin.y(), img_w, img_h)
        painter.drawImage(target, self._source)

        # paint overlay — always composite the layer (transparent = free); do not
        # gate on sparse empty-scan which can miss thin strokes and look like "no paint"
        if self._paint_mask is not None:
            painter.setOpacity(0.50)
            painter.drawImage(target, self._paint_mask)
            painter.setOpacity(1.0)

        # brush preview ring (paint / erase tools)
        if (
            self._tool in {CanvasTool.PAINT, CanvasTool.ERASE}
            and self._hover_pos is not None
            and self._drag not in {"pan"}
        ):
            r = float(self._brush_radius)
            erase_mode = self._tool == CanvasTool.ERASE
            fill = QColor(148, 163, 184, 40) if erase_mode else QColor(239, 68, 68, 35)
            ring = QColor(203, 213, 225, 220) if erase_mode else QColor(239, 68, 68, 200)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(self._hover_pos), r, r)
            painter.setPen(QPen(ring, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(self._hover_pos), r, r)

        # EdgeSAM prompt points: green = FG, red = BG
        if self._prompt_points and self._source is not None:
            for px, py, lab in self._prompt_points:
                wp = self._image_to_widget(QPointF(float(px), float(py)))
                col = QColor(34, 197, 94) if lab == 1 else QColor(239, 68, 68)
                painter.setBrush(col)
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.drawEllipse(wp, 5.5, 5.5)

        # multi-box annotations (training) — green boxes.
        # Under PAINT/ERASE/PAN: draw every box green (no blue "selected" chrome)
        # so paint→box never looks like the rect tool took over.
        # Under RECT: skip the active box here; it is drawn blue with handles below.
        if self._multi_box_mode and self._multi_boxes and self._source is not None:
            for i, br in enumerate(self._multi_boxes):
                if (
                    self._tool == CanvasTool.RECT
                    and i == self._active_multi_index
                    and not self._roi_rect.isNull()
                ):
                    continue
                ang = self._multi_box_angles[i] if i < len(self._multi_box_angles) else 0.0
                poly = self._widget_polygon_for_rect(br, ang)
                painter.setPen(QPen(QColor(52, 211, 153), 2))
                painter.setBrush(QColor(16, 185, 129, 45))
                painter.drawPolygon(poly)

        # rect rubber while creating
        if self._drag == "create_rect" and not self._rubber.isNull():
            painter.setPen(QPen(QColor(96, 165, 250), 2, Qt.DashLine))
            painter.setBrush(QColor(59, 130, 246, 50))
            painter.drawRect(self._rubber.adjusted(0, 0, -1, -1))

        # Active rect chrome (blue + handles) only while RECT tool is selected.
        show_active = not self._roi_rect.isNull() and self._tool == CanvasTool.RECT
        if show_active:
            poly = self._roi_widget_polygon()
            painter.setPen(QPen(QColor(96, 165, 250), 2))
            painter.setBrush(QColor(59, 130, 246, 55))
            painter.drawPolygon(poly)
            painter.setBrush(QColor(147, 197, 253))
            painter.setPen(QPen(QColor(37, 99, 235), 1))
            for p in self._resize_handle_widget_points().values():
                painter.drawRect(QRectF(p.x() - 3.5, p.y() - 3.5, 7, 7))
            # rotate handle (multi-box: baked to AABB on release for YOLO)
            handle = self._rotate_handle_pos()
            mid = self._roi_widget_center()
            painter.setPen(QPen(QColor(96, 165, 250), 1.5))
            painter.drawLine(mid, handle)
            painter.setBrush(QColor(251, 191, 36))
            painter.setPen(QPen(QColor(245, 158, 11), 1))
            painter.drawEllipse(handle, 6, 6)

        painter.setPen(QColor(148, 163, 184))
        angle_txt = (
            f"  旋转 {self._roi_angle:.0f}°"
            if abs(self._roi_angle) > 0.5
            else ""
        )
        tool_names = {
            CanvasTool.RECT: ("多框" if self._multi_box_mode else "矩形"),
            CanvasTool.PAINT: "涂抹",
            CanvasTool.ERASE: "擦除",
            CanvasTool.PAN: "平移",
        }
        tool_txt = tool_names.get(self._tool, "矩形")
        brush_txt = (
            f"  ·  笔刷 r={self._brush_radius}"
            if self._tool in {CanvasTool.PAINT, CanvasTool.ERASE}
            else ""
        )
        multi_txt = f"  ·  已标 {len(self._multi_boxes)}" if self._multi_box_mode else ""
        painter.drawText(
            10,
            self.height() - 10,
            f"{int(round(self._zoom * 100))}%  ·  {tool_txt}{angle_txt}{brush_txt}{multi_txt}",
        )
        painter.end()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Local undo/redo when canvas has focus (also handled globally by MainWindow)
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            if event.key() == Qt.Key_Z and not (mods & Qt.ShiftModifier):
                if self.undo_annotation():
                    event.accept()
                    return
            if event.key() == Qt.Key_Y or (
                event.key() == Qt.Key_Z and (mods & Qt.ShiftModifier)
            ):
                if self.redo_annotation():
                    event.accept()
                    return
        # Number keys switch tools (PS-style)
        key_map = {
            Qt.Key_1: CanvasTool.RECT,
            Qt.Key_2: CanvasTool.PAINT,
            Qt.Key_3: CanvasTool.ERASE,
            Qt.Key_4: CanvasTool.PAN,
        }
        if event.key() in key_map and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            self.set_tool(key_map[event.key()])
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selection_or_box():
                event.accept()
                return
        super().keyPressEvent(event)

    # ---- transform / geometry ----

    def _clear_selection_state(self) -> None:
        self._roi_rect = QRectF()
        self._roi_angle = 0.0
        self._roi_norm = None
        self._rubber = QRect()

    def _display_scale(self) -> float:
        return max(1e-6, self._fit_scale * self._zoom)

    def _image_origin(self) -> QPointF:
        if self._source is None:
            return QPointF(0, 0)
        scale = self._display_scale()
        img_w = self._source.width() * scale
        img_h = self._source.height() * scale
        base_x = (self.width() - img_w) / 2.0
        base_y = (self.height() - img_h) / 2.0
        return QPointF(base_x + self._pan.x(), base_y + self._pan.y())

    def _recompute_fit(self) -> None:
        if self._source is None or self._source.isNull():
            self._fit_scale = 1.0
            return
        pad = 8
        avail_w = max(1.0, float(self.width() - pad * 2))
        avail_h = max(1.0, float(self.height() - pad * 2))
        sx = avail_w / max(1, self._source.width())
        sy = avail_h / max(1, self._source.height())
        self._fit_scale = min(sx, sy)

    def _zoom_at(self, cursor: QPointF, factor: float) -> None:
        if self._source is None:
            return
        old_scale = self._display_scale()
        new_zoom = min(12.0, max(0.15, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        origin = self._image_origin()
        img_x = (cursor.x() - origin.x()) / old_scale
        img_y = (cursor.y() - origin.y()) / old_scale
        self._zoom = new_zoom
        new_scale = self._display_scale()
        new_origin_x = cursor.x() - img_x * new_scale
        new_origin_y = cursor.y() - img_y * new_scale
        img_w = self._source.width() * new_scale
        img_h = self._source.height() * new_scale
        base_x = (self.width() - img_w) / 2.0
        base_y = (self.height() - img_h) / 2.0
        self._pan = QPointF(new_origin_x - base_x, new_origin_y - base_y)
        self.update()
        self.view_changed.emit()

    def _widget_to_image(self, pos: QPoint) -> QPointF:
        scale = self._display_scale()
        origin = self._image_origin()
        return QPointF((pos.x() - origin.x()) / scale, (pos.y() - origin.y()) / scale)

    def _image_to_widget(self, pt: QPointF) -> QPointF:
        scale = self._display_scale()
        origin = self._image_origin()
        return QPointF(origin.x() + pt.x() * scale, origin.y() + pt.y() * scale)

    def _commit_rubber_to_roi(self) -> None:
        if self._source is None:
            return
        tl = self._widget_to_image(self._rubber.topLeft())
        br = self._widget_to_image(self._rubber.bottomRight())
        x1, x2 = sorted((tl.x(), br.x()))
        y1, y2 = sorted((tl.y(), br.y()))
        x1 = max(0.0, min(self._source.width() - 1.0, x1))
        y1 = max(0.0, min(self._source.height() - 1.0, y1))
        x2 = max(0.0, min(float(self._source.width()), x2))
        y2 = max(0.0, min(float(self._source.height()), y2))
        self._roi_rect = QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
        self._roi_angle = 0.0
        self._sync_roi_norm_from_rect()
        self.roi_changed.emit(self._roi_norm)

    def _sync_roi_norm_from_rect(self) -> None:
        """Store axis-aligned bounding box of rotated rect as RoiNorm."""
        if self._source is None or self._roi_rect.isNull():
            self._roi_norm = None
            return
        poly = self._roi_image_polygon()
        xs = [poly.at(i).x() for i in range(poly.count())]
        ys = [poly.at(i).y() for i in range(poly.count())]
        x1, x2 = max(0.0, min(xs)), min(float(self._source.width()), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(self._source.height()), max(ys))
        w = float(self._source.width())
        h = float(self._source.height())
        self._roi_norm = RoiNorm(left=x1 / w, top=y1 / h, right=x2 / w, bottom=y2 / h)

    def _roi_image_polygon(self):
        from PySide6.QtGui import QPolygonF

        r = self._roi_rect
        cx, cy = r.center().x(), r.center().y()
        corners = [
            QPointF(r.left(), r.top()),
            QPointF(r.right(), r.top()),
            QPointF(r.right(), r.bottom()),
            QPointF(r.left(), r.bottom()),
        ]
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(self._roi_angle)
        t.translate(-cx, -cy)
        return t.map(QPolygonF(corners))

    def _roi_widget_polygon(self):
        from PySide6.QtGui import QPolygonF

        poly = self._roi_image_polygon()
        pts = [self._image_to_widget(poly.at(i)) for i in range(poly.count())]
        return QPolygonF(pts)

    def _roi_widget_center(self) -> QPointF:
        return self._image_to_widget(self._roi_rect.center())

    def _rotate_handle_pos(self) -> QPointF:
        # handle above top-center of unrotated rect, then rotate around center
        r = self._roi_rect
        top_mid = QPointF(r.center().x(), r.top() - max(18.0, r.height() * 0.12))
        cx, cy = r.center().x(), r.center().y()
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(self._roi_angle)
        t.translate(-cx, -cy)
        img_pt = t.map(top_mid)
        return self._image_to_widget(img_pt)

    def _hit_rotate_handle(self, pos: QPoint) -> bool:
        if self._roi_rect.isNull():
            return False
        handle = self._rotate_handle_pos()
        dx = pos.x() - handle.x()
        dy = pos.y() - handle.y()
        return dx * dx + dy * dy <= 12 * 12

    def _resize_handle_widget_points(self) -> dict[str, QPointF]:
        """Corner + edge midpoints in widget space (respects rotation)."""
        if self._roi_rect.isNull():
            return {}
        r = self._roi_rect
        local = {
            "nw": QPointF(r.left(), r.top()),
            "n": QPointF(r.center().x(), r.top()),
            "ne": QPointF(r.right(), r.top()),
            "e": QPointF(r.right(), r.center().y()),
            "se": QPointF(r.right(), r.bottom()),
            "s": QPointF(r.center().x(), r.bottom()),
            "sw": QPointF(r.left(), r.bottom()),
            "w": QPointF(r.left(), r.center().y()),
        }
        cx, cy = r.center().x(), r.center().y()
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(self._roi_angle)
        t.translate(-cx, -cy)
        return {k: self._image_to_widget(t.map(pt)) for k, pt in local.items()}

    def _hit_resize_handle(self, pos: QPoint) -> str | None:
        if self._roi_rect.isNull():
            return None
        best: str | None = None
        best_d = 10 * 10  # hit radius² in widget px
        for name, p in self._resize_handle_widget_points().items():
            dx = pos.x() - p.x()
            dy = pos.y() - p.y()
            d = dx * dx + dy * dy
            if d <= best_d:
                best_d = d
                best = name
        return best

    @staticmethod
    def _cursor_for_resize(mode: str):
        mapping = {
            "n": Qt.SizeVerCursor,
            "s": Qt.SizeVerCursor,
            "e": Qt.SizeHorCursor,
            "w": Qt.SizeHorCursor,
            "ne": Qt.SizeBDiagCursor,
            "sw": Qt.SizeBDiagCursor,
            "nw": Qt.SizeFDiagCursor,
            "se": Qt.SizeFDiagCursor,
        }
        return mapping.get(mode, Qt.ArrowCursor)

    def _apply_resize_from_mouse(self, pos: QPoint) -> None:
        """Resize _roi_rect in local (unrotated) image space from widget mouse."""
        if self._source is None or not self._resize_mode:
            return
        img_pt = self._widget_to_image(pos)
        r0 = self._resize_origin_rect
        cx, cy = r0.center().x(), r0.center().y()
        # mouse → local coords of original rect frame
        t_inv = QTransform()
        t_inv.translate(cx, cy)
        t_inv.rotate(-self._roi_angle)
        t_inv.translate(-cx, -cy)
        local = t_inv.map(img_pt)

        left, top, right, bottom = r0.left(), r0.top(), r0.right(), r0.bottom()
        mode = self._resize_mode
        min_sz = 4.0
        if "n" in mode:
            top = min(local.y(), bottom - min_sz)
        if "s" in mode:
            bottom = max(local.y(), top + min_sz)
        if "w" in mode:
            left = min(local.x(), right - min_sz)
        if "e" in mode:
            right = max(local.x(), left + min_sz)

        # clamp to image
        w_img = float(self._source.width())
        h_img = float(self._source.height())
        left = max(0.0, min(left, w_img - min_sz))
        top = max(0.0, min(top, h_img - min_sz))
        right = max(left + min_sz, min(right, w_img))
        bottom = max(top + min_sz, min(bottom, h_img))
        self._roi_rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        self._sync_roi_norm_from_rect()
        self._flush_active_multi_to_list()
        if not self._multi_box_mode:
            self.roi_changed.emit(self._roi_norm)

    def _apply_move_from_mouse(self, pos: QPoint) -> None:
        """Translate _roi_rect by image-space delta from move start."""
        if self._source is None or self._resize_origin_rect.isNull():
            return
        cur = self._widget_to_image(pos)
        dx = cur.x() - self._move_start_img.x()
        dy = cur.y() - self._move_start_img.y()
        r0 = self._resize_origin_rect
        w_img = float(self._source.width())
        h_img = float(self._source.height())
        nw = r0.width()
        nh = r0.height()
        nx = r0.x() + dx
        ny = r0.y() + dy
        # clamp fully inside image
        nx = max(0.0, min(nx, w_img - nw))
        ny = max(0.0, min(ny, h_img - nh))
        self._roi_rect = QRectF(nx, ny, nw, nh)
        self._sync_roi_norm_from_rect()
        self._flush_active_multi_to_list()
        if not self._multi_box_mode:
            self.roi_changed.emit(self._roi_norm)

    def _hit_roi_body(self, pos: QPoint) -> bool:
        """True if widget pos is inside the rotated ROI polygon (not on handles)."""
        if self._roi_rect.isNull():
            return False
        poly = self._roi_widget_polygon()
        return bool(poly.containsPoint(QPointF(pos), Qt.OddEvenFill))

    def _append_multi_box(self, rect: QRectF, angle_deg: float = 0.0) -> None:
        self._multi_boxes.append(QRectF(rect))
        self._multi_box_angles.append(float(angle_deg))

    def _multi_angle_at(self, index: int) -> float:
        if 0 <= index < len(self._multi_box_angles):
            return float(self._multi_box_angles[index])
        return 0.0

    def _rect_angle_to_aabb(self, rect: QRectF, angle_deg: float) -> tuple[float, float, float, float]:
        """Image-space AABB (x1,y1,x2,y2) of a local rect rotated by angle_deg."""
        if rect.isNull():
            return (0.0, 0.0, 0.0, 0.0)
        if abs(angle_deg) < 0.5:
            return (
                float(rect.left()),
                float(rect.top()),
                float(rect.right()),
                float(rect.bottom()),
            )
        cx, cy = rect.center().x(), rect.center().y()
        corners = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom()),
        ]
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(angle_deg)
        t.translate(-cx, -cy)
        xs: list[float] = []
        ys: list[float] = []
        for p in corners:
            m = t.map(p)
            xs.append(m.x())
            ys.append(m.y())
        return (min(xs), min(ys), max(xs), max(ys))

    def _widget_polygon_for_rect(self, rect: QRectF, angle_deg: float):
        from PySide6.QtGui import QPolygonF

        if rect.isNull():
            return QPolygonF()
        cx, cy = rect.center().x(), rect.center().y()
        corners = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom()),
        ]
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(angle_deg)
        t.translate(-cx, -cy)
        return QPolygonF([self._image_to_widget(t.map(p)) for p in corners])

    def _select_multi_box(self, index: int) -> None:
        if not (0 <= index < len(self._multi_boxes)):
            self._active_multi_index = -1
            self._clear_selection_state()
            return
        self._flush_active_multi_to_list()
        self._active_multi_index = index
        r = self._multi_boxes[index]
        self._roi_rect = QRectF(r)
        self._roi_angle = self._multi_angle_at(index)
        self._sync_roi_norm_from_rect()

    def _flush_active_multi_to_list(self) -> None:
        if not self._multi_box_mode:
            return
        if 0 <= self._active_multi_index < len(self._multi_boxes) and not self._roi_rect.isNull():
            self._multi_boxes[self._active_multi_index] = QRectF(self._roi_rect)
            # Keep angle list aligned with box list
            while len(self._multi_box_angles) < len(self._multi_boxes):
                self._multi_box_angles.append(0.0)
            if self._active_multi_index < len(self._multi_box_angles):
                self._multi_box_angles[self._active_multi_index] = float(self._roi_angle)

    def _delete_active_multi_if_tiny(self) -> None:
        if 0 <= self._active_multi_index < len(self._multi_boxes):
            idx = self._active_multi_index
            self._multi_boxes.pop(idx)
            if idx < len(self._multi_box_angles):
                self._multi_box_angles.pop(idx)
        self._active_multi_index = -1
        self._clear_selection_state()
        self.multi_boxes_changed.emit()

    def _hit_multi_box_interaction(
        self, pos: QPoint
    ) -> tuple[str, int, str | None] | None:
        """Return (kind, index, edge|None) for resize/move/rotate on multi-boxes.

        Prefer the currently active box, then topmost later boxes.
        """
        if not self._multi_boxes:
            return None
        # Temporarily drive handle geometry from each candidate
        saved_roi = QRectF(self._roi_rect)
        saved_angle = self._roi_angle
        saved_idx = self._active_multi_index
        order: list[int] = []
        if 0 <= self._active_multi_index < len(self._multi_boxes):
            order.append(self._active_multi_index)
        for i in range(len(self._multi_boxes) - 1, -1, -1):
            if i not in order:
                order.append(i)
        try:
            for idx in order:
                self._roi_rect = QRectF(self._multi_boxes[idx])
                # Active box uses live angle (may be mid-rotate); others use stored
                if idx == saved_idx:
                    self._roi_angle = saved_angle
                else:
                    self._roi_angle = self._multi_angle_at(idx)
                if self._hit_rotate_handle(pos):
                    return ("rotate", idx, None)
                edge = self._hit_resize_handle(pos)
                if edge is not None:
                    return ("resize", idx, edge)
                # body hit (widget polygon respects rotation)
                if self._hit_roi_body(pos):
                    return ("move", idx, None)
        finally:
            self._roi_rect = saved_roi
            self._roi_angle = saved_angle
            self._active_multi_index = saved_idx
        return None

    def _update_rotation_from_mouse(self, pos: QPoint) -> None:
        center = self._roi_widget_center()
        ang = math.degrees(math.atan2(pos.y() - center.y(), pos.x() - center.x()))
        # handle is above center at angle -90 when unrotated; offset
        self._roi_angle = ang + 90.0
        # normalize -180..180
        while self._roi_angle > 180:
            self._roi_angle -= 360
        while self._roi_angle < -180:
            self._roi_angle += 360
        self._sync_roi_norm_from_rect()
        if not self._multi_box_mode:
            self.roi_changed.emit(self._roi_norm)

    def _stroke_brush(self, widget_pos: QPoint, *, erase: bool) -> None:
        """Paint/erase with gap-filling between mouse samples (smooth fast strokes)."""
        if self._paint_mask is None or self._source is None:
            return
        img_pt = self._widget_to_image(widget_pos)
        scale = self._display_scale()
        r_img = max(1.0, float(self._brush_radius) / scale)
        # Spacing as fraction of radius — dense enough that dabs form a solid ribbon
        step = max(0.35 * r_img, 0.75)

        if self._last_brush_img is None:
            self._stamp_brush_img(img_pt, r_img, erase=erase)
            self._last_brush_img = QPointF(img_pt)
            self.update()
            return

        x0, y0 = float(self._last_brush_img.x()), float(self._last_brush_img.y())
        x1, y1 = float(img_pt.x()), float(img_pt.y())
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        # One painter for the whole segment (faster + smoother)
        painter = QPainter(self._paint_mask)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if erase:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setBrush(Qt.transparent)
            painter.setPen(Qt.NoPen)
        else:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setBrush(QColor(239, 68, 68, 200))
            painter.setPen(Qt.NoPen)
        n = max(1, int(dist / step))
        for i in range(1, n + 1):
            t = i / n
            painter.drawEllipse(QPointF(x0 + dx * t, y0 + dy * t), r_img, r_img)
        painter.end()
        self._last_brush_img = QPointF(img_pt)
        self.update()

    def _stamp_brush_img(self, img_pt: QPointF, r_img: float, *, erase: bool) -> None:
        if self._paint_mask is None:
            return
        painter = QPainter(self._paint_mask)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if erase:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setBrush(Qt.transparent)
            painter.setPen(Qt.NoPen)
        else:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setBrush(QColor(239, 68, 68, 200))
            painter.setPen(Qt.NoPen)
        painter.drawEllipse(img_pt, r_img, r_img)
        painter.end()

    def _stamp_brush(self, widget_pos: QPoint, *, erase: bool) -> None:
        """Single dab at widget position (kept for callers that need one stamp)."""
        if self._paint_mask is None or self._source is None:
            return
        img_pt = self._widget_to_image(widget_pos)
        scale = self._display_scale()
        r_img = max(1.0, float(self._brush_radius) / scale)
        self._stamp_brush_img(img_pt, r_img, erase=erase)
        self.update()

    def _paint_mask_empty(self) -> bool:
        if self._paint_mask is None:
            return True
        # cheap sample scan
        step = max(1, min(self._paint_mask.width(), self._paint_mask.height()) // 64)
        for y in range(0, self._paint_mask.height(), step):
            for x in range(0, self._paint_mask.width(), step):
                if self._paint_mask.pixelColor(x, y).alpha() > 10:
                    return False
        return True

    def _apply_tool_cursor(self) -> None:
        if not self.has_image():
            self.setCursor(Qt.PointingHandCursor)
        elif self._tool == CanvasTool.PAINT:
            self.setCursor(self._make_brush_cursor(erase=False))
        elif self._tool == CanvasTool.ERASE:
            self.setCursor(self._make_brush_cursor(erase=True))
        elif self._tool == CanvasTool.PAN:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def _make_brush_cursor(self, *, erase: bool = False) -> QCursor:
        """Circular cursor whose radius matches on-screen brush radius (widget px).

        System cursors are size-capped (~128px); larger brushes still paint correctly
        and the canvas hover ring always shows the true radius.
        """
        r = int(self._brush_radius)
        # padding so the ring isn't clipped
        pad = 3
        d = r * 2 + pad * 2
        # Windows / Qt cursor soft limit — clamp pixmap, ring still drawn on canvas
        max_d = 128
        scale = 1.0
        if d > max_d:
            scale = max_d / float(d)
            d = max_d
        r_draw = max(2.0, r * scale)
        pm = QPixmap(d, d)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        cx = cy = d / 2.0
        if erase:
            ring = QColor(203, 213, 225)
            fill = QColor(148, 163, 184, 50)
        else:
            ring = QColor(248, 113, 113)
            fill = QColor(239, 68, 68, 45)
        painter.setPen(QPen(QColor(15, 23, 42, 180), 2.0))
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(cx, cy), r_draw, r_draw)
        painter.setPen(QPen(ring, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r_draw, r_draw)
        # center hotspot mark
        painter.setPen(Qt.NoPen)
        painter.setBrush(ring)
        painter.drawEllipse(QPointF(cx, cy), 1.6, 1.6)
        painter.end()
        return QCursor(pm, d // 2, d // 2)
