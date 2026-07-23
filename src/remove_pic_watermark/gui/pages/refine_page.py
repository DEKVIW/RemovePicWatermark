"""Multi-image refine: paint mask + LaMa/OpenCV (IOPaint-style).

- Default 1×1 layout; switchable 1×2 / 2×2 grids
- Queue keeps each image's work bitmap + undo in memory (path reference, no permanent copy)
- Export: save-as or overwrite original (confirm)
- Drag-drop import; main toolbar tools/shortcuts via active_canvas()
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...device_info import (
    fallback_dialog_text,
    probe_cuda,
    resolve_runtime_device,
)
from ...workspace import Workspace
from ..theme_style import SPACE_PAGE, SPACE_ROW
from ..widgets.icon_tool_button import IconToolButton
from ..widgets.image_canvas import CanvasTool, ImageCanvas
from ..widgets.page_chrome import ToolBar
from ..widgets.tool_icons import (
    icon_empty_queue,
    icon_next,
    icon_play,
    icon_prev,
    icon_reload,
    icon_remove,
)
from ..workers import RefineRequest, RefineWorker, start_refine_worker

try:
    from qfluentwidgets import BodyLabel, ComboBox
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QComboBox as ComboBox
    from PySide6.QtWidgets import QLabel as BodyLabel

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_LAYOUTS = {
    "1×1": (1, 1),
    "1×2": (1, 2),
    "2×2": (2, 2),
}


def _qimage_to_bgr(image: QImage):
    """Fast RGB888 → BGR numpy copy (no PNG encode; keeps UI responsive)."""
    import numpy as np

    rgb = image.convertToFormat(QImage.Format_RGB888)
    w, h = rgb.width(), rgb.height()
    if w <= 0 or h <= 0:
        raise ValueError("无效图片尺寸")
    byte_count = int(rgb.sizeInBytes())
    try:
        ptr = rgb.constBits()
    except Exception:  # noqa: BLE001
        ptr = rgb.bits()
    if hasattr(ptr, "setsize"):
        ptr.setsize(byte_count)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=byte_count)
    bpl = int(rgb.bytesPerLine())
    if bpl == w * 3 and byte_count >= h * w * 3:
        rgb_arr = arr[: h * w * 3].reshape((h, w, 3)).copy()
    else:
        rgb_arr = np.empty((h, w, 3), dtype=np.uint8)
        for y in range(h):
            row = arr[y * bpl : y * bpl + w * 3]
            rgb_arr[y] = row.reshape((w, 3))
    return rgb_arr[:, :, ::-1].copy()


def _collect_image_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in _IMAGE_EXTS:
                    key = str(child.resolve())
                    if key not in seen:
                        seen.add(key)
                        out.append(child)
        elif p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


@dataclass
class RefineItem:
    """One image in the refine queue — path reference + in-memory work state."""

    source_path: Path
    work_image: QImage | None = None  # None → load from disk on first show
    undo_stack: list[QImage] = field(default_factory=list)
    pending_mask: object | None = None  # np.ndarray HxW uint8, paint not yet run
    pass_index: int = 0
    has_edits: bool = False
    exported: bool = False
    export_path: Path | None = None  # last save-as / overwrite path for cold reload

    def display_name(self) -> str:
        return self.source_path.name

    def reload_path(self) -> Path:
        """Disk path to load when work_image is cold-released."""
        if self.export_path is not None and self.export_path.is_file():
            return self.export_path
        return self.source_path

    def ensure_work_image(self) -> QImage:
        if self.work_image is not None and not self.work_image.isNull():
            return self.work_image
        path = self.reload_path()
        img = QImage(str(path))
        if img.isNull():
            raise ValueError(f"无法打开：{path}")
        self.work_image = img
        return img

    def has_pending_mask(self) -> bool:
        import numpy as np

        m = self.pending_mask
        if m is None:
            return False
        arr = np.asarray(m)
        return int(np.count_nonzero(arr)) >= 8

    def release_bitmap(self) -> None:
        """Drop large in-memory images (keep path + status). Safe for exported cold items."""
        self.work_image = None
        self.undo_stack.clear()
        # Keep pending_mask if user painted but not yet run — still needed for batch


class _RefineTile(QFrame):
    """One refine cell: ImageCanvas + status + per-tile action strip."""

    focus_requested = Signal(object)  # self — click / hover select
    edit_focus_requested = Signal(object)  # self — user started drawing (undo target)
    history_changed = Signal()
    tool_changed = Signal(object)  # CanvasTool
    empty_open = Signal()
    tool_hint = Signal(str)
    # Per-tile actions (page handles; index = item_index)
    restore_requested = Signal(int)
    run_requested = Signal(int)
    remove_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.item_index = -1
        self._active = False
        self._actions_enabled = True
        self._hovering = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Canvas host so action strip can overlay top-right without blocking paint area layout
        self._canvas_host = QWidget()
        self._canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host_layout = QVBoxLayout(self._canvas_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        self.canvas = ImageCanvas()
        self.canvas.set_tool(CanvasTool.PAINT)
        self.canvas.set_brush_radius(20)
        self.canvas.setMinimumHeight(120)
        self.canvas._placeholder = "拖入或打开图片"
        host_layout.addWidget(self.canvas, 1)

        self._action_bar = QWidget(self._canvas_host)
        self._action_bar.setObjectName("refineTileActions")
        self._action_bar.setStyleSheet(
            "QWidget#refineTileActions {"
            "  background: rgba(15, 23, 42, 0.55);"
            "  border-radius: 6px;"
            "}"
            "QWidget#refineTileActions QToolButton {"
            "  background: transparent;"
            "  border: none;"
            "}"
        )
        ab = QHBoxLayout(self._action_bar)
        ab.setContentsMargins(3, 3, 3, 3)
        ab.setSpacing(2)
        self.btn_restore = IconToolButton(icon_reload(), "恢复原图", size=28)
        self.btn_run = IconToolButton(icon_play(), "去除水印", size=28)
        self.btn_remove = IconToolButton(icon_remove(), "移除", size=28)
        for btn, slot in (
            (self.btn_restore, self._emit_restore),
            (self.btn_run, self._emit_run),
            (self.btn_remove, self._emit_remove),
        ):
            btn.clicked.connect(slot)
            ab.addWidget(btn)
        self._action_bar.adjustSize()
        self._action_bar.hide()

        layout.addWidget(self._canvas_host, 1)
        self.status = BodyLabel("空")
        self.status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.status)

        # Refine: multiple solid rects + paint (same multi-box interaction as train)
        self.canvas.set_multi_box_mode(True)
        self.canvas.history_changed.connect(self._on_history)
        self.canvas.tool_changed.connect(self._on_tool)
        self.canvas.empty_clicked.connect(self._on_empty)
        self.canvas.mask_changed.connect(self._on_mask)
        self.canvas.roi_changed.connect(self._on_roi)
        self.canvas.tool_hint.connect(self._on_hint)
        self.canvas.edit_started.connect(self._on_edit_started)
        self.canvas.multi_boxes_changed.connect(self._on_mask)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._canvas_host.setMouseTracking(True)
        self.canvas.setMouseTracking(True)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_action_bar()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovering = True
        self._sync_action_bar_visibility()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovering = False
        self._sync_action_bar_visibility()
        super().leaveEvent(event)

    def _place_action_bar(self) -> None:
        bar = self._action_bar
        bar.adjustSize()
        host = self._canvas_host
        margin = 6
        x = max(0, host.width() - bar.width() - margin)
        bar.move(x, margin)
        bar.raise_()

    def _sync_action_bar_visibility(self) -> None:
        show = self.item_index >= 0 and (self._active or self._hovering)
        self._action_bar.setVisible(show)
        if show:
            self._place_action_bar()

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        self.btn_restore.setEnabled(self._actions_enabled and self.item_index >= 0)
        self.btn_remove.setEnabled(self._actions_enabled and self.item_index >= 0)
        # run enable depends on mask; page may call set_run_enabled after
        if not self._actions_enabled or self.item_index < 0:
            self.btn_run.setEnabled(False)

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run.setEnabled(
            bool(enabled) and self._actions_enabled and self.item_index >= 0
        )

    def _emit_restore(self) -> None:
        if self.item_index >= 0:
            self.focus_requested.emit(self)
            self.restore_requested.emit(self.item_index)

    def _emit_run(self) -> None:
        if self.item_index >= 0:
            self.focus_requested.emit(self)
            self.run_requested.emit(self.item_index)

    def _emit_remove(self) -> None:
        if self.item_index >= 0:
            self.focus_requested.emit(self)
            self.remove_requested.emit(self.item_index)

    def _on_history(self) -> None:
        self.history_changed.emit()

    def _on_tool(self, tool: CanvasTool) -> None:
        self.tool_changed.emit(tool)

    def _on_empty(self) -> None:
        self.empty_open.emit()

    def _on_mask(self) -> None:
        self.history_changed.emit()

    def _on_roi(self, _roi=None) -> None:
        # Rectangle selection is part of the inpaint mask
        self.history_changed.emit()

    def _on_hint(self, text: str) -> None:
        if text:
            self.tool_hint.emit(text)

    def _on_edit_started(self) -> None:
        """Drawing on this canvas marks it as the undo target (not mere click)."""
        self.edit_focus_requested.emit(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.focus_requested.emit(self)
        super().mousePressEvent(event)

    def clear_tile(self) -> None:
        self.item_index = -1
        self._active = False
        self.canvas.clear_image()
        self.status.setText("空")
        self.status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.setStyleSheet("")
        self.set_actions_enabled(True)
        self.set_run_enabled(False)
        self._sync_action_bar_visibility()

    def load_item(self, index: int, item: RefineItem, tool: CanvasTool = CanvasTool.PAINT) -> None:
        self.item_index = index
        try:
            img = item.ensure_work_image()
            self.canvas.load_qimage(
                img, logical_path=item.source_path, keep_view=False
            )
            # load_qimage resets annotation; keep multi-rect mode for refine
            self.canvas.set_multi_box_mode(True)
            if item.pending_mask is not None:
                self.canvas.set_paint_mask_from_numpy(item.pending_mask)
            self.canvas.set_tool(tool)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"无法打开: {exc}")
            self.status.setStyleSheet("color: #f97066; font-size: 11px;")
            self._sync_action_bar_visibility()
            return
        mark = ""
        if item.has_pending_mask():
            mark = " · 已涂"
        if item.has_edits:
            mark = (mark + " · 已修") if mark else " · 已修"
        if item.exported:
            mark += " · 已导出"
        self.status.setText(f"{item.display_name()}{mark}")
        self.status.setStyleSheet("color: #34d399; font-size: 11px;")
        self.set_run_enabled(item.has_pending_mask())
        self._sync_action_bar_visibility()

    def set_active_highlight(self, active: bool) -> None:
        """Quiet focus: no thick blue border (avoids layout jitter / visual noise)."""
        self._active = bool(active) and self.item_index >= 0
        # Keep frame style stable — selection is implied by action bar + status, not border
        self.setStyleSheet("")
        self._sync_action_bar_visibility()


class RefinePage(QWidget):
    """Multi-image refine workbench with grid layout + in-memory queue."""

    status_message = Signal(str)
    chrome_state_changed = Signal()

    _DEVICE_PREFS = ("auto", "cpu", "gpu")
    _MAX_UNDO = 8

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self._items: list[RefineItem] = []
        self._page = 0
        self._rows, self._cols = 1, 1
        self._tiles: list[_RefineTile] = []
        self._active_tile: _RefineTile | None = None
        # Undo target: last tile the user actually edited (not mere click-selection)
        self._last_edit_index: int = -1
        self._tool = CanvasTool.PAINT
        self._brush_radius = 20
        self._busy = False
        self._busy_index: int = -1
        self._batch_queue: list[int] = []
        self._batch_pos: int = 0
        self._batch_ok: int = 0
        self._batch_fail: list[str] = []
        self._thread = None
        self._worker = None
        self._probe = None  # lazy CUDA probe — avoid torch at startup
        self._temp_output: Path | None = None
        self._pending_undo_snapshot: QImage | None = None
        self._backend = "iopaint"
        self._device_pref = "auto"
        self.setAcceptDrops(True)
        self._build_ui()
        self._update_action_enabled()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_PAGE, SPACE_PAGE, SPACE_PAGE, SPACE_PAGE)
        root.setSpacing(SPACE_ROW)

        # Page-local only: layout / paging / clear queue.
        # Open · save/export · run live on the main toolbar (no duplicate icons).
        bar = ToolBar(slim=True)
        self.layout_combo = ComboBox()
        self.layout_combo.addItems(list(_LAYOUTS.keys()))
        self.layout_combo.setCurrentText("1×1")
        self.layout_combo.setToolTip("画布布局")
        self.layout_combo.setMinimumWidth(72)
        self.layout_combo.currentTextChanged.connect(self._on_layout_changed)
        bar.add(self.layout_combo)

        self.prev_btn = IconToolButton(icon_prev(), "上一批（PgUp）", size=32)
        self.prev_btn.setShortcut("PgUp")
        self.prev_btn.clicked.connect(self._prev_page)
        bar.add(self.prev_btn)
        self.next_btn = IconToolButton(icon_next(), "下一批（PgDown）", size=32)
        self.next_btn.setShortcut("PgDown")
        self.next_btn.clicked.connect(self._next_page)
        bar.add(self.next_btn)

        self.page_label = BodyLabel("第 1 / 1 批")
        self.page_label.setProperty("role", "caption")
        bar.add(self.page_label)
        self.stats_label = BodyLabel("0 张")
        self.stats_label.setProperty("role", "caption")
        bar.add(self.stats_label)

        bar.add_spacing(8)
        self.clear_queue_btn = IconToolButton(
            icon_empty_queue(),
            "清空精修队列",
            size=32,
        )
        self.clear_queue_btn.clicked.connect(self._clear_queue)
        bar.add(self.clear_queue_btn)
        bar.add_stretch(1)
        root.addWidget(bar)

        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        root.addWidget(self.grid_host, 1)
        self._rebuild_grid()

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("处理中…")
        self.progress.setFixedHeight(16)
        self.progress.hide()
        root.addWidget(self.progress)

    # ------------------------------------------------------------------ drag-drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    # ------------------------------------------------------------------ layout / grid
    def _page_size(self) -> int:
        return max(1, self._rows * self._cols)

    def _on_layout_changed(self, text: str) -> None:
        pair = _LAYOUTS.get(text) or (1, 1)
        self._rows, self._cols = pair
        # Always restart at batch 1 — keeping old page index lands on the last
        # batch after 1×1 → 1×2/2×2 (e.g. 6 images page 5 → clamped to page 2).
        self._page = 0
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        # Persist visible tiles before destroy
        self._flush_visible_tiles_to_items()
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._tiles = []
        self._active_tile = None
        for r in range(self._rows):
            for c in range(self._cols):
                tile = _RefineTile()
                tile.focus_requested.connect(self._on_tile_focus)
                tile.edit_focus_requested.connect(self._on_tile_edit)
                tile.history_changed.connect(self._on_history_changed)
                tile.tool_changed.connect(self._on_tile_tool)
                tile.empty_open.connect(self._open_images)
                tile.tool_hint.connect(self._on_tool_hint)
                tile.restore_requested.connect(self._on_tile_restore)
                tile.run_requested.connect(self._on_tile_run)
                tile.remove_requested.connect(self._on_tile_remove)
                tile.canvas.set_tool(self._tool)
                tile.canvas.set_brush_radius(self._brush_radius)
                self.grid_layout.addWidget(tile, r, c)
                self._tiles.append(tile)
        self._refresh_page()

    def _flush_visible_tiles_to_items(self) -> None:
        """Save each visible canvas base + inpaint mask (paint and/or rect) into RefineItem."""
        for tile in self._tiles:
            idx = tile.item_index
            if idx < 0 or idx >= len(self._items):
                continue
            item = self._items[idx]
            img = tile.canvas.source_qimage()
            if img is not None and not img.isNull():
                item.work_image = img.copy()
            mask = tile.canvas.export_inpaint_mask_numpy()
            item.pending_mask = mask  # None if empty

    def _visible_indices(self) -> set[int]:
        page_size = self._page_size()
        start = self._page * page_size
        return set(range(start, min(start + page_size, len(self._items))))

    def _release_cold_bitmaps(self) -> None:
        """Release bitmaps for exported items not on the current page (scheme B)."""
        visible = self._visible_indices()
        for i, item in enumerate(self._items):
            if i in visible:
                continue
            if item.exported and not item.has_pending_mask():
                item.release_bitmap()

    def _refresh_page(self, *, prefer_index: int | None = None) -> None:
        page_size = self._page_size()
        total = len(self._items)
        pages = max(1, (total + page_size - 1) // page_size) if total else 1
        self._page = max(0, min(self._page, pages - 1))
        start = self._page * page_size
        for tile in self._tiles:
            tile.clear_tile()
        for slot, tile in enumerate(self._tiles):
            gidx = start + slot
            if gidx >= total:
                break
            tile.load_item(gidx, self._items[gidx], tool=self._tool)
            tile.canvas.set_brush_radius(self._brush_radius)
        self.page_label.setText(f"第 {self._page + 1} / {pages} 批")
        self._update_stats_label()
        self.prev_btn.setEnabled(self._page > 0)
        self.next_btn.setEnabled(self._page + 1 < pages)
        self._release_cold_bitmaps()
        self._sync_tile_action_states()
        # Prefer a specific queue index (e.g. after remove); else first non-empty tile
        focus_tile: _RefineTile | None = None
        if prefer_index is not None and 0 <= prefer_index < total:
            focus_tile = self._tile_for_index(prefer_index)
        if focus_tile is None:
            for tile in self._tiles:
                if tile.item_index >= 0:
                    focus_tile = tile
                    break
        if focus_tile is not None:
            self._set_active_tile(focus_tile)
        else:
            self._active_tile = None
            self.chrome_state_changed.emit()

    def _update_stats_label(self) -> None:
        total = len(self._items)
        painted = sum(1 for it in self._items if it.has_pending_mask())
        edited = sum(1 for it in self._items if it.has_edits)
        parts = [f"{total} 张"]
        if painted:
            parts.append(f"已涂 {painted}")
        if edited:
            parts.append(f"已修 {edited}")
        self.stats_label.setText(" · ".join(parts))

    def _prev_page(self) -> None:
        if self._page > 0:
            self._flush_visible_tiles_to_items()
            self._page -= 1
            self._refresh_page()

    def _next_page(self) -> None:
        page_size = self._page_size()
        pages = max(1, (len(self._items) + page_size - 1) // page_size)
        if self._page + 1 < pages:
            self._flush_visible_tiles_to_items()
            self._page += 1
            self._refresh_page()

    def _clear_queue(self) -> None:
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再清空队列。")
            return
        if not self._items:
            self.status_message.emit("队列已是空的")
            return
        dirty = sum(
            1
            for it in self._items
            if (it.has_edits or it.has_pending_mask()) and not it.exported
        )
        msg = f"将清空队列中的 {len(self._items)} 张图片。\n电脑上的原图和已导出文件不会被删除。"
        if dirty:
            msg += f"\n其中 {dirty} 张尚未导出，清空后无法恢复。"
        msg += "\n\n确定清空？"
        reply = QMessageBox.question(
            self,
            "清空队列",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for it in self._items:
            it.release_bitmap()
            it.pending_mask = None
        self._items.clear()
        self._page = 0
        self._active_tile = None
        self._last_edit_index = -1
        self._refresh_page()
        self.status_message.emit("队列已清空")

    def _on_tile_focus(self, tile: _RefineTile) -> None:
        self._set_active_tile(tile, from_edit=False)

    def _on_tile_edit(self, tile: _RefineTile) -> None:
        """User started rect/paint on this tile → this is the undo target."""
        self._set_active_tile(tile, from_edit=True)

    def _set_active_tile(
        self, tile: _RefineTile | None, *, from_edit: bool = False
    ) -> None:
        prev = self._active_tile
        self._active_tile = tile
        if tile is not prev:
            for t in self._tiles:
                t.set_active_highlight(t is tile)
        elif tile is not None:
            tile.set_active_highlight(True)
        if tile is not None:
            if from_edit and tile.item_index >= 0:
                self._last_edit_index = tile.item_index
            # Avoid setFocus mid-stroke (layout/chrome thrash)
            if not from_edit:
                tile.canvas.setFocus()
            if tile.canvas.tool != self._tool:
                tile.canvas.set_tool(self._tool)
            tile.canvas.set_brush_radius(self._brush_radius)
        self._sync_tile_action_states()
        self.chrome_state_changed.emit()

    def _sync_tile_action_states(self) -> None:
        """Enable/disable per-tile strip; run only when that tile has a valid mask."""
        busy = self._busy
        for tile in self._tiles:
            if tile.item_index < 0:
                tile.set_actions_enabled(False)
                tile.set_run_enabled(False)
                continue
            tile.set_actions_enabled(not busy)
            if busy:
                tile.set_run_enabled(False)
                continue
            # Prefer live canvas mask for visible tiles; fall back to stored pending
            has_mask = False
            live = tile.canvas.export_inpaint_mask_numpy()
            if live is not None:
                import numpy as np

                has_mask = int(np.count_nonzero(np.asarray(live))) >= 8
            elif 0 <= tile.item_index < len(self._items):
                has_mask = self._items[tile.item_index].has_pending_mask()
            tile.set_run_enabled(has_mask)

    def _on_tile_tool(self, tool: CanvasTool) -> None:
        self._tool = CanvasTool(tool)
        for t in self._tiles:
            if t.canvas.tool != self._tool:
                t.canvas.set_tool(self._tool)
        self.chrome_state_changed.emit()

    # ------------------------------------------------------------------ import
    def _open_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "打开图片（可多选）",
            str(self.workspace.root),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if files:
            self._add_paths([Path(f) for f in files])

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹", str(self.workspace.root)
        )
        if folder:
            self._add_paths([Path(folder)])

    def _add_paths(self, paths: list[Path]) -> None:
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再导入。")
            return
        imgs = _collect_image_paths(paths)
        if not imgs:
            self.status_message.emit("未找到图片")
            return
        existing = {str(it.source_path.resolve()) for it in self._items}
        added = 0
        for p in imgs:
            key = str(p.resolve())
            if key in existing:
                continue
            self._items.append(RefineItem(source_path=p))
            existing.add(key)
            added += 1
        if added == 0:
            self.status_message.emit("没有新图片（可能已在队列中）")
            return
        # Jump to last page so new images are visible
        page_size = self._page_size()
        self._page = max(0, (len(self._items) - 1) // page_size)
        self._refresh_page()
        self.status_message.emit(f"已加入 {added} 张（队列 {len(self._items)} 张）")

    # ------------------------------------------------------------------ item helpers
    def _active_item(self) -> RefineItem | None:
        tile = self._active_tile
        if tile is None or tile.item_index < 0:
            # fallback first visible
            for t in self._tiles:
                if t.item_index >= 0:
                    tile = t
                    break
        if tile is None or tile.item_index < 0 or tile.item_index >= len(self._items):
            return None
        return self._items[tile.item_index]

    def _active_index(self) -> int:
        tile = self._active_tile
        if tile is not None and 0 <= tile.item_index < len(self._items):
            return tile.item_index
        for t in self._tiles:
            if 0 <= t.item_index < len(self._items):
                return t.item_index
        return -1

    def _tile_for_index(self, index: int) -> _RefineTile | None:
        for t in self._tiles:
            if t.item_index == index:
                return t
        return None

    # ------------------------------------------------------------------ chrome protocol
    def canvas_tool_caps(self) -> dict:
        return {
            "rect": True,
            "paint": True,
            "erase": True,
            "pan": True,
            "clear": True,
            "brush": True,
        }

    def current_canvas_tool(self) -> CanvasTool:
        c = self.active_canvas()
        return c.tool if c is not None else self._tool

    def canvas_brush_radius(self) -> int:
        return int(self._brush_radius)

    def set_canvas_tool(self, tool: CanvasTool) -> None:
        self._tool = CanvasTool(tool)
        for t in self._tiles:
            t.canvas.set_tool(self._tool)
        self.chrome_state_changed.emit()

    def set_canvas_brush(self, radius: int) -> None:
        self._brush_radius = int(radius)
        for t in self._tiles:
            t.canvas.set_brush_radius(self._brush_radius)

    def active_canvas(self) -> ImageCanvas | None:
        if self._active_tile is not None and self._active_tile.item_index >= 0:
            return self._active_tile.canvas
        for t in self._tiles:
            if t.item_index >= 0:
                return t.canvas
        return self._tiles[0].canvas if self._tiles else None

    def app_action_caps(self) -> set[str]:
        caps = {
            "open",
            "open_folder",
            "export",
            "export_menu",
            "undo",
            "redo",
            "clear",
            "delete_box",
            "restore",
            "zoom_in",
            "zoom_out",
            "zoom_fit",
            "zoom_1x",
            "run",
        }
        if self._busy:
            caps.discard("run")
        return caps

    def toolbar_action_labels(self) -> dict[str, str]:
        """Main-chrome tooltips for this page."""
        return {
            "open": "导入图片",
            "open_folder": "导入文件夹",
            "export": "导出",
            "run": "去除水印",
            "clear": "清除选区（本页全部图）",
        }

    def export_menu_actions(self) -> list[tuple[str, str]]:
        """(label, action_id) for main toolbar save menu."""
        return [
            ("另存当前…", "export"),
            ("覆盖当前原图…", "export_overwrite"),
            ("导出全部已修…", "export_all"),
            ("覆盖全部已修原图…", "export_all_overwrite"),
        ]

    def _tile_for_undo(self) -> _RefineTile | None:
        """Prefer last-edited tile; then active; then any with undo history."""
        if 0 <= self._last_edit_index < len(self._items):
            t = self._tile_for_index(self._last_edit_index)
            if t is not None:
                return t
        if self._active_tile is not None and self._active_tile.item_index >= 0:
            return self._active_tile
        # Fall back: most recently paintable history on visible tiles
        for t in reversed(self._tiles):
            if t.item_index < 0:
                continue
            it = self._items[t.item_index]
            if it.undo_stack or t.canvas.can_undo_annotation():
                return t
        return None

    def can_undo(self) -> bool:
        tile = self._tile_for_undo()
        if tile is None or tile.item_index < 0 or tile.item_index >= len(self._items):
            return False
        item = self._items[tile.item_index]
        if item.undo_stack:
            return True
        return bool(tile.canvas.can_undo_annotation())

    def can_redo(self) -> bool:
        canvas = self.active_canvas()
        return bool(canvas and canvas.can_redo_annotation())

    def is_busy(self) -> bool:
        return bool(self._busy)

    def handle_app_action(self, action: str) -> bool:
        canvas = self.active_canvas()
        if action == "open":
            self._open_images()
            return True
        if action == "open_folder":
            self._open_folder()
            return True
        if action == "export":
            self._export(overwrite=False)
            return True
        if action == "export_overwrite":
            self._export(overwrite=True)
            return True
        if action == "export_all":
            self._export_all_refined(overwrite=False)
            return True
        if action == "export_all_overwrite":
            self._export_all_refined(overwrite=True)
            return True
        if action == "undo":
            self._undo_unified()
            # Always handled so MainWindow does not steal undo onto another tile
            return True
        if action == "redo":
            if canvas is not None:
                canvas.redo_annotation()
                self.chrome_state_changed.emit()
            return True
        if action == "clear":
            self._clear_mask()
            return True
        if action == "delete_box":
            self._delete_selection()
            return True
        if action == "restore":
            self._restore_original()
            return True
        if action == "run":
            self._run()
            return True
        if canvas is None:
            return False
        if action == "zoom_in":
            canvas.zoom_in()
            return True
        if action == "zoom_out":
            canvas.zoom_out()
            return True
        if action == "zoom_fit":
            canvas.reset_view()
            return True
        if action == "zoom_1x":
            canvas.zoom_actual()
            return True
        return False

    def set_backend(self, backend: str) -> None:
        self._backend = "iopaint" if backend in {"iopaint", "lama"} else "opencv"
        self.chrome_state_changed.emit()

    def set_device_preference(self, pref: str) -> None:
        pref = str(pref).lower()
        if pref in {"cuda", "gpu"}:
            self._device_pref = "gpu"
        elif pref == "cpu":
            self._device_pref = "cpu"
        else:
            self._device_pref = "auto"
        # Do not probe CUDA here — keeps app startup free of torch import
        self.chrome_state_changed.emit()

    def current_device_preference(self) -> str:
        return self._device_pref

    def device_status_extra(self) -> str:
        n = len(self._items)
        return f"队列 {n}" if n else ""

    def current_backend(self) -> str:
        return self._backend

    def _update_action_enabled(self) -> None:
        self.chrome_state_changed.emit()

    def _on_history_changed(self) -> None:
        self._sync_tile_action_states()
        self.chrome_state_changed.emit()

    def _on_tool_hint(self, text: str) -> None:
        if text:
            self.status_message.emit(text)

    def _on_tile_restore(self, index: int) -> None:
        self._restore_original(index=index)

    def _on_tile_run(self, index: int) -> None:
        self._run(indices=[index])

    def _on_tile_remove(self, index: int) -> None:
        self._remove_item(index)

    # ------------------------------------------------------------------ edit ops
    def _clear_mask(self) -> None:
        """Toolbar clear: wipe rect/paint on every image on the current page."""
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再清除选区。")
            return
        self._flush_visible_tiles_to_items()
        targets = [t for t in self._tiles if t.item_index >= 0]
        if not targets:
            self.status_message.emit("没有可清除的选区")
            return
        # Only confirm when more than one tile has something to clear
        dirty_n = 0
        for t in targets:
            if t.canvas.has_selection():
                dirty_n += 1
            elif 0 <= t.item_index < len(self._items) and self._items[
                t.item_index
            ].has_pending_mask():
                dirty_n += 1
        if dirty_n == 0:
            self.status_message.emit("没有可清除的选区")
            return
        if dirty_n > 1:
            reply = QMessageBox.question(
                self,
                "清除选区",
                f"将清除本页 {dirty_n} 张图上的矩形与涂抹。\n\n继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        for t in targets:
            t.canvas.clear_roi()
            if 0 <= t.item_index < len(self._items):
                self._items[t.item_index].pending_mask = None
        self.status_message.emit(
            "已清除本页选区" if dirty_n > 1 else "已清除选区"
        )
        self._sync_tile_action_states()
        self.chrome_state_changed.emit()

    def _delete_selection(self) -> None:
        """Delete key: one multi-box on active canvas, else clear that canvas only."""
        if self._busy:
            return
        canvas = self.active_canvas()
        if canvas is None:
            return
        if canvas.delete_selection_or_box():
            self._flush_visible_tiles_to_items()
            self.status_message.emit("已删除选区")
            self._sync_tile_action_states()
            self.chrome_state_changed.emit()
        else:
            self.status_message.emit("没有可删除的选区")

    def _restore_original(self, index: int | None = None) -> None:
        """Restore source file for one queue item. Main toolbar: active; tile: explicit index."""
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再恢复原图。")
            return
        if index is None:
            index = self._active_index()
        if index < 0 or index >= len(self._items):
            QMessageBox.information(self, "提示", "没有可恢复的原图。")
            return
        # Activate tile so main-toolbar state and canvas stay consistent
        tile = self._tile_for_index(index)
        if tile is not None:
            self._set_active_tile(tile)
        item = self._items[index]
        canvas = tile.canvas if tile is not None else None
        if item.has_edits or item.undo_stack:
            reply = QMessageBox.question(
                self,
                "恢复原图",
                f"将放弃「{item.display_name()}」的精修进度。\n\n继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            img = QImage(str(item.source_path))
            if img.isNull():
                raise ValueError(f"无法打开：{item.source_path}")
            item.work_image = img
            item.undo_stack.clear()
            item.pending_mask = None
            item.pass_index = 0
            item.has_edits = False
            item.exported = False
            item.export_path = None
            if canvas is not None:
                canvas.load_qimage(img, logical_path=item.source_path, keep_view=False)
                canvas.set_multi_box_mode(True)
                canvas.clear_roi()  # clear paint after restore
                canvas.set_tool(self._tool)
            self._refresh_page_status_labels()
            self._sync_tile_action_states()
            self.status_message.emit(f"已恢复原图：{item.display_name()}")
            self.chrome_state_changed.emit()
        except ValueError as error:
            QMessageBox.warning(self, "无法打开", str(error))

    def _remove_item(self, index: int) -> None:
        """Drop one image from the refine queue (does not delete files on disk)."""
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再移除。")
            return
        if index < 0 or index >= len(self._items):
            return
        self._flush_visible_tiles_to_items()
        item = self._items[index]
        dirty = (item.has_edits or item.has_pending_mask()) and not item.exported
        msg = (
            f"将「{item.display_name()}」移出精修队列。\n"
            "电脑上的原图和已导出文件不会被删除。"
        )
        if dirty:
            msg += "\n未导出的精修进度将丢失。"
        msg += "\n\n确定移除？"
        reply = QMessageBox.question(
            self,
            "移除",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        item.release_bitmap()
        item.pending_mask = None
        del self._items[index]
        # Keep focus near the hole: same slot index, or previous if tail
        prefer = min(index, len(self._items) - 1) if self._items else None
        self._refresh_page(prefer_index=prefer)
        self.status_message.emit(f"已移除，队列 {len(self._items)} 张")

    def _undo_unified(self) -> None:
        """Undo the last-edited image (not mere click-selection)."""
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再撤销。")
            return
        tile = self._tile_for_undo()
        if tile is None or tile.item_index < 0 or tile.item_index >= len(self._items):
            self.status_message.emit("没有可撤销的操作")
            return
        # Align active tile quietly (no border thrash)
        if self._active_tile is not tile:
            self._set_active_tile(tile, from_edit=False)
        item = self._items[tile.item_index]
        canvas = tile.canvas
        if item.undo_stack:
            image = item.undo_stack.pop()
            if image is None or image.isNull():
                self.chrome_state_changed.emit()
                return
            item.work_image = image.copy()
            canvas.load_qimage(image, logical_path=item.source_path, keep_view=True)
            canvas.set_multi_box_mode(True)
            canvas.set_tool(self._tool)
            if item.pass_index > 0:
                item.pass_index -= 1
            item.has_edits = item.pass_index > 0 or bool(item.undo_stack)
            self._last_edit_index = tile.item_index
            self._refresh_page_status_labels()
            self._sync_tile_action_states()
            self.status_message.emit(f"已撤销上一次去除：{item.display_name()}")
            self.chrome_state_changed.emit()
            return
        if canvas.undo_annotation():
            self._last_edit_index = tile.item_index
            self.status_message.emit(f"已撤销：{item.display_name()}")
            self._sync_tile_action_states()
            self.chrome_state_changed.emit()
            return
        self.status_message.emit("没有可撤销的操作")

    def _refresh_page_status_labels(self) -> None:
        for tile in self._tiles:
            if 0 <= tile.item_index < len(self._items):
                it = self._items[tile.item_index]
                mark = ""
                if it.has_pending_mask():
                    mark = " · 已涂"
                if it.has_edits:
                    mark = (mark + " · 已修") if mark else " · 已修"
                if it.exported:
                    mark += " · 已导出"
                tile.status.setText(f"{it.display_name()}{mark}")
        self._update_stats_label()

    # ------------------------------------------------------------------ run (batch all with valid paint, or explicit indices)
    def _run(self, indices: list[int] | None = None) -> None:
        """Main toolbar: all painted items. Tile strip: pass a single index."""
        try:
            self._run_batch_start(indices=indices)
        except Exception as error:  # noqa: BLE001
            detail = traceback.format_exc()
            self.status_message.emit("精修启动失败")
            QMessageBox.warning(self, "去除失败", str(error)[:500])
            self._reset_busy_ui()

    def _collect_run_indices(self) -> list[int]:
        """Flush visible canvases, then every queue item with a valid paint mask."""
        self._flush_visible_tiles_to_items()
        return [i for i, it in enumerate(self._items) if it.has_pending_mask()]

    def _run_batch_start(self, indices: list[int] | None = None) -> None:
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请稍后再试。")
            return
        if not self._items:
            QMessageBox.information(self, "提示", "请先导入图片。")
            return
        if indices is None:
            # Main toolbar: every item with a valid paint mask
            indices = self._collect_run_indices()
        else:
            # Tile strip (or explicit list): flush then keep only valid painted indices
            self._flush_visible_tiles_to_items()
            indices = [
                i
                for i in indices
                if 0 <= i < len(self._items) and self._items[i].has_pending_mask()
            ]
        if not indices:
            QMessageBox.information(
                self,
                "提示",
                "请先框选或涂抹要去除的水印区域。",
            )
            return

        use_lama = self._backend != "opencv"
        model_name = "高质量" if use_lama else "快速"
        self._batch_queue = indices
        self._batch_pos = 0
        self._batch_ok = 0
        self._batch_fail = []
        self._busy = True
        self.progress.show()
        n = len(indices)
        self.progress.setRange(0, n)
        self.progress.setValue(0)
        self.progress.setFormat(f"准备中… 0/{n}")
        self.status_message.emit(f"开始精修 {n} 张（{model_name}）…")
        self._sync_tile_action_states()
        self.chrome_state_changed.emit()
        QApplication.processEvents()
        self._start_next_in_batch()

    def _start_next_in_batch(self) -> None:
        if self._batch_pos >= len(self._batch_queue):
            self._finish_batch()
            return

        idx = self._batch_queue[self._batch_pos]
        item = self._items[idx]
        n = len(self._batch_queue)
        cur = self._batch_pos + 1
        self._busy_index = idx
        self.progress.setValue(self._batch_pos)
        self.progress.setFormat(f"{cur}/{n} {item.display_name()}")
        self.status_message.emit(f"精修 {cur}/{n}：{item.display_name()}")
        QApplication.processEvents()

        try:
            base = item.ensure_work_image()
        except ValueError as error:
            self._batch_fail.append(f"{item.display_name()}: {error}")
            self._batch_pos += 1
            self._start_next_in_batch()
            return

        mask = item.pending_mask
        import numpy as np

        if mask is None or int(np.count_nonzero(np.asarray(mask))) < 8:
            self._batch_fail.append(f"{item.display_name()}: 涂抹无效")
            self._batch_pos += 1
            self._start_next_in_batch()
            return

        self._pending_undo_snapshot = base.copy()
        try:
            image_bgr = _qimage_to_bgr(base)
        except ValueError as error:
            self._pending_undo_snapshot = None
            self._batch_fail.append(f"{item.display_name()}: {error}")
            self._batch_pos += 1
            self._start_next_in_batch()
            return

        use_lama = self._backend != "opencv"
        backend = "iopaint" if use_lama else "opencv"
        device = "cpu"
        if use_lama:
            device, _, _ = resolve_runtime_device(
                self.current_device_preference(), self._probe
            )

        request = RefineRequest(
            backend=backend,
            device=device,
            model_dir=self.workspace.models_dir,
            output_path=None,
            image_bgr=image_bgr,
            mask=np.asarray(mask),
        )
        worker = RefineWorker(self.workspace, request)
        worker.stage.connect(self._on_stage, Qt.ConnectionType.QueuedConnection)
        worker.finished_ok.connect(self._on_ok, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_failed, Qt.ConnectionType.QueuedConnection)
        self._worker = worker
        self._thread = start_refine_worker(self, worker)
        # Do not connect finished→_on_thread_finished for batch; we chain in _on_ok/_on_failed

    def _on_stage(self, text: str) -> None:
        n = len(self._batch_queue) or 1
        cur = self._batch_pos + 1
        self.progress.setFormat(f"{cur}/{n} {text}")
        self.status_message.emit(f"{cur}/{n} {text}")

    def _on_ok(self, path_str: str) -> None:
        out = Path(path_str)
        self._temp_output = out
        idx = self._busy_index
        item = self._items[idx] if 0 <= idx < len(self._items) else None
        tile = self._tile_for_index(idx) if idx >= 0 else None
        canvas = tile.canvas if tile is not None else None
        try:
            result = QImage(str(out))
            if result.isNull():
                raise ValueError(f"无法读取结果: {out}")
            if item is not None:
                if (
                    self._pending_undo_snapshot is not None
                    and not self._pending_undo_snapshot.isNull()
                ):
                    item.undo_stack.append(self._pending_undo_snapshot)
                    while len(item.undo_stack) > self._MAX_UNDO:
                        item.undo_stack.pop(0)
                    self._pending_undo_snapshot = None
                item.work_image = result.copy()
                item.pass_index += 1
                item.has_edits = True
                item.exported = False
                item.pending_mask = None  # consumed
                self._last_edit_index = idx
            if canvas is not None:
                canvas.load_qimage(
                    result,
                    logical_path=item.source_path if item else None,
                    keep_view=True,
                )
                canvas.set_multi_box_mode(True)
                canvas.set_tool(self._tool)
            self._batch_ok += 1
            self._refresh_page_status_labels()
        except (ValueError, OSError) as error:
            self._pending_undo_snapshot = None
            name = item.display_name() if item else "?"
            self._batch_fail.append(f"{name}: {error}")
        finally:
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_output = None

        self._batch_pos += 1
        self._cleanup_worker_only()
        self._start_next_in_batch()

    def _on_failed(self, message: str) -> None:
        self._pending_undo_snapshot = None
        idx = self._busy_index
        name = (
            self._items[idx].display_name()
            if 0 <= idx < len(self._items)
            else "?"
        )
        short = (message or "失败").replace("\n", " ")
        if len(short) > 80:
            short = short[:80] + "…"
        self._batch_fail.append(f"{name}: {short}")
        self._batch_pos += 1
        self._cleanup_worker_only()
        self._start_next_in_batch()

    def _cleanup_worker_only(self) -> None:
        self._thread = None
        self._worker = None
        self._cleanup_temps()

    def _finish_batch(self) -> None:
        n = len(self._batch_queue)
        ok, fail = self._batch_ok, self._batch_fail
        self._busy = False
        self._busy_index = -1
        self._batch_queue = []
        self._batch_pos = 0
        self.progress.hide()
        self._update_action_enabled()
        self._release_cold_bitmaps()
        self._refresh_page_status_labels()
        self._sync_tile_action_states()
        if not fail:
            self.status_message.emit(f"完成：成功 {ok}/{n} 张")
            if ok:
                QMessageBox.information(self, "完成", f"已处理 {ok} 张。")
        else:
            self.status_message.emit(f"精修结束：成功 {ok}，失败 {len(fail)}")
            detail = "\n".join(fail[:12])
            if len(fail) > 12:
                detail += f"\n…共 {len(fail)} 条失败"
            QMessageBox.warning(
                self,
                "部分失败",
                f"成功 {ok} 张，失败 {len(fail)} 张：\n\n{detail}",
            )

    def _reset_busy_ui(self) -> None:
        self._busy = False
        self._busy_index = -1
        self._batch_queue = []
        self._batch_pos = 0
        self._batch_ok = 0
        self._batch_fail = []
        self.progress.hide()
        self._sync_tile_action_states()
        self._update_action_enabled()

    def _cleanup_temps(self) -> None:
        path = self._temp_output
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_output = None

    # ------------------------------------------------------------------ export
    def _export(self, *, overwrite: bool = False) -> None:
        item = self._active_item()
        canvas = self.active_canvas()
        if item is None or canvas is None or not canvas.has_image():
            QMessageBox.information(self, "提示", "请先打开图片并完成精修。")
            return
        img = canvas.source_qimage()
        if img is None:
            return
        item.work_image = img.copy()
        mask = canvas.export_inpaint_mask_numpy()
        if mask is not None:
            item.pending_mask = mask

        if overwrite:
            out = item.source_path
            reply = QMessageBox.question(
                self,
                "覆盖原图",
                f"将覆盖原文件：\n{out}\n\n确定覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            fmt = "PNG" if out.suffix.lower() == ".png" else "JPG"
            if not img.save(str(out), fmt, 95 if fmt == "JPG" else -1):
                QMessageBox.warning(self, "导出失败", f"无法写入：\n{out}")
                return
            item.exported = True
            item.export_path = out
            self._refresh_page_status_labels()
            self._release_cold_bitmaps()
            self.status_message.emit(f"已覆盖 {out.name}")
            return

        default_name = f"{item.source_path.stem}_clean.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            str(item.source_path.parent / default_name),
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        out = Path(path)
        fmt = (
            "JPG"
            if out.suffix.lower() in {".jpg", ".jpeg"} or "JPEG" in selected_filter
            else "PNG"
        )
        if fmt == "JPG" and out.suffix.lower() not in {".jpg", ".jpeg"}:
            out = out.with_suffix(".jpg")
        if fmt == "PNG" and out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        if not img.save(str(out), fmt, 95 if fmt == "JPG" else -1):
            QMessageBox.warning(self, "导出失败", f"无法写入：\n{out}")
            return
        item.exported = True
        item.export_path = out
        self._refresh_page_status_labels()
        self._release_cold_bitmaps()
        self.status_message.emit(f"已导出 {out.name}")

    def _export_all_refined(self, *, overwrite: bool = False) -> None:
        """Write every refined (has_edits) item: save-as folder or overwrite originals."""
        if self._busy:
            QMessageBox.information(self, "请稍候", "正在处理中，请结束后再导出。")
            return
        self._flush_visible_tiles_to_items()
        targets = [it for it in self._items if it.has_edits]
        if not targets:
            QMessageBox.information(
                self,
                "提示",
                "没有可导出的已处理图片。",
            )
            return

        if overwrite:
            reply = QMessageBox.question(
                self,
                "覆盖全部已修原图",
                f"将覆盖 {len(targets)} 张原文件，且不可撤销。\n\n确定全部覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            out_dir: Path | None = None
        else:
            folder = QFileDialog.getExistingDirectory(
                self, "选择导出文件夹（全部已修）", str(self.workspace.root)
            )
            if not folder:
                return
            out_dir = Path(folder)

        ok = 0
        fail: list[str] = []
        for it in targets:
            try:
                img = it.ensure_work_image()
                if overwrite:
                    out = it.source_path
                    fmt = "PNG" if out.suffix.lower() == ".png" else "JPG"
                    if not img.save(str(out), fmt, 95 if fmt == "JPG" else -1):
                        fail.append(it.display_name())
                        continue
                else:
                    assert out_dir is not None
                    out = out_dir / f"{it.source_path.stem}_clean.png"
                    if out.exists():
                        n = 2
                        while True:
                            cand = out_dir / f"{it.source_path.stem}_clean_{n}.png"
                            if not cand.exists():
                                out = cand
                                break
                            n += 1
                    if not img.save(str(out), "PNG", -1):
                        fail.append(it.display_name())
                        continue
                it.exported = True
                it.export_path = out
                ok += 1
            except Exception:  # noqa: BLE001
                fail.append(it.display_name())
        self._refresh_page_status_labels()
        self._release_cold_bitmaps()
        if fail:
            self.status_message.emit(f"已写出 {ok} 张，失败 {len(fail)}")
            QMessageBox.warning(
                self,
                "部分失败",
                f"成功 {ok} 张，失败 {len(fail)}：\n" + "\n".join(fail[:10]),
            )
        elif overwrite:
            self.status_message.emit(f"已覆盖 {ok} 张原图")
            QMessageBox.information(self, "完成", f"已覆盖 {ok} 张原图。")
        else:
            assert out_dir is not None
            self.status_message.emit(f"已导出 {ok} 张")
            QMessageBox.information(self, "完成", f"已导出 {ok} 张到：\n{out_dir}")
