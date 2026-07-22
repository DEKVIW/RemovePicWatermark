"""YOLO train page: multi-tile annotation queue + train + auto-deploy weights.

User flow (optional enhancement — not required for daily batch):
  import images once → box watermarks on a grid → next page → train
  → best.pt auto-copied to workspace/models/yolo/watermark.pt
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...device_info import (
    header_device_caption,
    probe_cuda,
    resolve_ultralytics_device,
)
from ...services.yolo_dataset import BoxNorm, YoloDatasetService
from ...workspace import Workspace
from ..theme_style import SPACE_PAGE, SPACE_ROW
from ..widgets.icon_tool_button import IconToolButton
from ..widgets.image_canvas import CanvasTool, ImageCanvas
from ..widgets.page_chrome import ToolBar
from ..widgets.tool_icons import icon_empty_queue, icon_next, icon_prev
from ..workers import YoloTrainRequest, YoloTrainWorker, start_yolo_train_worker

try:
    from qfluentwidgets import BodyLabel, ComboBox, PrimaryPushButton, PushButton
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QComboBox as ComboBox
    from PySide6.QtWidgets import QLabel as BodyLabel
    from PySide6.QtWidgets import QPushButton as PrimaryPushButton
    from PySide6.QtWidgets import QPushButton as PushButton


# layout presets: (rows, cols) → page size
_LAYOUTS = {
    "1×1 专注": (1, 1),
    "2×2 默认": (2, 2),
    "2×3": (2, 3),
    "2×4": (2, 4),
    "3×3": (3, 3),
}


class _Tile(QFrame):
    """One annotation cell: canvas + compact status strip."""

    boxes_changed = Signal(int)  # global index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.global_index = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.canvas = ImageCanvas()
        self.canvas.set_multi_box_mode(True)
        self.canvas.set_tool(CanvasTool.RECT)
        self.canvas.set_brush_radius(18)
        self.canvas.setMinimumHeight(120)
        # Empty cells are not file pickers — use toolbar import + 下一批
        self.canvas._placeholder = "本格无图"
        layout.addWidget(self.canvas, 1)
        self.status = BodyLabel("空")
        self.status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.status)
        self.canvas.multi_boxes_changed.connect(self._on_boxes)
        self.canvas.mask_changed.connect(self._on_paint_mask)
        self.canvas.history_changed.connect(self._on_history)
        self.canvas.tool_changed.connect(self._sync_tool_from_canvas)
        # Swallow empty click so users don't expect "click to open"
        self.canvas.empty_clicked.connect(lambda: None)

    def _sync_tool_from_canvas(self, tool: CanvasTool) -> None:
        """Keyboard tool switch on a tile → update TrainPage + main toolbar."""
        w = self.parentWidget() if hasattr(self, "parentWidget") else None
        while w is not None:
            if hasattr(w, "_set_tool"):
                w._set_tool(CanvasTool(tool))
                break
            w = w.parentWidget()

    def _on_history(self) -> None:
        w = self.parentWidget() if hasattr(self, "parentWidget") else None
        while w is not None:
            if hasattr(w, "chrome_state_changed"):
                w.chrome_state_changed.emit()
                break
            w = w.parentWidget()

    def annotation_boxes(self) -> list[BoxNorm]:
        """Current multi-boxes as BoxNorm (keeps OBB angles)."""
        out: list[BoxNorm] = []
        for cx, cy, w, h, ang in self.canvas.multi_boxes_oriented_norm():
            out.append(BoxNorm.from_oriented_norm(cx, cy, w, h, ang))
        return out

    def _refresh_status(self) -> None:
        if self.global_index < 0:
            self.status.setText("空")
            self.status.setStyleSheet("color: #94a3b8; font-size: 11px;")
            return
        boxes = self.annotation_boxes()
        n = len(boxes)
        if n <= 0:
            self.status.setText("未标注")
            self.status.setStyleSheet("color: #fbbf24; font-size: 11px;")
            return
        n_obb = sum(1 for b in boxes if b.is_obb)
        if n_obb:
            self.status.setText(f"已标 {n} 框（斜 {n_obb}）")
        else:
            self.status.setText(f"已标 {n} 框")
        self.status.setStyleSheet("color: #34d399; font-size: 11px;")

    def _on_boxes(self) -> None:
        self._refresh_status()
        if self.global_index >= 0:
            self.boxes_changed.emit(self.global_index)

    def _on_paint_mask(self) -> None:
        """Paint stroke finished → oriented min-area rects (match stroke tilt).

        Interaction aligns with YOLO labels: user sees the same boxes that will
        be trained (OBB if tilted, axis-aligned if nearly horizontal).
        """
        if self.global_index < 0:
            return
        if self.canvas.tool != CanvasTool.PAINT:
            return
        if not self.canvas.has_paint_mask():
            return
        if getattr(self, "_committing_paint", False):
            return
        self._committing_paint = True
        try:
            added = self.canvas.commit_paint_mask_as_multi_boxes(clear_paint=True)
            if added:
                self._on_boxes()
            else:
                self.status.setText("涂抹未成框，请加大笔刷再涂")
                self.status.setStyleSheet("color: #fbbf24; font-size: 11px;")
        finally:
            self._committing_paint = False

    def clear_tile(self) -> None:
        self.global_index = -1
        self.canvas.clear_image()
        self.status.setText("空")
        self.status.setStyleSheet("color: #94a3b8; font-size: 11px;")

    def load_item(self, index: int, path: Path, boxes: list[BoxNorm]) -> None:
        self.global_index = index
        try:
            self.canvas.load_path(path, keep_view=False)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"无法打开: {exc}")
            return
        oriented = [b.to_oriented_norm() for b in boxes]
        self.canvas.set_multi_boxes_oriented_norm(oriented)
        self._refresh_status()

    def set_tool(self, tool: CanvasTool) -> None:
        self.canvas.set_tool(tool)

    def set_brush_radius(self, radius: int) -> None:
        self.canvas.set_brush_radius(radius)


class TrainPage(QWidget):
    """Annotate watermark boxes on a paged grid, then train YOLO and deploy."""

    status_message = Signal(str)
    chrome_state_changed = Signal()

    def __init__(self, workspace: Workspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.dataset = YoloDatasetService(workspace).ensure()
        self._page = 0
        self._rows, self._cols = 2, 2
        self._tiles: list[_Tile] = []
        self._tool = CanvasTool.RECT
        self._brush_radius = 18
        self._train_thread = None
        self._train_worker: YoloTrainWorker | None = None
        self._device_pref = "auto"
        self._probe = None  # lazy CUDA probe
        self.setAcceptDrops(True)
        self._build_ui()
        self.dataset.load_manifest()
        self._refresh_page()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_PAGE, SPACE_PAGE, SPACE_PAGE, SPACE_PAGE)
        root.setSpacing(SPACE_ROW)

        # Page-local: layout / paging / epochs / clear dataset.
        # Import uses main toolbar open / open_folder.
        bar = ToolBar(slim=True)
        self.layout_combo = ComboBox()
        self.layout_combo.addItems(list(_LAYOUTS.keys()))
        self.layout_combo.setCurrentText("2×2 默认")
        self.layout_combo.setToolTip("布局")
        self.layout_combo.setMinimumWidth(100)
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
        self.stats_label = BodyLabel("已标注 0 / 0")
        self.stats_label.setProperty("role", "caption")
        bar.add(self.stats_label)

        bar.add_spacing(12)
        bar.add(BodyLabel("轮数"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(5, 300)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSingleStep(5)
        self.epochs_spin.setMinimumWidth(96)
        self.epochs_spin.setFixedHeight(30)
        self.epochs_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.epochs_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.epochs_spin.setToolTip("训练轮数")
        bar.add(self.epochs_spin)

        bar.add_spacing(8)
        self.clear_btn = IconToolButton(
            icon_empty_queue(),
            "清空训练数据",
            size=32,
        )
        self.clear_btn.clicked.connect(self._clear_dataset)
        bar.add(self.clear_btn)
        bar.add_stretch(1)
        root.addWidget(bar)

        # Hidden buttons for start/stop state (main toolbar ▶ / ■)
        self.train_btn = PrimaryPushButton("开始训练")
        self.train_btn.clicked.connect(self._start_train)
        self.train_btn.hide()
        self.stop_train_btn = PushButton("停止训练")
        self.stop_train_btn.clicked.connect(self._stop_train)
        self.stop_train_btn.hide()

        # Canvas tools live on the main toolbar; grid fills the page
        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        root.addWidget(self.grid_host, 1)
        self._rebuild_grid()

    def set_device_preference(self, pref: str) -> None:
        pref = str(pref).lower()
        if pref in {"cuda", "gpu"}:
            self._device_pref = "gpu"
        elif pref == "cpu":
            self._device_pref = "cpu"
        else:
            self._device_pref = "auto"
        # Prefer lazy probe on train start / showEvent
        self.chrome_state_changed.emit()

    def current_device_preference(self) -> str:
        return self._device_pref

    def device_status_extra(self) -> str:
        # Lightweight: do not force YOLO/torch import just for the chip label
        from ...detectors.yolo_watermark import _YOLO_CLS, yolo_import_error

        if _YOLO_CLS is not None:
            yolo = "检测就绪"
        elif yolo_import_error():
            yolo = "检测未就绪"
        else:
            yolo = "检测待加载"
        wt = self.workspace.yolo_dir / "watermark.pt"
        wtxt = "已有模型" if wt.is_file() else "尚无模型"
        return f"{yolo} · {wtxt}"

    def _resolve_yolo_base_weights(self, *, task: str = "detect") -> str:
        """Prefer bundled weights so train works offline.

        OBB training needs an OBB pretrained checkpoint (e.g. yolov8n-obb.pt).
        """
        from ...paths import app_root

        if task == "obb":
            names = ("yolov8n-obb.pt", "yolo11n-obb.pt")
        else:
            names = ("yolov8n.pt", "yolo11n.pt")
        roots = (app_root(), self.workspace.yolo_dir, Path.cwd())
        for name in names:
            for root in roots:
                path = Path(root) / name
                try:
                    if path.is_file() and path.stat().st_size > 500_000:
                        return str(path.resolve())
                except OSError:
                    continue
        # Fallback name: ultralytics may download if online
        return names[0]

    def _set_tool(self, tool: CanvasTool) -> None:
        tool = CanvasTool(tool)
        changed = self._tool != tool
        self._tool = tool
        for tile in self._tiles:
            if tile.canvas.tool != self._tool:
                tile.set_tool(self._tool)
        if changed:
            self.chrome_state_changed.emit()

    def _on_brush(self, value: int) -> None:
        self._brush_radius = int(value)
        for tile in self._tiles:
            tile.set_brush_radius(self._brush_radius)

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
        return self._tool

    def canvas_brush_radius(self) -> int:
        return int(self._brush_radius)

    def set_canvas_tool(self, tool: CanvasTool) -> None:
        self._set_tool(tool)

    def set_canvas_brush(self, radius: int) -> None:
        self._on_brush(radius)

    def _active_tile(self) -> _Tile | None:
        for tile in self._tiles:
            if tile.global_index >= 0 and tile.canvas.has_image():
                # Prefer focused canvas if any
                if tile.canvas.hasFocus():
                    return tile
        for tile in self._tiles:
            if tile.global_index >= 0 and tile.canvas.has_image():
                return tile
        return None

    def active_canvas(self):
        tile = self._active_tile()
        return tile.canvas if tile is not None else None

    def _clear_active_annotation(self) -> None:
        canvas = self.active_canvas()
        if canvas is not None:
            canvas.clear_roi()
            self.chrome_state_changed.emit()

    def _undo_active_annotation(self) -> None:
        canvas = self.active_canvas()
        if canvas is not None and canvas.undo_annotation():
            self.chrome_state_changed.emit()

    def app_action_caps(self) -> set[str]:
        caps = {
            "open",
            "open_folder",
            "undo",
            "redo",
            "clear",
            "delete_box",
            "zoom_in",
            "zoom_out",
            "zoom_fit",
            "zoom_1x",
            "run",
        }
        if self._train_thread is not None and self._train_thread.isRunning():
            caps.add("stop")
            caps.discard("run")
        return caps

    def toolbar_action_labels(self) -> dict[str, str]:
        return {
            "open": "导入图片",
            "open_folder": "导入文件夹",
            "run": "开始训练",
        }

    def can_undo(self) -> bool:
        canvas = self.active_canvas()
        return bool(canvas and canvas.can_undo_annotation())

    def can_redo(self) -> bool:
        canvas = self.active_canvas()
        return bool(canvas and canvas.can_redo_annotation())

    def is_busy(self) -> bool:
        return bool(self._train_thread is not None and self._train_thread.isRunning())

    def handle_app_action(self, action: str) -> bool:
        if action == "open":
            self._import_images()
            return True
        if action == "open_folder":
            self._import_folder()
            return True
        if action == "run":
            self._start_train()
            return True
        if action == "stop":
            self._stop_train()
            return True
        canvas = self.active_canvas()
        if canvas is None:
            return False
        if action == "undo":
            ok = bool(canvas.undo_annotation())
            if ok:
                self.chrome_state_changed.emit()
            return ok  # False → main window may try other train tiles
        if action == "redo":
            ok = bool(canvas.redo_annotation())
            if ok:
                self.chrome_state_changed.emit()
            return ok
        if action == "clear":
            canvas.clear_roi()
            return True
        if action == "delete_box":
            canvas.pop_last_multi_box()
            return True
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

    def _stop_train(self) -> None:
        if self._train_worker is not None:
            self._train_worker.request_cancel()
            self.status_message.emit("正在停止训练…")
            self.chrome_state_changed.emit()

    def _rebuild_grid(self) -> None:
        # clear
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._tiles = []
        for r in range(self._rows):
            for c in range(self._cols):
                tile = _Tile()
                tile.boxes_changed.connect(self._on_tile_boxes)
                tile.set_tool(self._tool)
                tile.set_brush_radius(self._brush_radius)
                self.grid_layout.addWidget(tile, r, c)
                self._tiles.append(tile)
        self._refresh_page()

    def _page_size(self) -> int:
        return max(1, self._rows * self._cols)

    def _on_layout_changed(self, text: str) -> None:
        pair = _LAYOUTS.get(text) or (2, 2)
        self._rows, self._cols = pair
        # Restart at batch 1 when grid density changes (same as refine page).
        self._page = 0
        self._rebuild_grid()

    def _refresh_page(self) -> None:
        page_size = self._page_size()
        pages = self.dataset.page_count(page_size)
        self._page = max(0, min(self._page, pages - 1))
        slice_items = self.dataset.page_slice(self._page, page_size)
        for tile in self._tiles:
            tile.clear_tile()
        for slot, (gidx, item) in enumerate(slice_items):
            if slot >= len(self._tiles):
                break
            self._tiles[slot].load_item(gidx, item.image_path, item.boxes)
        self.page_label.setText(f"第 {self._page + 1} / {pages} 批")
        self.stats_label.setText(
            f"已标注 {self.dataset.labeled_count()} / {self.dataset.count()}"
        )
        self.prev_btn.setEnabled(self._page > 0)
        self.next_btn.setEnabled(self._page < pages - 1)

    def _on_tile_boxes(self, global_index: int) -> None:
        # Persist oriented boxes (OBB angles preserved when tilted).
        for tile in self._tiles:
            if tile.global_index == global_index:
                self.dataset.set_boxes(global_index, tile.annotation_boxes())
                break
        self.stats_label.setText(
            f"已标注 {self.dataset.labeled_count()} / {self.dataset.count()}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._import_path_list(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _import_path_list(self, paths: list[Path]) -> None:
        files: list[Path] = []
        folders: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                folders.append(p)
            elif p.is_file():
                files.append(p)
        n = 0
        if files:
            n += self.dataset.import_paths(files, copy=True)
        for folder in folders:
            n += self.dataset.import_folder(folder, recursive=False)
        if n <= 0:
            self.status_message.emit("未导入新图片")
            return
        self._refresh_page()
        self.status_message.emit(f"已导入 {n} 张（队列共 {self.dataset.count()} 张）")

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择带水印的图片",
            str(self.workspace.root),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if not files:
            return
        self._import_path_list([Path(f) for f in files])

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(self.workspace.root))
        if not folder:
            return
        self._import_path_list([Path(folder)])

    def _clear_dataset(self) -> None:
        total = self.dataset.count()
        labeled = self.dataset.labeled_count()
        if total <= 0:
            self.status_message.emit("数据集已是空的")
            return
        reply = QMessageBox.question(
            self,
            "清空训练数据",
            f"将清空队列中的 {total} 张图片及标注（已标 {labeled} 张）。\n"
            "已训练的检测模型不受影响。\n\n确定清空？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self._train_thread is not None and self._train_thread.isRunning():
            QMessageBox.information(self, "训练中", "训练进行中，请结束后再清空。")
            return
        n = self.dataset.clear_dataset(delete_files=True)
        self._page = 0
        self._refresh_page()
        self.status_message.emit(f"已清空训练数据集（{n} 张）")

    def _prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._refresh_page()

    def _next_page(self) -> None:
        if self._page + 1 < self.dataset.page_count(self._page_size()):
            self._page += 1
            self._refresh_page()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            from ...device_info import clear_probe_cache

            clear_probe_cache()
        except Exception:  # noqa: BLE001
            pass
        self.dataset.load_manifest()
        self._refresh_page()
        self.chrome_state_changed.emit()

    def _resolve_device(self) -> str:
        """Return ultralytics device string: 'cpu' or '0' (first CUDA GPU)."""
        try:
            self._probe = probe_cuda()
        except Exception:  # noqa: BLE001
            pass
        return resolve_ultralytics_device(self._device_pref, self._probe)

    def _start_train(self) -> None:
        labeled = self.dataset.labeled_count()
        # Soft floor: 1 image is allowed for a dense multi-instance stamp page
        # (one photo can contain dozens of boxes). Recommend ≥5 for better recall.
        if labeled < 1:
            QMessageBox.warning(self, "标注不足", "请先至少标注 1 张图片。")
            return
        if labeled < 5:
            reply = QMessageBox.question(
                self,
                "标注偏少",
                f"当前仅 {labeled} 张已标注，建议不少于 5 张。\n仍要继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
        if self._train_thread is not None and self._train_thread.isRunning():
            QMessageBox.information(self, "训练中", "已有训练任务在进行，请稍候。")
            return
        try:
            from ...detectors.yolo_watermark import import_yolo_class

            import_yolo_class()
        except RuntimeError as exc:
            QMessageBox.warning(self, "无法训练", "检测组件不可用。")
            return

        total = self.dataset.count()
        unlabeled = total - labeled
        use_obb = self.dataset.dataset_uses_obb()
        task = "obb" if use_obb else "detect"
        task_label = "斜框" if use_obb else "矩形框"
        extra = ""
        if unlabeled > 0:
            extra = f"\n队列共 {total} 张，未标注的 {unlabeled} 张不参与本次训练。"
        device_human = header_device_caption(self._device_pref, self._probe)
        base_model = self._resolve_yolo_base_weights(task=task)
        reply = QMessageBox.question(
            self,
            "开始训练",
            f"将用已标注的 {labeled} 张图片训练（{task_label}）。{extra}\n"
            f"轮数 {self.epochs_spin.value()} · {device_human}\n"
            "完成后将更新检测模型。\n\n继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        yaml_path = self.dataset.write_data_yaml()
        req = YoloTrainRequest(
            data_yaml=yaml_path,
            project_dir=self.dataset.runs_dir,
            name="gui_train",
            model=base_model,
            epochs=int(self.epochs_spin.value()),
            imgsz=640,
            batch=4 if self._resolve_device() != "cpu" else 2,
            device=self._resolve_device(),
            task=task,
        )
        worker = YoloTrainWorker(self.workspace, req)
        self._train_worker = worker
        self.train_btn.setEnabled(False)
        worker.log_line.connect(self._on_train_log)
        worker.stage.connect(self._on_train_stage)
        worker.progress.connect(self._on_train_progress)
        worker.finished_ok.connect(self._on_train_ok)
        worker.failed.connect(self._on_train_fail)
        self._train_thread = start_yolo_train_worker(self, worker)
        epochs = int(self.epochs_spin.value())
        self.status_message.emit(f"训练中 0/{epochs}（0%）")
        self.chrome_state_changed.emit()

    def _on_train_stage(self, stage: str) -> None:
        text = (stage or "").strip()
        if text:
            self.status_message.emit(text[:120])

    def _on_train_progress(self, epoch: int, total: int, percent: int) -> None:
        """Status-bar friendly progress: 训练中 3/100（3%）."""
        total = max(1, int(total))
        epoch = max(0, min(int(epoch), total))
        percent = max(0, min(100, int(percent)))
        self.status_message.emit(f"训练中 {epoch}/{total}（{percent}%）")

    def _on_train_log(self, line: str) -> None:
        # Prefer structured stage/progress for status bar; only surface short
        # non-progress logs (avoid drowning percent with ultralytics noise).
        short = line.strip().splitlines()[-1] if line.strip() else line
        if not short:
            return
        low = short.lower()
        # Skip verbose ultralytics table rows / tqdm that look like progress spam
        if any(
            tok in low
            for tok in (
                "epoch",
                "gpu_mem",
                "box_loss",
                "cls_loss",
                "dfl_loss",
                "instances",
                "size",
                "%|",
                "it/s",
            )
        ):
            return
        if len(short) > 100:
            short = short[:100] + "…"
        # Don't overwrite a live percent line with random logs unless useful
        if short.startswith("轮数") or short.startswith("已部署") or short.startswith("效果"):
            self.status_message.emit(short)

    def _on_train_ok(self, dest: str) -> None:
        self.train_btn.setEnabled(True)
        parts = str(dest).split("\n", 1)
        path_only = parts[0].strip()
        note = parts[1].strip() if len(parts) > 1 else ""
        self.status_message.emit("训练完成 100%")
        self.chrome_state_changed.emit()
        body = "检测模型已更新。"
        if note:
            body += f"\n\n{note}"
        QMessageBox.information(self, "训练完成", body)

    def _on_train_fail(self, msg: str) -> None:
        self.train_btn.setEnabled(True)
        self.status_message.emit("训练结束")
        self.chrome_state_changed.emit()
        if "取消" in msg or "已取消" in msg:
            QMessageBox.information(self, "已停止", "训练已停止。")
        else:
            QMessageBox.critical(self, "训练失败", msg[:1500])
