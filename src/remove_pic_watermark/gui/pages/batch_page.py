"""Batch import + remove watermarks.

Layout (top → bottom):
  ┌─────────────┬──────────────┐
  │ 待处理图片   │ 本次样式      │
  └─────────────┴──────────────┘
  Options (single row): 定位 | 查找 | 模型状态
  Process log
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ...device_info import (
    device_tooltip,
    fallback_dialog_text,
    probe_cuda,
    resolve_runtime_device,
)
from ...services.profile_service import ProfileService
from ...workspace import Workspace
from ..theme_style import SPACE_CARD_GAP, SPACE_PAGE, SPACE_ROW
from ..widgets.icon_tool_button import IconToolButton
from ..widgets.log_panel import LogPanel
from ..widgets.page_chrome import CountBadge, SectionPanel, form_row
from ..widgets.tool_icons import icon_add, icon_empty_queue, icon_remove
from ..workers import JobWorker, RunJobRequest, start_job_worker

try:
    from qfluentwidgets import (
        BodyLabel,
        CheckBox,
        ComboBox,
        PrimaryPushButton,
        PushButton,
    )
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QCheckBox as CheckBox
    from PySide6.QtWidgets import QComboBox as ComboBox
    from PySide6.QtWidgets import QLabel as BodyLabel
    from PySide6.QtWidgets import QPushButton as PrimaryPushButton
    from PySide6.QtWidgets import QPushButton as PushButton


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class DropListWidget(QListWidget):
    """QListWidget that accepts image file / folder drops."""

    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAlternatingRowColors(False)
        self.setToolTip("可拖入图片或文件夹")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths: list[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class BatchPage(QWidget):
    job_finished = Signal(object)
    status_message = Signal(str)
    chrome_state_changed = Signal()

    _DEVICE_PREFS = ("auto", "cpu", "gpu")

    def app_action_caps(self) -> set[str]:
        caps = {"open", "open_folder", "run"}
        if self._busy:
            caps.add("stop")
        return caps

    def toolbar_action_labels(self) -> dict[str, str]:
        return {
            "open": "添加图片",
            "open_folder": "添加文件夹",
            "run": "开始处理",
        }

    def can_undo(self) -> bool:
        return False

    def can_redo(self) -> bool:
        return False

    def is_busy(self) -> bool:
        return bool(self._busy)

    def handle_app_action(self, action: str) -> bool:
        if action == "open":
            self._add_files()
            return True
        if action == "open_folder":
            self._add_dir()
            return True
        if action == "run":
            self._run()
            return True
        if action == "stop":
            self._stop()
            return True
        return False

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.profile_service = ProfileService(workspace)
        self._input_paths: list[Path] = []
        self._thread = None
        self._worker = None
        self._busy = False
        self._last_job_dir: Path | None = None
        # Lazy CUDA probe — importing torch at startup is expensive
        self._probe = None
        self._suppress_device_signal = False
        self._build_ui()
        self.reload_profiles()
        self._sync_detect_for_strategy()
        self._refresh_counts()
        # Model status + device tip: refresh when page is shown (not at app start)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Always re-read latest detection weights when entering the page
        if self._probe is None:
            try:
                self._probe = probe_cuda()
            except Exception:  # noqa: BLE001
                pass
        self._refresh_model_status()
        self._refresh_device_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_PAGE, SPACE_PAGE, SPACE_PAGE, SPACE_PAGE)
        root.setSpacing(SPACE_ROW)

        # Top: image queue + profiles (main workspace)
        mid_host = QWidget()
        mid = QHBoxLayout(mid_host)
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(SPACE_CARD_GAP)
        mid.addWidget(self._build_images_panel(), 3)
        mid.addWidget(self._build_profiles_panel(), 2)
        root.addWidget(mid_host, 1)

        # Middle: options + start (primary actions sit above the log)
        root.addWidget(self._build_options_bar())

        # Bottom: process log (conventional output strip at page bottom)
        self._log_frame = QFrame()
        self._log_frame.setObjectName("collapsibleLog")
        self._log_frame.setMinimumHeight(120)
        log_outer = QVBoxLayout(self._log_frame)
        log_outer.setContentsMargins(10, 6, 10, 8)
        log_outer.setSpacing(4)

        log_head = QHBoxLayout()
        log_title = BodyLabel("处理日志")
        log_title.setProperty("role", "section")
        log_head.addWidget(log_title)
        log_head.addStretch(1)
        self._log_toggle = PushButton("▾")
        self._log_toggle.setObjectName("inspectorToggle")
        self._log_toggle.setFixedSize(28, 28)
        self._log_toggle.setToolTip("折叠 / 展开日志")
        self._log_toggle.clicked.connect(self._toggle_log)
        log_head.addWidget(self._log_toggle)
        log_outer.addLayout(log_head)

        self.log = LogPanel()
        self._log_body = QWidget()
        log_body_l = QVBoxLayout(self._log_body)
        log_body_l.setContentsMargins(0, 0, 0, 0)
        log_body_l.addWidget(self.log)
        log_outer.addWidget(self._log_body, 1)
        self._log_expanded = True
        self._log_frame.setMaximumHeight(220)
        root.addWidget(self._log_frame)

        self.log.append("就绪")

    def _toggle_log(self) -> None:
        self._log_expanded = not self._log_expanded
        self._log_body.setVisible(self._log_expanded)
        self._log_toggle.setText("▾" if self._log_expanded else "▸")
        self._log_toggle.setToolTip("折叠日志" if self._log_expanded else "展开日志")
        if self._log_expanded:
            self._log_frame.setMinimumHeight(120)
            self._log_frame.setMaximumHeight(220)
        else:
            # Header strip only
            self._log_frame.setMinimumHeight(40)
            self._log_frame.setMaximumHeight(44)

    def _build_images_panel(self) -> SectionPanel:
        panel = SectionPanel("待处理图片")
        self.image_count = CountBadge("0")
        panel.add_header_widget(self.image_count)

        self.image_list = DropListWidget()
        self.image_list.files_dropped.connect(self._on_paths_dropped)
        self.image_list.itemDoubleClicked.connect(self._reveal_image)
        panel.body.addWidget(self.image_list, 1)

        # Add files/folders via main toolbar open / open_folder (drag-drop still works)
        btns = QHBoxLayout()
        btns.setSpacing(6)
        self.remove_sel_btn = IconToolButton(icon_remove(), "移除选中", size=34)
        self.clear_images_btn = IconToolButton(
            icon_empty_queue(),
            "清空待处理队列",
            size=34,
        )
        self.remove_sel_btn.clicked.connect(self._remove_selected_images)
        self.clear_images_btn.clicked.connect(self._clear_images)
        for b in (self.remove_sel_btn, self.clear_images_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        panel.body.addLayout(btns)
        return panel

    def _build_profiles_panel(self) -> SectionPanel:
        panel = SectionPanel("本次使用的样式")
        self.profile_count = CountBadge("0")
        panel.add_header_widget(self.profile_count)

        self.profile_list = QListWidget()
        self.profile_list.setToolTip("勾选的样式将用于本次处理")
        panel.body.addWidget(self.profile_list, 1)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        from ..widgets.tool_icons import icon_refresh

        self.select_all_btn = IconToolButton(icon_add(), "全选样式", size=34)
        self.select_none_btn = IconToolButton(icon_remove(), "全不选", size=34)
        self.reload_profiles_btn = IconToolButton(icon_refresh(), "刷新样式列表", size=34)
        self.select_all_btn.clicked.connect(lambda: self._set_all_profiles(True))
        self.select_none_btn.clicked.connect(lambda: self._set_all_profiles(False))
        self.reload_profiles_btn.clicked.connect(self.reload_profiles)
        for b in (self.select_all_btn, self.select_none_btn, self.reload_profiles_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        panel.body.addLayout(btns)
        return panel

    def _build_options_bar(self) -> QFrame:
        """Single-row: 定位 | 查找 | 模型状态 (no double-height option cards)."""
        bar = QFrame()
        bar.setObjectName("optionsBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(14)

        # Backend / device live in main-window「设置」menu
        self.backend_combo = ComboBox()
        self._backend_values = ["iopaint", "opencv"]
        self.backend_combo.addItems(["高质量", "快速"])
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.hide()

        self.device_combo = ComboBox()
        for label in ("自动", "CPU", "GPU"):
            self.device_combo.addItem(label)
        self.device_combo.setCurrentIndex(0)
        self.device_combo.hide()

        # 定位: pin | follow | search
        self.strategy_combo = ComboBox()
        self._strategy_values = ["pin", "follow", "search"]
        self.strategy_combo.addItems(["固定位置", "附近匹配", "全图匹配"])
        self.strategy_combo.setToolTip(
            "固定位置：认定该处一定有，按录入位置与遮罩直接覆盖\n"
            "附近匹配：认定大致在该区域，用样式在附近比对，像了再盖\n"
            "全图匹配：同款可能出现在任意位置，整图查找"
        )
        self.strategy_combo.setCurrentIndex(1)  # 附近匹配
        self.strategy_combo.setMinimumWidth(108)
        self.strategy_combo.setFixedHeight(30)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        layout.addLayout(form_row("定位", self.strategy_combo))

        # 查找: styles | both | ai → 水印样式 / 样式+模型 / 水印模型
        self.detect_combo = ComboBox()
        self._detect_values = ["styles", "both", "ai"]
        self.detect_combo.addItems(["水印样式", "样式+模型", "水印模型"])
        self.detect_combo.setToolTip(
            "水印样式：只用已保存样式匹配\n"
            "样式+模型：先样式；本张样式未检出时才用检测模型补漏\n"
            "水印模型：主要用训练好的检测模型"
        )
        self.detect_combo.setCurrentIndex(0)
        self.detect_combo.setMinimumWidth(108)
        self.detect_combo.setFixedHeight(30)
        self.detect_combo.currentIndexChanged.connect(self._on_detect_mode_changed)
        layout.addLayout(form_row("查找", self.detect_combo))

        # Read-only model status (no enable checkbox)
        self.model_status = BodyLabel("")
        self.model_status.setProperty("role", "caption")
        self.model_status.setMinimumWidth(120)
        layout.addWidget(self.model_status, 0)

        # Keep alias for any old references during transition
        self.yolo_status = self.model_status
        self.yolo_enable = CheckBox("启用")
        self.yolo_enable.setChecked(True)
        self.yolo_enable.hide()

        layout.addStretch(1)

        self.open_job_btn = PushButton("打开上次结果")
        self.open_job_btn.setToolTip("打开最近一次批量处理的文件夹")
        self.open_job_btn.clicked.connect(self._open_last_job)
        self.open_job_btn.hide()
        self.stop_btn = PushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.hide()
        self.run_btn = PrimaryPushButton("开始处理")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.hide()
        return bar

    # ----- profiles -----

    def reload_profiles(self) -> None:
        selected = set(self.selected_profile_ids())
        self.profile_list.clear()
        # Do not auto-seed builtins — packaged / fresh installs stay empty until user creates styles
        for profile in self.profile_service.list_profiles():
            item = QListWidgetItem(f"{profile.name}")
            item.setToolTip(profile.name)
            item.setData(Qt.UserRole, profile.id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            if selected:
                item.setCheckState(Qt.Checked if profile.id in selected else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked if profile.enabled else Qt.Unchecked)
            self.profile_list.addItem(item)
        self._refresh_counts()
        checked = len(self.selected_profile_ids())
        total = self.profile_list.count()
        self.log.append(f"样式共 {total} 个，本次勾选 {checked} 个")

    def selected_profile_ids(self) -> list[str]:
        ids: list[str] = []
        for index in range(self.profile_list.count()):
            item = self.profile_list.item(index)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                pid = item.data(Qt.UserRole)
                if pid:
                    ids.append(str(pid))
        return ids

    def _set_all_profiles(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for index in range(self.profile_list.count()):
            item = self.profile_list.item(index)
            if item is not None:
                item.setCheckState(state)
        self._refresh_counts()
        self.log.append("已全选样式" if checked else "已取消全部勾选")

    # ----- backend / device prefs -----

    def set_backend(self, backend: str) -> None:
        if backend == "none":
            backend = "iopaint"
        if backend in self._backend_values:
            self.backend_combo.setCurrentIndex(self._backend_values.index(backend))
        # Light enable/tooltip only — no CUDA probe (startup path)
        lama = backend in {"iopaint", "lama"} or (
            backend in self._backend_values
            and self._backend_values[self.backend_combo.currentIndex()] == "iopaint"
        )
        self.device_combo.setEnabled(lama if backend != "opencv" else False)
        if not lama or backend == "opencv":
            self.device_combo.setEnabled(False)
            self.device_combo.setToolTip("快速预览使用 CPU")
        else:
            self.device_combo.setEnabled(True)
        self.chrome_state_changed.emit()

    def current_backend(self) -> str:
        index = self.backend_combo.currentIndex()
        if 0 <= index < len(self._backend_values):
            return self._backend_values[index]
        return "iopaint"

    def set_match_strategy(self, strategy: str) -> None:
        key = (strategy or "follow").strip().lower()
        if key == "auto":
            key = "follow"
        if key in self._strategy_values:
            self.strategy_combo.blockSignals(True)
            self.strategy_combo.setCurrentIndex(self._strategy_values.index(key))
            self.strategy_combo.blockSignals(False)
            self._sync_detect_for_strategy()
            self._refresh_model_status()

    def current_match_strategy(self) -> str:
        idx = self.strategy_combo.currentIndex()
        if 0 <= idx < len(self._strategy_values):
            return self._strategy_values[idx]
        return "follow"

    def set_detect_mode(self, mode: str) -> None:
        key = (mode or "styles").strip().lower()
        if key in {"styles+ai", "styles_ai", "all", "union"}:
            key = "both"
        if key not in self._detect_values:
            key = "styles"
        # Pin forces styles
        if self.current_match_strategy() == "pin":
            key = "styles"
        self.detect_combo.blockSignals(True)
        self.detect_combo.setCurrentIndex(self._detect_values.index(key))
        self.detect_combo.blockSignals(False)
        # Status line is cheap now (no torch); still skip on pure styles to avoid work
        if key in {"both", "ai"}:
            self._refresh_model_status()
        elif hasattr(self, "model_status"):
            self.model_status.setText("")
            self.model_status.setVisible(False)

    def current_detect_mode(self) -> str:
        if self.current_match_strategy() == "pin":
            return "styles"
        idx = self.detect_combo.currentIndex()
        if 0 <= idx < len(self._detect_values):
            return self._detect_values[idx]
        return "styles"

    def yolo_enabled(self) -> bool:
        """Derived from lookup mode — no separate checkbox."""
        if self.current_match_strategy() == "pin":
            return False
        return self.current_detect_mode() in {"both", "ai"}

    def set_yolo_enabled(self, enabled: bool) -> None:
        """Compat: map old prefs onto detect mode when possible."""
        if enabled and self.current_detect_mode() == "styles":
            # keep styles unless user had both/ai already
            pass
        elif not enabled and self.current_detect_mode() in {"both", "ai"}:
            self.set_detect_mode("styles")
        self._refresh_model_status()

    def _on_strategy_changed(self, _index: int = 0) -> None:
        self._sync_detect_for_strategy()
        self._refresh_model_status()

    def _on_detect_mode_changed(self, _index: int = 0) -> None:
        self._refresh_model_status()

    def _sync_detect_for_strategy(self) -> None:
        """固定位置 → lock 查找 to 水印样式."""
        pin = self.current_match_strategy() == "pin"
        self.detect_combo.setEnabled(not pin)
        if pin:
            self.detect_combo.blockSignals(True)
            self.detect_combo.setCurrentIndex(self._detect_values.index("styles"))
            self.detect_combo.blockSignals(False)
            self.detect_combo.setToolTip("固定位置仅使用水印样式")
        else:
            self.detect_combo.setToolTip(
                "水印样式：只用样式\n"
                "样式+模型：先样式，未检出时检测模型补漏\n"
                "水印模型：主要用检测模型"
            )

    def _refresh_model_status(self) -> None:
        """Read-only model line; always re-resolve latest weights file."""
        if not hasattr(self, "model_status"):
            return
        from ...detectors.yolo_watermark import probe_yolo, resolve_yolo_weights

        mode = self.current_detect_mode()
        want = self.yolo_enabled()
        if not want:
            self.model_status.setText("")
            self.model_status.setToolTip("")
            self.model_status.setVisible(False)
            return

        self.model_status.setVisible(True)
        weights = resolve_yolo_weights(self.workspace.models_dir)
        probe = probe_yolo(self.workspace.models_dir, try_load=False)

        if not probe.ultralytics:
            self.model_status.setText("模型 不可用")
            self.model_status.setToolTip("检测组件未就绪")
            return
        if weights is None:
            self.model_status.setText("模型 未训练")
            self.model_status.setToolTip("请先在「训练检测」页完成训练")
            return

        mtime = ""
        try:
            from datetime import datetime

            mtime = datetime.fromtimestamp(weights.stat().st_mtime).strftime("%m-%d %H:%M")
        except OSError:
            pass
        label = f"模型 已就绪" + (f" · {mtime}" if mtime else "")
        self.model_status.setText(label)
        tip = f"将使用最新检测模型"
        if mtime:
            tip += f"\n更新于 {mtime}"
        if mode == "both":
            tip += "\n查找：样式+模型"
        elif mode == "ai":
            tip += "\n查找：水印模型"
        self.model_status.setToolTip(tip)

    # Back-compat alias
    def _refresh_yolo_ui(self) -> None:
        self._refresh_model_status()

    def set_device_preference(self, preference: str) -> None:
        pref = self._normalize_pref(preference)
        idx = self._DEVICE_PREFS.index(pref)
        self._suppress_device_signal = True
        self.device_combo.setCurrentIndex(idx)
        self._suppress_device_signal = False
        # Light UI only — skip CUDA probe until page show / run
        backend = self.current_backend()
        lama = backend == "iopaint"
        self.device_combo.setEnabled(lama)
        if not lama:
            self.device_combo.setToolTip("快速预览使用 CPU")
        self.chrome_state_changed.emit()

    def current_device_preference(self) -> str:
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self._DEVICE_PREFS):
            return self._DEVICE_PREFS[idx]
        return "auto"

    def _normalize_pref(self, preference: str) -> str:
        pref = (preference or "auto").strip().lower()
        if pref in {"cuda", "gpu"}:
            return "gpu"
        if pref in {"auto", "cpu", "gpu"}:
            return pref
        return "auto"

    def _on_backend_changed(self, _index: int = 0) -> None:
        self._refresh_device_ui()

    def _ensure_probe(self):
        if self._probe is None:
            try:
                self._probe = probe_cuda()
            except Exception:  # noqa: BLE001
                self._probe = None
        return self._probe

    def _on_device_pref_changed(self, _index: int = 0) -> None:
        if self._suppress_device_signal:
            return
        pref = self.current_device_preference()
        probe = self._ensure_probe()
        if pref == "gpu" and probe is not None and not probe.cuda_available:
            QMessageBox.information(self, "无法使用 GPU", fallback_dialog_text(probe))
            self._suppress_device_signal = True
            self.device_combo.setCurrentIndex(self._DEVICE_PREFS.index("auto"))
            self._suppress_device_signal = False
        self._refresh_device_ui()

    def _refresh_device_ui(self) -> None:
        probe = self._ensure_probe()
        pref = self.current_device_preference()
        backend = self.current_backend()
        lama = backend == "iopaint"
        self.device_combo.setEnabled(lama)
        tip = device_tooltip(pref, probe) if probe is not None else "运行设备"
        if not lama:
            tip = "快速预览使用 CPU"
        self.device_combo.setToolTip(tip)
        self.chrome_state_changed.emit()

    def device_status_extra(self) -> str:
        if self.current_backend() != "iopaint":
            return "快速预览"
        return "高质量"

    def resolve_job_device(self) -> tuple[str, str]:
        if self.current_backend() != "iopaint":
            return "cpu", "设备：CPU（快速预览）"
        pref = self.current_device_preference()
        probe = self._ensure_probe()
        device, log_line, _fell_back = resolve_runtime_device(pref, probe)
        return device, log_line

    # ----- files -----

    def _refresh_counts(self) -> None:
        self.image_count.set_count(len(self._input_paths))
        checked = len(self.selected_profile_ids())
        total = self.profile_list.count()
        self.profile_count.set_count(checked)
        self.profile_count.setToolTip(f"勾选 {checked} / 共 {total}")

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            str(self.workspace.root),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        self._append_paths([Path(p) for p in paths])

    def _add_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(self.workspace.root))
        if not directory:
            return
        root = Path(directory)
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS and p.is_file())
        self._append_paths(paths)

    def _on_paths_dropped(self, paths: list[str]) -> None:
        collected: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                collected.append(path)
            elif path.is_dir():
                collected.extend(
                    sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTS and p.is_file())
                )
        self._append_paths(collected)

    def _append_paths(self, paths: list[Path]) -> None:
        existing = {str(p.resolve()) for p in self._input_paths}
        added = 0
        for path in paths:
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)
            if key in existing:
                continue
            self._input_paths.append(path)
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            self.image_list.addItem(item)
            existing.add(key)
            added += 1
        self._refresh_counts()
        self.log.append(f"加入 {added} 张，队列共 {len(self._input_paths)} 张")
        self.status_message.emit(f"当前队列 {len(self._input_paths)} 张图片")

    def _remove_selected_images(self) -> None:
        rows = sorted({idx.row() for idx in self.image_list.selectedIndexes()}, reverse=True)
        if not rows:
            QMessageBox.information(self, "提示", "请先在列表中选中要移除的图片。")
            return
        for row in rows:
            self.image_list.takeItem(row)
            if 0 <= row < len(self._input_paths):
                del self._input_paths[row]
        self._refresh_counts()
        self.log.append(f"已移除选中项，队列剩 {len(self._input_paths)} 张")

    def _clear_images(self) -> None:
        self._input_paths.clear()
        self.image_list.clear()
        self._refresh_counts()
        self.log.append("已清空待处理图片")

    def _reveal_image(self, item: QListWidgetItem) -> None:
        raw = item.data(Qt.UserRole)
        if not raw:
            return
        path = Path(str(raw))
        if path.exists():
            self._open_path(path.parent if path.is_file() else path)

    # ----- run / stop -----

    def _stop(self) -> None:
        if not self._busy or self._worker is None:
            return
        self._worker.request_cancel()
        self.stop_btn.setEnabled(False)
        self.log.append("—— 已请求停止，等待当前图片处理完… ——")
        self.log.set_stage("正在停止…")
        self.status_message.emit("正在停止…")

    def _run(self) -> None:
        self.log.append("—— 开始处理 ——")
        QApplication.processEvents()

        if self._busy:
            self.log.append("已有处理在进行中，请稍候或点「停止」")
            QMessageBox.information(self, "请稍候", "正在处理中。可点「停止」。")
            return

        try:
            if not self._input_paths:
                self.log.append("请先添加待处理图片")
                QMessageBox.information(self, "提示", "请先添加待处理图片。")
                return

            detect_mode = self.current_detect_mode()
            strategy = self.current_match_strategy()
            profile_ids = self.selected_profile_ids()
            if strategy == "pin" or detect_mode == "styles":
                if not profile_ids:
                    self.log.append("需要至少勾选一个样式")
                    QMessageBox.information(
                        self,
                        "提示",
                        "请至少勾选一个水印样式。\n"
                        "也可将查找改为「水印模型」。",
                    )
                    return
            if detect_mode == "ai" and not profile_ids:
                self.log.append("水印模型：不使用样式库")

            # Always re-read latest model before run
            self._refresh_model_status()
            enable_yolo = self.yolo_enabled()
            if enable_yolo:
                from ...detectors.yolo_watermark import resolve_yolo_weights

                if resolve_yolo_weights(self.workspace.models_dir) is None and detect_mode == "ai":
                    QMessageBox.information(
                        self,
                        "提示",
                        "尚无检测模型，请先在「训练检测」页训练。\n"
                        "或将查找改为「水印样式」。",
                    )
                    return

            for pid in profile_ids:
                self.profile_service.get(pid)

            from ...device_info import clear_probe_cache

            clear_probe_cache()
            self._probe = probe_cuda()
            self._refresh_device_ui()

            backend = self.current_backend()
            device, device_log = self.resolve_job_device()

            self.log.append(f"图片数: {len(self._input_paths)}")
            self.log.append(f"样式: {', '.join(profile_ids) if profile_ids else '（无）'}")
            model_name = {"iopaint": "高质量", "opencv": "快速"}.get(backend, backend)
            strategy_name = {
                "pin": "固定位置",
                "follow": "附近匹配",
                "search": "全图匹配",
                "auto": "附近匹配",
            }.get(strategy, strategy)
            detect_name = {
                "styles": "水印样式",
                "both": "样式+模型",
                "ai": "水印模型",
            }.get(detect_mode, detect_mode)
            self.log.append(f"修补：{model_name}")
            self.log.append(f"定位：{strategy_name}")
            self.log.append(f"查找：{detect_name}")
            self.log.append(device_log)
            self.log.set_stage("准备图片…")
            QApplication.processEvents()

            input_path = self._stage_selected_images()
            self.log.append(f"已准备输入：{input_path.name}")

            if backend == "iopaint":
                from ...backends.iopaint import (
                    ensure_lama_checkpoint,
                    project_root_from_path,
                    resolve_model_dir,
                )

                project_root = project_root_from_path(self.workspace.root)
                model_dir = resolve_model_dir(self.workspace.models_dir, project_root)
                ckpt = ensure_lama_checkpoint(
                    model_dir,
                    [
                        self.workspace.models_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt",
                        Path("data/iopaint-models/torch/hub/checkpoints/big-lama.pt"),
                    ],
                )
                if ckpt is not None:
                    self.log.append(f"修补模型已就绪（约 {ckpt.stat().st_size // (1024 * 1024)} MB）")
                else:
                    self.log.append("本地暂无修补模型，首次运行可能需下载")

            from ...backends.iopaint import project_root_from_path, resolve_model_dir
            from ...detectors.yolo_watermark import ensure_yolo_dir, probe_yolo, resolve_yolo_weights

            yolo_weights = resolve_yolo_weights(self.workspace.models_dir) if enable_yolo else None
            ensure_yolo_dir(self.workspace.models_dir)

            if enable_yolo:
                yolo_info = probe_yolo(
                    self.workspace.models_dir,
                    try_load=False,
                    device=device,
                )
                if yolo_info.ready and yolo_info.weights:
                    self.log.append(f"检测模型：已就绪 · {Path(yolo_info.weights).name}")
                elif yolo_info.status == "missing_ultralytics":
                    self.log.append("检测模型：组件未就绪")
                elif yolo_info.status == "missing_weights":
                    self.log.append("检测模型：尚无模型")
                else:
                    self.log.append("检测模型：未就绪")
            else:
                self.log.append("检测模型：不使用")

            request = RunJobRequest(
                input_path=input_path,
                profile_ids=profile_ids,
                backend=backend,
                iopaint_device=device,
                iopaint_model_dir=(
                    resolve_model_dir(self.workspace.models_dir, project_root_from_path(self.workspace.root))
                    if backend == "iopaint"
                    else self.workspace.models_dir
                ),
                match_strategy=strategy,
                detect_mode=detect_mode,
                enable_yolo=enable_yolo,
                yolo_weights=yolo_weights,
            )
            worker = JobWorker(self.workspace, request)
            worker.log_line.connect(self.log.append)
            worker.stage.connect(self.log.set_stage)
            worker.progress.connect(self.log.set_progress)
            worker.finished_ok.connect(self._on_job_ok)
            worker.failed.connect(self._on_job_failed)

            self._busy = True
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log.set_stage("启动处理…")
            self.log.reset_progress()
            self.log.append("正在启动…")
            self.chrome_state_changed.emit()
            QApplication.processEvents()

            self._worker = worker
            self._thread = start_job_worker(self, worker)
            self._thread.finished.connect(self._on_thread_finished)
            self.log.append("处理已开始")
            self.status_message.emit("正在处理…")

        except Exception as error:  # noqa: BLE001
            detail = traceback.format_exc()
            self.log.append(f"启动失败: {error}")
            self.log.append(detail)
            self._busy = False
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log.set_stage("失败")
            self.chrome_state_changed.emit()
            QMessageBox.warning(self, "启动失败", str(error)[:500])

    def _stage_selected_images(self) -> Path:
        staging = self.workspace.jobs_dir / "_staging_input"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        for path in self._input_paths:
            if not path.exists():
                raise FileNotFoundError(f"图片不存在: {path}")
            target = staging / path.name
            if target.exists():
                stem, suffix = path.stem, path.suffix
                n = 2
                while target.exists():
                    target = staging / f"{stem}_{n}{suffix}"
                    n += 1
            shutil.copy2(path, target)
            self.log.append(f"  已加入 {path.name}")
        return staging

    def _on_thread_finished(self) -> None:
        self._busy = False
        self._thread = None
        self._worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.chrome_state_changed.emit()

    def _on_job_ok(self, result) -> None:
        self._last_job_dir = result.job_dir
        summary = result.summary or {}
        n_hit = summary.get("detected")
        n_img = summary.get("image_count")
        self.log.append(f"结果文件夹：{result.job_dir}")
        if n_img is not None and n_hit is not None:
            empty = summary.get("empty_mask")
            extra = f"，{empty} 张未检出" if empty else ""
            self.log.append(f"摘要：{n_img} 张图，其中 {n_hit} 张有识别结果{extra}")
        yolo_info = summary.get("yolo") or {}
        if yolo_info.get("ready"):
            wt = yolo_info.get("weights") or ""
            name = Path(str(wt)).name if wt else "检测模型"
            self.log.append(f"检测模型：{name}")
        ai = summary.get("ai_stats") or {}
        if ai:
            self.log.append(
                "检测统计："
                f"样式 {ai.get('style_detections', 0)} · "
                f"检测模型 {ai.get('yolo_detections', 0)} · "
                f"基础检测 {ai.get('residual_detections', 0)}"
            )
        cascade = summary.get("cascade_stats") or {}
        if cascade:
            hit = int(cascade.get("style_hit_skip_ai") or 0)
            miss_fill = int(cascade.get("style_miss_ai_fill") or 0)
            weak_fill = int(cascade.get("style_weak_ai_fill") or 0)
            if hit or miss_fill or weak_fill:
                self.log.append(
                    "级联："
                    f"样式命中 {hit} · "
                    f"样式未中补漏 {miss_fill} · "
                    f"样式偏弱补漏 {weak_fill}"
                )
        cancelled = bool(summary.get("cancelled"))
        if cancelled:
            self.log.mark_done("已停止")
            self.job_finished.emit(result)
            self.status_message.emit("已停止")
            QMessageBox.information(
                self,
                "已停止",
                "已停止处理。\n已完成的部分可在「处理结果」中查看。",
            )
            return
        self.log.mark_done("全部完成")
        self.job_finished.emit(result)
        self.status_message.emit("处理完成")
        QMessageBox.information(
            self,
            "处理完成",
            f"批量处理已完成。\n\n"
            f"输出图：\n{result.output_dir}\n\n"
            f"可在左侧「处理结果」中再次打开。",
        )

    def _on_job_failed(self, message: str) -> None:
        self.log.append(f"失败：{message}")
        self.log.set_stage("失败")
        self.status_message.emit("处理失败")
        QMessageBox.warning(self, "处理失败", message[:2000])

    def _open_last_job(self) -> None:
        path = self._last_job_dir or self.workspace.jobs_dir
        self._open_path(path)

    @staticmethod
    def _open_path(path: Path) -> None:
        import os
        import subprocess
        import sys

        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
