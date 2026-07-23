"""Main workbench window — profiles / batch / refine / results / train.

Chrome (Inpaint / PS inspired):
  Menu: 文件 | 编辑 | 显示 | 帮助
  Main toolbar: open · undo/redo · canvas tools · zoom · primary action
  Left nav: page modes
  Content: page stack
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QUrl, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSlider,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..workspace import Workspace
from .branding import (
    APP_ABOUT_HTML,
    APP_NAME_ZH,
    APP_WINDOW_TITLE,
    AUTHOR_BLOG_LABEL,
    AUTHOR_BLOG_URL,
    app_icon,
)
from .pages.batch_page import BatchPage
from .pages.profiles_page import ProfilesPage
from .pages.refine_page import RefinePage
from .pages.results_page import ResultsPage
from .pages.train_page import TrainPage
from .state.prefs import load_prefs, save_prefs
from .widgets.icon_tool_button import IconToolButton
from .widgets.image_canvas import CanvasTool
from .widgets.tool_icons import (
    icon_brush,
    icon_clear,
    icon_device,
    icon_eraser,
    icon_open,
    icon_pan,
    icon_play,
    icon_rect,
    icon_redo,
    icon_save,
    icon_stop,
    icon_undo,
    icon_zoom_1x,
    icon_zoom_fit,
    icon_zoom_in,
    icon_zoom_out,
)

try:
    from qfluentwidgets import FluentIcon, NavigationInterface, NavigationItemPosition
except ImportError:  # pragma: no cover
    NavigationInterface = None
    FluentIcon = None
    NavigationItemPosition = None
    from PySide6.QtWidgets import QListWidget


MODE_PROFILES = "profiles"
MODE_BATCH = "batch"
MODE_REFINE = "refine"
MODE_TRAIN = "train"
MODE_RESULTS = "results"


class _GlobalEditShortcutFilter(QObject):
    """Catch Ctrl+Z / Ctrl+Y at application level (beats LineEdit / SpinBox)."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        et = event.type()
        if et not in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride):
            return False
        try:
            key = event.key()
            mods = event.modifiers()
        except Exception:  # noqa: BLE001
            return False
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if not ctrl:
            return False
        if key == Qt.Key.Key_Z and not shift:
            if et == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            self._window._dispatch("undo")
            return True
        if key == Qt.Key.Key_Y or (key == Qt.Key.Key_Z and shift):
            if et == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            self._window._dispatch("redo")
            return True
        return False


class MainWindow(QMainWindow):
    def __init__(self, workspace: Workspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.setWindowTitle(APP_WINDOW_TITLE)
        self.setMinimumSize(960, 640)
        self.resize(1280, 800)
        icon = app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self._prefs = load_prefs(workspace)
        # Flow: 建样式 → 训模型 → 批量 → 精修 → 结果
        self._mode_keys = [
            MODE_PROFILES,
            MODE_TRAIN,
            MODE_BATCH,
            MODE_REFINE,
            MODE_RESULTS,
        ]
        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self._apply_prefs()
        self._sync_chrome()
        # Defer device chip probe (imports torch) until after first paint
        try:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(100, self._refresh_device_chip)
        except Exception:  # noqa: BLE001
            pass
        # Install last so we own Ctrl+Z even when focus is on spinbox / line edit
        self._edit_filter = _GlobalEditShortcutFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._edit_filter)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("文件")
        self.act_open = QAction("打开图片…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(lambda: self._dispatch("open"))
        file_menu.addAction(self.act_open)

        self.act_open_folder = QAction("导入文件夹…", self)
        self.act_open_folder.triggered.connect(lambda: self._dispatch("open_folder"))
        file_menu.addAction(self.act_open_folder)

        self.act_export = QAction("导出…", self)
        self.act_export.setShortcut(QKeySequence.StandardKey.Save)
        self.act_export.triggered.connect(lambda: self._dispatch("export"))
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()

        act_open_ws = QAction("打开数据文件夹…", self)
        act_open_ws.triggered.connect(lambda: self._open_path(self.workspace.root))
        file_menu.addAction(act_open_ws)
        act_open_profiles = QAction("打开样式文件夹", self)
        act_open_profiles.triggered.connect(lambda: self._open_path(self.workspace.profiles_dir))
        file_menu.addAction(act_open_profiles)
        act_open_jobs = QAction("打开处理记录文件夹", self)
        act_open_jobs.triggered.connect(lambda: self._open_path(self.workspace.jobs_dir))
        file_menu.addAction(act_open_jobs)
        file_menu.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        edit_menu = menu.addMenu("编辑")
        self.act_undo = QAction("撤销", self)
        # ApplicationShortcut + window action: Fluent focus / spinbox / list must not eat Ctrl+Z
        self.act_undo.setShortcuts(
            [QKeySequence.StandardKey.Undo, QKeySequence("Ctrl+Z")]
        )
        self.act_undo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_undo.triggered.connect(lambda: self._dispatch("undo"))
        edit_menu.addAction(self.act_undo)
        self.addAction(self.act_undo)  # register on window, not only menubar
        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcuts(
            [QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
        )
        self.act_redo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_redo.triggered.connect(lambda: self._dispatch("redo"))
        edit_menu.addAction(self.act_redo)
        self.addAction(self.act_redo)
        # Hard shortcuts (some Fluent / Qt combos ignore QAction ApplicationShortcut)
        self._sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._sc_undo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_undo.activated.connect(lambda: self._dispatch("undo"))
        self._sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._sc_redo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_redo.activated.connect(lambda: self._dispatch("redo"))
        self._sc_redo2 = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self._sc_redo2.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_redo2.activated.connect(lambda: self._dispatch("redo"))
        edit_menu.addSeparator()
        self.act_clear = QAction("清除选区", self)
        self.act_clear.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.act_clear.triggered.connect(lambda: self._dispatch("clear"))
        edit_menu.addAction(self.act_clear)
        self.act_delete_box = QAction("删除选中框", self)
        self.act_delete_box.setShortcut(QKeySequence.StandardKey.Delete)
        self.act_delete_box.triggered.connect(lambda: self._dispatch("delete_box"))
        edit_menu.addAction(self.act_delete_box)
        self.act_restore = QAction("恢复原图", self)
        self.act_restore.setToolTip("恢复原图")
        self.act_restore.triggered.connect(lambda: self._dispatch("restore"))
        edit_menu.addAction(self.act_restore)

        view_menu = menu.addMenu("显示")
        self.act_zoom_in = QAction("放大", self)
        self.act_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.act_zoom_in.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_zoom_in.triggered.connect(lambda: self._dispatch("zoom_in"))
        view_menu.addAction(self.act_zoom_in)
        self.act_zoom_out = QAction("缩小", self)
        self.act_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.act_zoom_out.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_zoom_out.triggered.connect(lambda: self._dispatch("zoom_out"))
        view_menu.addAction(self.act_zoom_out)
        self.act_zoom_fit = QAction("适应窗口", self)
        self.act_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.act_zoom_fit.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_zoom_fit.triggered.connect(lambda: self._dispatch("zoom_fit"))
        view_menu.addAction(self.act_zoom_fit)
        self.act_zoom_1x = QAction("实际像素 1:1", self)
        self.act_zoom_1x.setShortcut(QKeySequence("Ctrl+1"))
        self.act_zoom_1x.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_zoom_1x.triggered.connect(lambda: self._dispatch("zoom_1x"))
        view_menu.addAction(self.act_zoom_1x)

        # Shared processing settings (batch + refine)
        settings_menu = menu.addMenu("设置")
        backend_menu = settings_menu.addMenu("修补方式")
        self._backend_group = QActionGroup(self)
        self._backend_group.setExclusive(True)
        self.act_backend_lama = QAction("高质量修补", self)
        self.act_backend_lama.setCheckable(True)
        self.act_backend_opencv = QAction("快速预览", self)
        self.act_backend_opencv.setCheckable(True)
        self._backend_group.addAction(self.act_backend_lama)
        self._backend_group.addAction(self.act_backend_opencv)
        backend_menu.addAction(self.act_backend_lama)
        backend_menu.addAction(self.act_backend_opencv)
        self.act_backend_lama.triggered.connect(lambda: self._set_shared_backend("iopaint"))
        self.act_backend_opencv.triggered.connect(lambda: self._set_shared_backend("opencv"))

        device_menu = settings_menu.addMenu("运行设备")
        self._device_group = QActionGroup(self)
        self._device_group.setExclusive(True)
        self.act_device_auto = QAction("自动", self)
        self.act_device_cpu = QAction("CPU", self)
        self.act_device_gpu = QAction("GPU", self)
        for a in (self.act_device_auto, self.act_device_cpu, self.act_device_gpu):
            a.setCheckable(True)
            self._device_group.addAction(a)
            device_menu.addAction(a)
        self.act_device_auto.triggered.connect(lambda: self._set_shared_device("auto"))
        self.act_device_cpu.triggered.connect(lambda: self._set_shared_device("cpu"))
        self.act_device_gpu.triggered.connect(lambda: self._set_shared_device("gpu"))

        help_menu = menu.addMenu("帮助")
        act_blog = QAction(AUTHOR_BLOG_LABEL, self)
        act_blog.setToolTip(AUTHOR_BLOG_URL)
        act_blog.triggered.connect(self._open_author_blog)
        help_menu.addAction(act_blog)
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setObjectName("mainToolBar")
        tb.setMovable(False)
        tb.setFloatable(False)
        from PySide6.QtCore import QSize

        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        self.main_toolbar = tb

        self.tb_open = IconToolButton(icon_open(), "打开", tb)
        self.tb_open.clicked.connect(lambda: self._dispatch("open"))
        tb.addWidget(self.tb_open)

        # Save: default click = export; pages may attach a menu (refine: 另存/覆盖/全部)
        self.tb_save = IconToolButton(icon_save(), "保存", tb)
        self.tb_save.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
        self.tb_save.clicked.connect(self._on_toolbar_save_clicked)
        tb.addWidget(self.tb_save)
        self._export_menu = QMenu(self)

        tb.addSeparator()

        self.tb_undo = IconToolButton(icon_undo(), "撤销", tb)
        self.tb_undo.clicked.connect(lambda: self._dispatch("undo"))
        tb.addWidget(self.tb_undo)

        self.tb_redo = IconToolButton(icon_redo(), "重做", tb)
        self.tb_redo.clicked.connect(lambda: self._dispatch("redo"))
        tb.addWidget(self.tb_redo)

        tb.addSeparator()

        # Canvas tools (shared; visibility depends on current page).
        # QToolBar.addWidget returns a QAction — use that for show/hide.
        self._act_canvas_sep = tb.addSeparator()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._syncing_tool_ui = False
        self._tool_btn_actions: dict[str, object] = {}

        self.tb_rect = IconToolButton(
            icon_rect(),
            "矩形框选",
            tb,
            checkable=True,
        )
        self.tb_paint = IconToolButton(
            icon_brush(),
            "涂抹",
            tb,
            checkable=True,
        )
        self.tb_erase = IconToolButton(
            icon_eraser(),
            "擦除",
            tb,
            checkable=True,
        )
        self.tb_pan = IconToolButton(
            icon_pan(),
            "平移",
            tb,
            checkable=True,
        )
        for key, btn, tool in (
            ("rect", self.tb_rect, CanvasTool.RECT),
            ("paint", self.tb_paint, CanvasTool.PAINT),
            ("erase", self.tb_erase, CanvasTool.ERASE),
            ("pan", self.tb_pan, CanvasTool.PAN),
        ):
            # autoExclusive alone is unreliable across toolbar widgets; group owns it
            btn.setAutoExclusive(False)
            self._tool_group.addButton(btn)
            self._tool_btn_actions[key] = tb.addWidget(btn)
            btn.clicked.connect(lambda _checked=False, t=tool: self._on_toolbar_tool(t))

        self.tb_clear = IconToolButton(
            icon_clear(),
            "清除选区",
            tb,
        )
        self.tb_clear.clicked.connect(lambda: self._dispatch("clear"))
        self._tool_btn_actions["clear"] = tb.addWidget(self.tb_clear)

        self.tb_brush_label = QLabel("18")
        self.tb_brush_label.setObjectName("toolbarBrushLabel")
        self.tb_brush_label.setMinimumWidth(22)
        self.tb_brush_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tb_brush_label.setToolTip("笔刷半径")
        self._tool_btn_actions["brush_label"] = tb.addWidget(self.tb_brush_label)

        self.tb_brush = QSlider(Qt.Orientation.Horizontal)
        self.tb_brush.setObjectName("toolbarBrush")
        self.tb_brush.setRange(4, 64)
        self.tb_brush.setValue(18)
        self.tb_brush.setFixedWidth(96)
        self.tb_brush.setToolTip("笔刷大小")
        self.tb_brush.valueChanged.connect(self._on_toolbar_brush)
        self._tool_btn_actions["brush"] = tb.addWidget(self.tb_brush)

        self._act_zoom_sep = tb.addSeparator()

        self.tb_zoom_in = IconToolButton(icon_zoom_in(), "放大", tb)
        self.tb_zoom_in.clicked.connect(lambda: self._dispatch("zoom_in"))
        tb.addWidget(self.tb_zoom_in)
        self.tb_zoom_out = IconToolButton(icon_zoom_out(), "缩小", tb)
        self.tb_zoom_out.clicked.connect(lambda: self._dispatch("zoom_out"))
        tb.addWidget(self.tb_zoom_out)
        self.tb_zoom_1x = IconToolButton(icon_zoom_1x(), "实际像素 1:1", tb)
        self.tb_zoom_1x.clicked.connect(lambda: self._dispatch("zoom_1x"))
        tb.addWidget(self.tb_zoom_1x)
        self.tb_zoom_fit = IconToolButton(icon_zoom_fit(), "适应窗口", tb)
        self.tb_zoom_fit.clicked.connect(lambda: self._dispatch("zoom_fit"))
        tb.addWidget(self.tb_zoom_fit)

        tb.addSeparator()

        # Icon only — tooltip carries the page-specific action name
        self.tb_run = IconToolButton(icon_play(), "运行", tb, size=36)
        self.tb_run.clicked.connect(lambda: self._dispatch("run"))
        tb.addWidget(self.tb_run)

        self.tb_stop = IconToolButton(icon_stop(), "停止", tb, size=36)
        self.tb_stop.clicked.connect(lambda: self._dispatch("stop"))
        self.tb_stop.setEnabled(False)
        tb.addWidget(self.tb_stop)

        from PySide6.QtWidgets import QSizePolicy

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        # Compact device chip (right side) — click for full tip, no heavy dialog
        self.tb_device = IconToolButton(icon_device(), "运行设备", tb, size=32)
        self.tb_device.setObjectName("toolbarDeviceChip")
        self.tb_device.clicked.connect(self._on_device_chip_clicked)
        tb.addWidget(self.tb_device)
        # Device probe deferred (torch import) — see QTimer after show

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.stack = QStackedWidget()
        self.profiles_page = ProfilesPage(self.workspace)
        self.train_page = TrainPage(self.workspace)
        self.batch_page = BatchPage(self.workspace)
        self.refine_page = RefinePage(self.workspace)
        self.results_page = ResultsPage(self.workspace)
        # Must match self._mode_keys order
        self.stack.addWidget(self.profiles_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.batch_page)
        self.stack.addWidget(self.refine_page)
        self.stack.addWidget(self.results_page)

        self.profiles_page.profiles_changed.connect(self.batch_page.reload_profiles)
        self.profiles_page.status_message.connect(self.statusBar().showMessage)
        self.batch_page.status_message.connect(self.statusBar().showMessage)
        self.refine_page.status_message.connect(self.statusBar().showMessage)
        self.train_page.status_message.connect(self.statusBar().showMessage)
        self.batch_page.job_finished.connect(self._on_job_finished)

        # history enable sync from canvas pages
        for page in (
            self.profiles_page,
            self.batch_page,
            self.refine_page,
            self.train_page,
        ):
            if hasattr(page, "chrome_state_changed"):
                page.chrome_state_changed.connect(self._sync_chrome)

        if NavigationInterface is not None:
            self.navigation = NavigationInterface(self, showMenuButton=True, showReturnButton=False)
            self.navigation.setExpandWidth(176)
            self.navigation.addItem(
                routeKey=MODE_PROFILES,
                icon=FluentIcon.EDIT,
                text="水印样式",
                onClick=lambda: self._switch_mode(MODE_PROFILES),
                position=NavigationItemPosition.TOP,
            )
            self.navigation.addItem(
                routeKey=MODE_TRAIN,
                icon=FluentIcon.MARKET if hasattr(FluentIcon, "MARKET") else FluentIcon.LIBRARY,
                text="训练检测",
                onClick=lambda: self._switch_mode(MODE_TRAIN),
                selectable=True,
                position=NavigationItemPosition.TOP,
            )
            self.navigation.addItem(
                routeKey=MODE_BATCH,
                icon=FluentIcon.PHOTO,
                text="批量去除",
                onClick=lambda: self._switch_mode(MODE_BATCH),
                position=NavigationItemPosition.TOP,
            )
            self.navigation.addItem(
                routeKey=MODE_REFINE,
                icon=FluentIcon.BRUSH if hasattr(FluentIcon, "BRUSH") else FluentIcon.EDIT,
                text="单张精修",
                onClick=lambda: self._switch_mode(MODE_REFINE),
                selectable=True,
                position=NavigationItemPosition.TOP,
            )
            self.navigation.addItem(
                routeKey=MODE_RESULTS,
                icon=FluentIcon.FOLDER,
                text="处理结果",
                onClick=lambda: self._switch_mode(MODE_RESULTS),
                position=NavigationItemPosition.BOTTOM,
            )
            self.navigation.addItem(
                routeKey="workspace",
                icon=FluentIcon.LIBRARY,
                text="数据文件夹",
                onClick=lambda: self._open_path(self.workspace.root),
                position=NavigationItemPosition.BOTTOM,
            )
            outer.addWidget(self.navigation)
        else:
            nav = QListWidget()
            for key, label in (
                (MODE_PROFILES, "水印样式"),
                (MODE_TRAIN, "训练检测"),
                (MODE_BATCH, "批量去除"),
                (MODE_REFINE, "单张精修"),
                (MODE_RESULTS, "处理结果"),
            ):
                nav.addItem(label)
            nav.currentRowChanged.connect(lambda row: self.stack.setCurrentIndex(max(0, row)))
            nav.setFixedWidth(160)
            outer.addWidget(nav)
            self.navigation = nav

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.stack)
        outer.addWidget(content, 1)

        self.statusBar().showMessage("就绪")

    # ------------------------------------------------------------------ canvas tools (toolbar)
    def _on_toolbar_tool(self, tool: CanvasTool) -> None:
        if self._syncing_tool_ui:
            return
        page = self._current_page()
        if hasattr(page, "set_canvas_tool"):
            try:
                page.set_canvas_tool(tool)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"切换工具失败：{exc}", 3000)
        self._sync_tool_highlight(tool)
        self._set_brush_controls_enabled(tool in {CanvasTool.PAINT, CanvasTool.ERASE})

    def _on_toolbar_brush(self, value: int) -> None:
        self.tb_brush_label.setText(str(int(value)))
        if self._syncing_tool_ui:
            return
        page = self._current_page()
        if hasattr(page, "set_canvas_brush"):
            try:
                page.set_canvas_brush(int(value))
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"笔刷调整失败：{exc}", 3000)

    def _sync_tool_highlight(self, tool: CanvasTool | str | None) -> None:
        """Ensure exactly one tool button is checked.

        QButtonGroup exclusive mode ignores setChecked(False) on the only
        checked button, so we briefly turn exclusive off while forcing state.
        """
        self._syncing_tool_ui = True
        try:
            mapping = {
                CanvasTool.RECT: ("rect", self.tb_rect),
                CanvasTool.PAINT: ("paint", self.tb_paint),
                CanvasTool.ERASE: ("erase", self.tb_erase),
                CanvasTool.PAN: ("pan", self.tb_pan),
            }
            t = CanvasTool(tool) if tool is not None else None
            # Must disable exclusivity: otherwise unchecking the active tool is a no-op
            # and the previous button (often RECT) stays visually selected.
            self._tool_group.setExclusive(False)
            active_btn = None
            for btn_tool, (key, btn) in mapping.items():
                act = self._tool_btn_actions.get(key)
                available = bool(act is not None and act.isVisible() and btn.isEnabled())
                want = t is not None and btn_tool == t and available
                btn.setChecked(want)
                if want:
                    active_btn = btn
            self._tool_group.setExclusive(True)
            # Re-assert the active button after re-enabling exclusivity
            if active_btn is not None and not active_btn.isChecked():
                active_btn.setChecked(True)
            self._set_brush_controls_enabled(
                t in {CanvasTool.PAINT, CanvasTool.ERASE} if t is not None else False
            )
        finally:
            self._syncing_tool_ui = False

    def _set_brush_controls_enabled(self, enabled: bool) -> None:
        self.tb_brush.setEnabled(enabled)
        self.tb_brush_label.setEnabled(enabled)

    def _set_action_visible(self, key: str, visible: bool) -> None:
        act = self._tool_btn_actions.get(key)
        if act is not None:
            act.setVisible(bool(visible))

    def _set_canvas_tools_visible(
        self,
        *,
        rect: bool = False,
        paint: bool = False,
        erase: bool = False,
        pan: bool = False,
        clear: bool = False,
        brush: bool = False,
    ) -> None:
        any_tool = rect or paint or erase or pan or clear or brush
        self._act_canvas_sep.setVisible(any_tool)
        pairs = (
            ("rect", self.tb_rect, rect),
            ("paint", self.tb_paint, paint),
            ("erase", self.tb_erase, erase),
            ("pan", self.tb_pan, pan),
            ("clear", self.tb_clear, clear),
        )
        for key, btn, vis in pairs:
            self._set_action_visible(key, vis)
            btn.setEnabled(vis)
        show_brush = brush and (paint or erase)
        self._set_action_visible("brush", show_brush)
        self._set_action_visible("brush_label", show_brush)
        self.tb_brush.setEnabled(show_brush)
        self.tb_brush_label.setEnabled(show_brush)
        if not any_tool:
            self._sync_tool_highlight(None)

    def _apply_page_canvas_tools(self) -> None:
        """Show/hide canvas tools from current page and sync checked + brush."""
        page = self._current_page()
        caps: dict = {}
        if hasattr(page, "canvas_tool_caps"):
            try:
                caps = dict(page.canvas_tool_caps() or {})
            except Exception:  # noqa: BLE001
                caps = {}
        if not caps:
            self._set_canvas_tools_visible()
            return
        self._set_canvas_tools_visible(
            rect=bool(caps.get("rect")),
            paint=bool(caps.get("paint")),
            erase=bool(caps.get("erase")),
            pan=bool(caps.get("pan")),
            clear=bool(caps.get("clear", True)),
            brush=bool(caps.get("brush", caps.get("paint") or caps.get("erase"))),
        )
        # brush value
        radius = 18
        if hasattr(page, "canvas_brush_radius"):
            try:
                radius = int(page.canvas_brush_radius())
            except Exception:  # noqa: BLE001
                radius = 18
        self._syncing_tool_ui = True
        try:
            self.tb_brush.blockSignals(True)
            self.tb_brush.setValue(max(4, min(64, radius)))
            self.tb_brush_label.setText(str(max(4, min(64, radius))))
            self.tb_brush.blockSignals(False)
        finally:
            self._syncing_tool_ui = False
        tool = None
        if hasattr(page, "current_canvas_tool"):
            try:
                tool = page.current_canvas_tool()
            except Exception:  # noqa: BLE001
                tool = None
        if tool is None:
            if caps.get("rect"):
                tool = CanvasTool.RECT
            elif caps.get("paint"):
                tool = CanvasTool.PAINT
            elif caps.get("pan"):
                tool = CanvasTool.PAN
        self._sync_tool_highlight(tool)

    # ------------------------------------------------------------------ dispatch
    def _current_page(self) -> QWidget:
        return self.stack.currentWidget()

    def _dispatch(self, action: str) -> None:
        page = self._current_page()
        handled = False
        if hasattr(page, "handle_app_action"):
            try:
                handled = bool(page.handle_app_action(action))
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"操作失败：{exc}", 5000)
                handled = True
        if not handled:
            # Generic canvas helpers if page exposes active_canvas()
            canvas = getattr(page, "active_canvas", lambda: None)()
            if canvas is not None:
                if action == "undo" and hasattr(canvas, "undo_annotation"):
                    handled = bool(canvas.undo_annotation())
                elif action == "redo" and hasattr(canvas, "redo_annotation"):
                    handled = bool(canvas.redo_annotation())
                elif action == "clear" and hasattr(canvas, "clear_roi"):
                    canvas.clear_roi()
                    handled = True
                elif action == "delete_box":
                    if hasattr(canvas, "delete_selection_or_box"):
                        handled = bool(canvas.delete_selection_or_box())
                    elif hasattr(canvas, "pop_last_multi_box"):
                        handled = bool(canvas.pop_last_multi_box())
                elif action == "zoom_in" and hasattr(canvas, "zoom_in"):
                    canvas.zoom_in()
                    handled = True
                elif action == "zoom_out" and hasattr(canvas, "zoom_out"):
                    canvas.zoom_out()
                    handled = True
                elif action == "zoom_fit" and hasattr(canvas, "reset_view"):
                    canvas.reset_view()
                    handled = True
                elif action == "zoom_1x" and hasattr(canvas, "zoom_actual"):
                    canvas.zoom_actual()
                    handled = True
        if action == "undo" and not handled:
            # Train page: try every tile that has history (focus may have left the edited cell)
            if hasattr(page, "_tiles"):
                for tile in getattr(page, "_tiles", []) or []:
                    c = getattr(tile, "canvas", None)
                    if c is not None and hasattr(c, "undo_annotation") and c.can_undo_annotation():
                        if c.undo_annotation():
                            handled = True
                            break
            if not handled:
                self.statusBar().showMessage("没有可撤销的操作", 2000)
        elif action == "redo" and not handled:
            self.statusBar().showMessage("没有可重做的操作", 2000)
        self._sync_chrome()

    def _sync_chrome(self) -> None:
        """Enable/disable menu + toolbar from current page capabilities."""
        page = self._current_page()
        caps: set[str] = set()
        if hasattr(page, "app_action_caps"):
            try:
                caps = set(page.app_action_caps() or [])
            except Exception:  # noqa: BLE001
                caps = set()
        # Defaults by mode if page silent
        mode = self._mode_keys[self.stack.currentIndex()] if self.stack.count() else ""
        if not caps:
            if mode in {MODE_PROFILES, MODE_REFINE, MODE_TRAIN}:
                caps = {
                    "open",
                    "export",
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
            elif mode == MODE_BATCH:
                caps = {"open", "open_folder", "run"}
            elif mode == MODE_RESULTS:
                caps = {"export"}

        def _en(name: str) -> bool:
            return name in caps

        for act, name in (
            (self.act_open, "open"),
            (self.act_open_folder, "open_folder"),
            (self.act_export, "export"),
            (self.act_clear, "clear"),
            (self.act_delete_box, "delete_box"),
            (self.act_restore, "restore"),
            (self.act_zoom_in, "zoom_in"),
            (self.act_zoom_out, "zoom_out"),
            (self.act_zoom_fit, "zoom_fit"),
            (self.act_zoom_1x, "zoom_1x"),
        ):
            act.setEnabled(_en(name))

        # Undo/redo: ask page for live state
        can_undo = False
        can_redo = False
        if hasattr(page, "can_undo"):
            try:
                can_undo = bool(page.can_undo())
            except Exception:  # noqa: BLE001
                can_undo = False
        if hasattr(page, "can_redo"):
            try:
                can_redo = bool(page.can_redo())
            except Exception:  # noqa: BLE001
                can_redo = False
        # Always keep undo/redo enabled on edit-capable pages so shortcuts fire;
        # empty stack still shows status "没有可撤销的操作".
        self.act_undo.setEnabled(_en("undo"))
        self.act_redo.setEnabled(_en("redo"))
        # Toolbar: enable whenever page supports undo (even if stack empty) so
        # click always reaches dispatch; visual "active" when can_undo.
        self.tb_undo.setEnabled(_en("undo"))
        self.tb_redo.setEnabled(_en("redo"))
        if hasattr(self.tb_undo, "setToolTip"):
            self.tb_undo.setToolTip("撤销")
            self.tb_redo.setToolTip("重做")
        self.tb_open.setEnabled(_en("open") or _en("open_folder"))
        self.tb_save.setEnabled(_en("export"))
        for btn, name in (
            (self.tb_zoom_in, "zoom_in"),
            (self.tb_zoom_out, "zoom_out"),
            (self.tb_zoom_fit, "zoom_fit"),
            (self.tb_zoom_1x, "zoom_1x"),
        ):
            btn.setEnabled(_en(name))

        run_enabled = _en("run")
        stop_enabled = _en("stop")
        if hasattr(page, "is_busy"):
            try:
                busy = bool(page.is_busy())
                if busy:
                    run_enabled = False
                    stop_enabled = True
            except Exception:  # noqa: BLE001
                pass
        self.tb_run.setEnabled(run_enabled)
        self.tb_stop.setEnabled(stop_enabled)

        # Page-specific tooltips (structured chrome)
        labels = {
            "open": "打开",
            "export": "保存",
            "run": "运行",
        }
        if hasattr(page, "toolbar_action_labels"):
            try:
                labels.update(dict(page.toolbar_action_labels() or {}))
            except Exception:  # noqa: BLE001
                pass
        self.tb_open.setToolTip(labels.get("open", "打开"))
        self.tb_save.setToolTip(labels.get("export", "保存"))
        self.tb_run.setToolTip(labels.get("run", "运行"))
        self.tb_stop.setToolTip("停止")

        # Export submenu when page provides export_menu_actions
        self._export_menu.clear()
        menu_actions: list = []
        if hasattr(page, "export_menu_actions"):
            try:
                menu_actions = list(page.export_menu_actions() or [])
            except Exception:  # noqa: BLE001
                menu_actions = []
        if menu_actions and _en("export"):
            for label, act_id in menu_actions:
                act = self._export_menu.addAction(label)
                act.triggered.connect(
                    lambda _checked=False, a=act_id: self._dispatch(a)
                )
            self.tb_save.setMenu(self._export_menu)
            self.tb_save.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        else:
            self.tb_save.setMenu(None)
            self.tb_save.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)

        # Canvas tools live on the main toolbar (not page side rails)
        self._apply_page_canvas_tools()
        # Device chip tip: refreshed by deferred timer / user click (not every chrome sync)

    def _on_toolbar_save_clicked(self) -> None:
        """Primary save click — always default export action."""
        self._dispatch("export")

    def _switch_mode(self, mode: str) -> None:
        if mode not in self._mode_keys:
            return
        index = self._mode_keys.index(mode)
        self.stack.setCurrentIndex(index)
        self.stack.widget(index).show()
        self._prefs["nav_mode"] = mode
        if NavigationInterface is not None and hasattr(self, "navigation"):
            nav = self.navigation
            if hasattr(nav, "setCurrentItem"):
                try:
                    nav.setCurrentItem(mode)
                except Exception:  # noqa: BLE001
                    pass
        self.statusBar().showMessage(
            {
                MODE_PROFILES: "水印样式",
                MODE_BATCH: "批量去除",
                MODE_REFINE: "单张精修",
                MODE_TRAIN: "训练检测",
                MODE_RESULTS: "处理结果",
            }.get(mode, "就绪")
        )
        # Refresh batch model status when entering batch (latest weights after train)
        if mode == MODE_BATCH and hasattr(self.batch_page, "_refresh_model_status"):
            try:
                self.batch_page._refresh_model_status()
            except Exception:  # noqa: BLE001
                pass
        # Avoid torch probe on every nav switch during startup; chip has its own timer
        self._sync_chrome()

    def _on_job_finished(self, result) -> None:
        self.results_page.refresh()
        self.results_page.select_job(result.job_dir)
        self._switch_mode(MODE_RESULTS)
        if NavigationInterface is not None and hasattr(self.navigation, "setCurrentItem"):
            try:
                self.navigation.setCurrentItem(MODE_RESULTS)
            except Exception:  # noqa: BLE001
                pass

    def _set_shared_backend(self, backend: str) -> None:
        backend = "iopaint" if backend in {"iopaint", "lama"} else "opencv"
        self._prefs["backend"] = backend
        if hasattr(self, "batch_page"):
            self.batch_page.set_backend(backend)
        if hasattr(self, "refine_page"):
            self.refine_page.set_backend(backend)
        self._sync_settings_menu()

    def _set_shared_device(self, pref: str) -> None:
        pref = str(pref).lower()
        if pref in {"cuda", "gpu"}:
            pref = "gpu"
        elif pref == "cpu":
            pref = "cpu"
        else:
            pref = "auto"
        self._prefs["device_preference"] = pref
        self._prefs["iopaint_device"] = pref
        for page_name in (
            "batch_page",
            "refine_page",
            "train_page",
            "profiles_page",
        ):
            page = getattr(self, page_name, None)
            if page is not None and hasattr(page, "set_device_preference"):
                page.set_device_preference(pref)
        self._sync_settings_menu()
        self._refresh_device_chip()

    def _device_preference(self) -> str:
        pref = self._prefs.get("device_preference") or "auto"
        pref = str(pref).lower()
        if pref in {"cuda", "gpu"}:
            return "gpu"
        if pref == "cpu":
            return "cpu"
        return "auto"

    def _refresh_device_chip(self) -> None:
        """Update toolbar device chip label/tooltip from prefs + current page extra."""
        if not hasattr(self, "tb_device"):
            return
        from ..device_info import (
            device_tooltip,
            header_device_caption,
            probe_cuda,
        )

        pref = self._device_preference()
        try:
            probe = probe_cuda()
        except Exception:  # noqa: BLE001
            probe = None
        extra = ""
        page = self._current_page() if hasattr(self, "stack") else None
        if page is not None and hasattr(page, "device_status_extra"):
            try:
                extra = str(page.device_status_extra() or "")
            except Exception:  # noqa: BLE001
                extra = ""
        tip = device_tooltip(pref, probe)
        if extra:
            tip = f"{tip}\n{extra}"
        self.tb_device.setToolTip(tip)
        self._device_chip_summary = header_device_caption(pref, probe, extra=extra)

    def _on_device_chip_clicked(self) -> None:
        """Lightweight feedback: status bar only (no dialog)."""
        self._refresh_device_chip()
        text = getattr(self, "_device_chip_summary", "") or "设备状态未知"
        self.statusBar().showMessage(text, 8000)

    def _sync_settings_menu(self) -> None:
        backend = self._prefs.get("backend") or "iopaint"
        if backend == "opencv":
            self.act_backend_opencv.setChecked(True)
        else:
            self.act_backend_lama.setChecked(True)
        pref = self._prefs.get("device_preference") or "auto"
        if pref == "gpu":
            self.act_device_gpu.setChecked(True)
        elif pref == "cpu":
            self.act_device_cpu.setChecked(True)
        else:
            self.act_device_auto.setChecked(True)

    def _apply_prefs(self) -> None:
        backend = self._prefs.get("backend") or "iopaint"
        self.batch_page.set_backend(str(backend))
        device_pref = self._prefs.get("device_preference") or self._prefs.get("iopaint_device") or "auto"
        if str(device_pref).lower() in {"cuda", "gpu"}:
            device_pref = "gpu"
        elif str(device_pref).lower() == "cpu":
            device_pref = "cpu"
        else:
            device_pref = "auto"
        self.batch_page.set_device_preference(str(device_pref))
        self.refine_page.set_backend(str(backend))
        self.refine_page.set_device_preference(str(device_pref))
        if hasattr(self, "train_page") and hasattr(self.train_page, "set_device_preference"):
            self.train_page.set_device_preference(str(device_pref))
        if hasattr(self, "profiles_page") and hasattr(
            self.profiles_page, "set_device_preference"
        ):
            self.profiles_page.set_device_preference(str(device_pref))
        self._sync_settings_menu()
        # Device chip probes torch — deferred after first paint (see QTimer in __init__)
        strategy = self._prefs.get("match_strategy") or "follow"
        if str(strategy) == "auto":
            strategy = "follow"
        self.batch_page.set_match_strategy(str(strategy))
        detect_mode = self._prefs.get("detect_mode") or "styles"
        self.batch_page.set_detect_mode(str(detect_mode))
        mode = self._prefs.get("nav_mode") or MODE_PROFILES
        if mode not in self._mode_keys:
            mode = MODE_PROFILES
        self._switch_mode(str(mode))
        geometry = self._prefs.get("window_geometry")
        if isinstance(geometry, str) and self._prefs.get("geometry_version") == 2:
            try:
                self.restoreGeometry(bytes.fromhex(geometry))
            except (ValueError, TypeError):
                pass

    def _persist_prefs(self) -> None:
        self._prefs["backend"] = self.batch_page.current_backend()
        self._prefs["device_preference"] = self.batch_page.current_device_preference()
        self._prefs["iopaint_device"] = self._prefs["device_preference"]
        self._prefs["match_strategy"] = self.batch_page.current_match_strategy()
        self._prefs["detect_mode"] = self.batch_page.current_detect_mode()
        self._prefs["selected_profiles"] = self.batch_page.selected_profile_ids()
        self._prefs["window_geometry"] = self.saveGeometry().data().hex()
        self._prefs["geometry_version"] = 2
        save_prefs(self._prefs, self.workspace)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._persist_prefs()
        super().closeEvent(event)

    def _about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(f"关于 {APP_NAME_ZH}")
        box.setIcon(QMessageBox.Icon.Information)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        box.setText(APP_ABOUT_HTML)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _open_author_blog(self) -> None:
        QDesktopServices.openUrl(QUrl(AUTHOR_BLOG_URL))

    @staticmethod
    def _open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
