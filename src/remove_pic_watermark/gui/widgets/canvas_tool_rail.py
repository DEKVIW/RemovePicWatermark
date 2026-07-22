"""Left vertical tool rail (Inpaint / PS style) shared by canvas pages."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QLabel, QSlider, QVBoxLayout

from .icon_tool_button import IconToolButton
from .image_canvas import CanvasTool
from .tool_icons import (
    icon_brush,
    icon_clear,
    icon_eraser,
    icon_pan,
    icon_rect,
    icon_undo,
)


class CanvasToolRail(QFrame):
    """Mutually exclusive tools (rect / paint / erase / pan) + brush + commands."""

    tool_changed = Signal(object)  # CanvasTool
    brush_changed = Signal(int)
    clear_clicked = Signal()
    undo_clicked = Signal()
    restore_clicked = Signal()  # optional: restore original (refine)

    def __init__(
        self,
        parent=None,
        *,
        allow_rect: bool = True,
        allow_paint: bool = True,
        allow_erase: bool = True,
        allow_pan: bool = True,
        show_brush: bool = True,
        show_clear: bool = True,
        show_undo: bool = True,
        show_restore: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolRail")
        self.setFixedWidth(48)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(6)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self.rect_btn: IconToolButton | None = None
        self.paint_btn: IconToolButton | None = None
        self.erase_btn: IconToolButton | None = None
        self.pan_btn: IconToolButton | None = None

        if allow_rect:
            self.rect_btn = IconToolButton(
                icon_rect(),
                "矩形框选",
                self,
                checkable=True,
                object_name="railTool",
            )
            self._group.addButton(self.rect_btn)
            lay.addWidget(self.rect_btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self.rect_btn.clicked.connect(lambda: self._emit_tool(CanvasTool.RECT))

        if allow_paint:
            self.paint_btn = IconToolButton(
                icon_brush(),
                "涂抹",
                self,
                checkable=True,
                object_name="railTool",
            )
            self._group.addButton(self.paint_btn)
            lay.addWidget(self.paint_btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self.paint_btn.clicked.connect(lambda: self._emit_tool(CanvasTool.PAINT))

        if allow_erase and allow_paint:
            self.erase_btn = IconToolButton(
                icon_eraser(),
                "擦除",
                self,
                checkable=True,
                object_name="railTool",
            )
            self._group.addButton(self.erase_btn)
            lay.addWidget(self.erase_btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self.erase_btn.clicked.connect(lambda: self._emit_tool(CanvasTool.ERASE))

        if allow_pan:
            self.pan_btn = IconToolButton(
                icon_pan(),
                "平移",
                self,
                checkable=True,
                object_name="railTool",
            )
            self._group.addButton(self.pan_btn)
            lay.addWidget(self.pan_btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self.pan_btn.clicked.connect(lambda: self._emit_tool(CanvasTool.PAN))

        lay.addSpacing(8)

        if show_undo:
            self.undo_btn = IconToolButton(
                icon_undo(), "撤销", self, object_name="railTool"
            )
            self.undo_btn.clicked.connect(self.undo_clicked.emit)
            self.undo_btn.setEnabled(False)
            lay.addWidget(self.undo_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        else:
            self.undo_btn = None

        if show_clear:
            self.clear_btn = IconToolButton(
                icon_clear(), "清除选区", self, object_name="railTool"
            )
            self.clear_btn.clicked.connect(self.clear_clicked.emit)
            lay.addWidget(self.clear_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        else:
            self.clear_btn = None

        if show_restore:
            from .tool_icons import icon_reload

            self.restore_btn = IconToolButton(
                icon_reload(), "恢复原图", self, object_name="railTool"
            )
            self.restore_btn.clicked.connect(self.restore_clicked.emit)
            self.restore_btn.setEnabled(False)
            lay.addWidget(self.restore_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        else:
            self.restore_btn = None

        lay.addStretch(1)

        self._brush_block = QFrame()
        brush_l = QVBoxLayout(self._brush_block)
        brush_l.setContentsMargins(0, 0, 0, 0)
        brush_l.setSpacing(2)
        self.brush_label = QLabel("18")
        self.brush_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.brush_label.setObjectName("railCaption")
        self.brush_slider = QSlider(Qt.Orientation.Vertical)
        self.brush_slider.setRange(4, 64)
        self.brush_slider.setValue(18)
        self.brush_slider.setFixedHeight(88)
        self.brush_slider.setToolTip("笔刷大小")
        self.brush_slider.valueChanged.connect(self._on_brush)
        brush_l.addWidget(self.brush_slider, 0, Qt.AlignmentFlag.AlignHCenter)
        brush_l.addWidget(self.brush_label)
        lay.addWidget(self._brush_block, 0, Qt.AlignmentFlag.AlignHCenter)
        self._brush_block.setVisible(show_brush and allow_paint)

        # default tool
        if self.rect_btn is not None:
            self.rect_btn.setChecked(True)
            self._set_brush_enabled(False)
        elif self.paint_btn is not None:
            self.paint_btn.setChecked(True)
            self._set_brush_enabled(True)

    def _set_brush_enabled(self, enabled: bool) -> None:
        self._brush_block.setEnabled(enabled)

    def _emit_tool(self, tool: CanvasTool) -> None:
        self._set_brush_enabled(tool in {CanvasTool.PAINT, CanvasTool.ERASE})
        self.tool_changed.emit(tool)

    def _on_brush(self, value: int) -> None:
        self.brush_label.setText(str(value))
        self.brush_changed.emit(int(value))

    def set_tool(self, tool: CanvasTool) -> None:
        mapping = {
            CanvasTool.RECT: self.rect_btn,
            CanvasTool.PAINT: self.paint_btn,
            CanvasTool.ERASE: self.erase_btn,
            CanvasTool.PAN: self.pan_btn,
        }
        btn = mapping.get(tool)
        if btn is not None:
            btn.setChecked(True)
        self._set_brush_enabled(tool in {CanvasTool.PAINT, CanvasTool.ERASE})

    def set_brush_radius(self, radius: int) -> None:
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(int(radius))
        self.brush_label.setText(str(int(radius)))
        self.brush_slider.blockSignals(False)

    def set_undo_enabled(self, enabled: bool) -> None:
        if self.undo_btn is not None:
            self.undo_btn.setEnabled(bool(enabled))

    def set_restore_enabled(self, enabled: bool) -> None:
        if self.restore_btn is not None:
            self.restore_btn.setEnabled(bool(enabled))
