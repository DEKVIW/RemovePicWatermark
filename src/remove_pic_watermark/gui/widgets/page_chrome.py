"""Shared page chrome: headers, section panels, option groups, inspector.

Layout conventions (PS / Figma / Fluent workbench):
- Canvas-first: thin tool strip over the stage
- Parameters in a right inspector (collapsible)
- Primary action in inspector footer or bottom bar
- Captions secondary — never fight titles for space
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme_style import SPACE_CARD_GAP, SPACE_CARD_H, SPACE_CARD_V, SPACE_ROW

try:
    from qfluentwidgets import BodyLabel, PushButton, SubtitleLabel
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QLabel as BodyLabel
    from PySide6.QtWidgets import QLabel as SubtitleLabel
    from PySide6.QtWidgets import QPushButton as PushButton


class PageHeader(QWidget):
    """Compact title row + optional one-line caption + trailing actions."""

    def __init__(
        self,
        title: str,
        caption: str = "",
        parent: QWidget | None = None,
        *,
        compact: bool = True,
    ) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE_ROW)

        self.title_label = SubtitleLabel(title)
        self.title_label.setProperty("role", "pageTitle")
        row.addWidget(self.title_label, 0)

        self.caption_label = BodyLabel("")
        self.caption_label.setObjectName("deviceStatus")
        self.caption_label.setProperty("role", "caption")
        self.caption_label.setWordWrap(False)
        self.caption_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if compact:
            self.caption_label.setMaximumWidth(720)
        # Place caption immediately after title (not stretched away)
        row.addWidget(self.caption_label, 0)
        row.addStretch(1)
        if caption:
            self.set_caption(caption)
        else:
            self.caption_label.hide()

        self.actions = QHBoxLayout()
        self.actions.setContentsMargins(0, 0, 0, 0)
        self.actions.setSpacing(SPACE_ROW)
        row.addLayout(self.actions, 0)

    def set_caption(self, text: str) -> None:
        """Show status right after the title, wrapped in parentheses."""
        text = (text or "").strip()
        if text:
            # Avoid double-wrapping if caller already passed parentheses
            already = (
                (text.startswith("(") and text.endswith(")"))
                or (text.startswith("（") and text.endswith("）"))
            )
            if not already:
                text = f"（{text}）"
            self.caption_label.setText(text)
            self.caption_label.setVisible(True)
        else:
            self.caption_label.setText("")
            self.caption_label.setVisible(False)

    def set_caption_tooltip(self, text: str) -> None:
        self.caption_label.setToolTip(text or "")

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget)


class SectionPanel(QFrame):
    """White rounded panel with optional section title."""

    def __init__(
        self,
        title: str = "",
        *,
        object_name: str = "workPanel",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(SPACE_CARD_H, SPACE_CARD_V, SPACE_CARD_H, SPACE_CARD_V)
        self._root.setSpacing(SPACE_ROW)

        self.title_label: QLabel | None = None
        if title:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(SPACE_ROW)
            self.title_label = BodyLabel(title)
            self.title_label.setProperty("role", "section")
            head.addWidget(self.title_label)
            head.addStretch(1)
            self.header_actions = QHBoxLayout()
            self.header_actions.setContentsMargins(0, 0, 0, 0)
            self.header_actions.setSpacing(6)
            head.addLayout(self.header_actions)
            self._root.addLayout(head)
        else:
            self.header_actions = QHBoxLayout()

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SPACE_ROW)
        self._root.addLayout(self.body, 1)

    def add_header_widget(self, widget: QWidget) -> None:
        self.header_actions.addWidget(widget)

    def set_title(self, text: str) -> None:
        if self.title_label is not None:
            self.title_label.setText(text)


class ToolBar(QFrame):
    """Compact horizontal toolbar strip (use slim=True over canvas)."""

    def __init__(self, parent: QWidget | None = None, *, slim: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("canvasToolBar" if slim else "toolbarBar")
        self.row = QHBoxLayout(self)
        if slim:
            self.row.setContentsMargins(8, 4, 8, 4)
            self.row.setSpacing(6)
        else:
            self.row.setContentsMargins(10, 8, 10, 8)
            self.row.setSpacing(8)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self.row.addWidget(widget, stretch)

    def add_stretch(self, stretch: int = 1) -> None:
        self.row.addStretch(stretch)

    def add_spacing(self, px: int = 10) -> None:
        self.row.addSpacing(px)

    def add_label(self, text: str) -> BodyLabel:
        label = BodyLabel(text)
        label.setProperty("role", "caption")
        self.row.addWidget(label)
        return label


class InspectorPanel(QFrame):
    """Right-side properties panel (PS / Figma inspector style).

    - Expanded: fixed width body + footer primary actions
    - Collapsed: thin rail with expand button only
    Signals stay on child widgets; this only reparents layout chrome.
    """

    collapsed_changed = Signal(bool)

    def __init__(
        self,
        title: str = "设置",
        parent: QWidget | None = None,
        *,
        width: int = 280,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("inspectorPanel")
        self._expanded_width = width
        self._collapsed = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # collapsed rail
        self._rail = QFrame()
        self._rail.setObjectName("inspectorRail")
        self._rail.setFixedWidth(36)
        rail_l = QVBoxLayout(self._rail)
        rail_l.setContentsMargins(4, 8, 4, 8)
        rail_l.setSpacing(6)
        self._expand_btn = PushButton("›")
        self._expand_btn.setObjectName("inspectorToggle")
        self._expand_btn.setFixedSize(28, 28)
        self._expand_btn.setToolTip("展开设置")
        self._expand_btn.clicked.connect(self.expand)
        rail_l.addWidget(self._expand_btn, 0, Qt.AlignHCenter)
        rail_l.addStretch(1)
        self._rail.hide()
        outer.addWidget(self._rail)

        # expanded column
        self._panel = QFrame()
        self._panel.setObjectName("inspectorBody")
        self._panel.setFixedWidth(width)
        col = QVBoxLayout(self._panel)
        col.setContentsMargins(12, 10, 12, 12)
        col.setSpacing(10)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self._title = BodyLabel(title)
        self._title.setProperty("role", "section")
        head.addWidget(self._title, 1)
        self._collapse_btn = PushButton("‹")
        self._collapse_btn.setObjectName("inspectorToggle")
        self._collapse_btn.setFixedSize(28, 28)
        self._collapse_btn.setToolTip("收起设置")
        self._collapse_btn.clicked.connect(self.collapse)
        head.addWidget(self._collapse_btn)
        col.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(10)
        col.addLayout(self.body, 1)

        self.footer = QVBoxLayout()
        self.footer.setContentsMargins(0, 4, 0, 0)
        self.footer.setSpacing(8)
        col.addLayout(self.footer)

        outer.addWidget(self._panel)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def add_body(self, widget: QWidget) -> None:
        self.body.addWidget(widget)

    def add_footer(self, widget: QWidget) -> None:
        self.footer.addWidget(widget)

    def add_field(self, label: str, widget: QWidget) -> None:
        """Caption above control (inspector form stack)."""
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = BodyLabel(label)
        cap.setProperty("role", "caption")
        lay.addWidget(cap)
        lay.addWidget(widget)
        self.body.addWidget(wrap)

    def collapse(self) -> None:
        if self._collapsed:
            return
        self._collapsed = True
        self._panel.hide()
        self._rail.show()
        self.setFixedWidth(36)
        self.collapsed_changed.emit(True)

    def expand(self) -> None:
        if not self._collapsed:
            return
        self._collapsed = False
        self._rail.hide()
        self._panel.show()
        self.setFixedWidth(self._expanded_width)
        self.collapsed_changed.emit(False)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        if self._collapsed:
            self.expand()
        else:
            self.collapse()


class OptionGroup(QFrame):
    """Labeled control cluster for batch options (去水印 / 定位 / 设备)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("optionGroup")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.title = BodyLabel(title)
        self.title.setProperty("role", "caption")
        layout.addWidget(self.title)
        self.row = QHBoxLayout()
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(6)
        layout.addLayout(self.row)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self.row.addWidget(widget, stretch)


class ActionBar(QFrame):
    """Bottom action strip: hint left, buttons right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(10, 8, 10, 8)
        self.row.setSpacing(8)
        self.hint = BodyLabel("")
        self.hint.setProperty("role", "caption")
        self.hint.setWordWrap(False)
        self.hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.row.addWidget(self.hint, 1)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)
        self.hint.setToolTip(text)

    def add(self, widget: QWidget) -> None:
        self.row.addWidget(widget)


class CountBadge(QLabel):
    """Small count chip for list headers."""

    def __init__(self, text: str = "0", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("countBadge")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(28)

    def set_count(self, n: int) -> None:
        self.setText(str(n))


def form_row(label: str, *widgets: QWidget) -> QHBoxLayout:
    """Label + controls horizontal row."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(SPACE_ROW)
    cap = BodyLabel(label)
    cap.setProperty("role", "caption")
    row.addWidget(cap)
    for w in widgets:
        row.addWidget(w)
    return row


__all__ = [
    "ActionBar",
    "CountBadge",
    "InspectorPanel",
    "OptionGroup",
    "PageHeader",
    "SectionPanel",
    "ToolBar",
    "form_row",
    "SPACE_CARD_GAP",
]
