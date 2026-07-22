"""Icon-first tool buttons for main toolbar and canvas tool rail."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QSizePolicy, QToolButton


class IconToolButton(QToolButton):
    """Compact icon button with tooltip; optional checkable tool-toggle mode."""

    def __init__(
        self,
        icon: QIcon,
        tooltip: str,
        parent=None,
        *,
        checkable: bool = False,
        object_name: str = "iconTool",
        size: int = 32,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setIcon(icon)
        self.setIconSize(QSize(20, 20))
        self.setFixedSize(size, size)
        self.setToolTip(tooltip)
        self.setAutoRaise(True)
        self.setCheckable(checkable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
