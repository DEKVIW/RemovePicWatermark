"""Log + progress.

UX notes (common desktop pattern):
- Determinate bar = completed units / total units; 100% only when work is done.
- Status text sits on its own row (can elide), not fighting the bar for width.
- Unknown-length phases use an indeterminate pulse, then return to determinate.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme_style import FONT_MONO, SPACE_ROW


class LogPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Row 1: progress bar + numeric readout + clear
        bar_row = QHBoxLayout()
        bar_row.setSpacing(SPACE_ROW)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("待命")
        self.progress.setMinimumHeight(22)
        self.count_label = QLabel("0 / 0")
        self.count_label.setStyleSheet("color:#667085; min-width:52px;")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.setFixedWidth(72)
        self.clear_btn.clicked.connect(self.clear)
        bar_row.addWidget(self.progress, 1)
        bar_row.addWidget(self.count_label, 0)
        bar_row.addWidget(self.clear_btn, 0)
        layout.addLayout(bar_row)

        # Row 2: status only (elide long paths/messages)
        self.stage_label = QLabel("就绪")
        self.stage_label.setObjectName("progressStage")
        self.stage_label.setStyleSheet("color:#344054;")
        self.stage_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.stage_label.setWordWrap(False)
        self.stage_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.stage_label)

        # Log body
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setObjectName("jobLogView")
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(FONT_MONO)
        self.view.setFont(font)
        self.view.setMinimumHeight(140)
        self.view.setStyleSheet(
            "QPlainTextEdit#jobLogView {"
            " background: #0f172a;"
            " color: #e2e8f0;"
            " border: 1px solid rgba(148,163,184,0.25);"
            " border-radius: 8px;"
            " padding: 8px;"
            " selection-background-color: #2563eb;"
            "}"
        )
        layout.addWidget(self.view)

        self._total = 0
        self._current = 0

    def append(self, text: str) -> None:
        line = text if text.endswith("\n") else text + "\n"
        stamp = datetime.now().strftime("%H:%M:%S")
        if not line.startswith("["):
            line = f"[{stamp}] {line}"
        self.view.moveCursor(QTextCursor.End)
        self.view.insertPlainText(line)
        self.view.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        self.view.clear()

    def set_stage(self, stage: str) -> None:
        text = stage or "—"
        self.stage_label.setText(text)
        self.stage_label.setToolTip(text)

    def set_progress(self, current: int, total: int) -> None:
        """Report completed work units. 100% only when current >= total > 0."""
        if total <= 0:
            # indeterminate
            self.progress.setRange(0, 0)
            self.count_label.setText("…")
            self.progress.setFormat("处理中")
            return

        self._total = total
        self._current = max(0, min(current, total))
        self.progress.setRange(0, total)
        self.progress.setValue(self._current)
        pct = int(round(100.0 * self._current / total)) if total else 0
        self.progress.setFormat(f"{pct}%")
        self.count_label.setText(f"{self._current} / {total}")

    def reset_progress(self) -> None:
        self._total = 0
        self._current = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("待命")
        self.count_label.setText("0 / 0")
        self.stage_label.setText("就绪")
        self.stage_label.setToolTip("")

    def mark_done(self, message: str = "完成") -> None:
        if self._total > 0:
            self.set_progress(self._total, self._total)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("100%")
            self.count_label.setText("完成")
        self.set_stage(message)
