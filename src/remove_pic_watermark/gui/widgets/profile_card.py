"""Compact profile card: elided name + icon delete."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...profiles.models import Profile

try:
    from qfluentwidgets import FluentIcon, ToolButton

    def _make_delete_btn(parent: QWidget) -> QPushButton:
        btn = ToolButton(FluentIcon.DELETE, parent)
        btn.setFixedSize(28, 28)
        btn.setToolTip("删除此样式")
        return btn  # type: ignore[return-value]

except ImportError:  # pragma: no cover

    def _make_delete_btn(parent: QWidget) -> QPushButton:
        btn = QPushButton("✕", parent)
        btn.setObjectName("cardDeleteBtn")
        btn.setFixedSize(28, 28)
        btn.setToolTip("删除此样式")
        btn.setCursor(Qt.PointingHandCursor)
        return btn


class ProfileCard(QFrame):
    """
    Compact row:
      [thumb]  Name (elided)          [🗑]
               模板  ☑ 启用
    """

    selected = Signal(str)
    enable_toggled = Signal(str, bool)
    delete_clicked = Signal(str)

    def __init__(self, profile: Profile, profile_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile_id = profile.id
        self._full_name = profile.name or profile.id
        self.setObjectName("profileCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(64)
        self._build(profile, profile_dir)
        self.set_selected(False)

    def _build(self, profile: Profile, profile_dir: Path) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 6, 6)
        row.setSpacing(8)

        self.thumb = QLabel()
        self.thumb.setFixedSize(44, 44)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setObjectName("profileThumb")
        pix = self._load_thumb(profile_dir)
        if pix is not None:
            self.thumb.setPixmap(pix.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.thumb.setText("WM")
        row.addWidget(self.thumb)

        mid = QVBoxLayout()
        mid.setSpacing(2)
        mid.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel()
        self.title.setObjectName("profileTitle")
        self.title.setToolTip(self._full_name)
        mid.addWidget(self.title)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)
        # Locate mode is chosen on batch page — do not show misleading strategy labels
        self.meta = QLabel("样式")
        self.meta.setObjectName("profileMeta")
        self.enable_box = QCheckBox("启用")
        self.enable_box.setObjectName("profileEnable")
        self.enable_box.setChecked(profile.enabled)
        self.enable_box.setToolTip("启用后，批量处理时默认勾选")
        self.enable_box.toggled.connect(self._on_enable)
        meta_row.addWidget(self.meta)
        meta_row.addWidget(self.enable_box)
        meta_row.addStretch(1)
        mid.addLayout(meta_row)
        row.addLayout(mid, 1)

        self.delete_btn = _make_delete_btn(self)
        self.delete_btn.setObjectName("cardDeleteBtn")
        self.delete_btn.clicked.connect(self._emit_delete)
        row.addWidget(self.delete_btn, 0, Qt.AlignVCenter)

        self._apply_elide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        # Leave room for thumb + delete + padding
        max_w = max(40, self.width() - 44 - 28 - 32)
        fm = QFontMetrics(self.title.font())
        self.title.setText(fm.elidedText(self._full_name, Qt.ElideRight, max_w))

    def _emit_delete(self) -> None:
        self.delete_clicked.emit(self.profile_id)

    def _on_enable(self, checked: bool) -> None:
        self.enable_toggled.emit(self.profile_id, bool(checked))

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            child = self.childAt(event.position().toPoint())
            if child is not None and (
                child is self.enable_box
                or child is self.delete_btn
                or self.enable_box.isAncestorOf(child)
                or self.delete_btn.isAncestorOf(child)
            ):
                super().mousePressEvent(event)
                return
            self.selected.emit(self.profile_id)
            event.accept()
            return
        super().mousePressEvent(event)

    @staticmethod
    def _load_thumb(profile_dir: Path) -> QPixmap | None:
        for name in ("preview_overlay.png", "sample_crop.png", "template_mask.png"):
            path = profile_dir / name
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    return pix
        return None
