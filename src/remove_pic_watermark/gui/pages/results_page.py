"""Browse recent batch results and open output folders."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...workspace import Workspace
from ..theme_style import SPACE_CARD_GAP, SPACE_PAGE, SPACE_ROW
from ..widgets.icon_tool_button import IconToolButton
from ..widgets.page_chrome import CountBadge, SectionPanel
from ..widgets.tool_icons import icon_debug, icon_folder_open, icon_mask, icon_open, icon_refresh

try:
    from qfluentwidgets import PrimaryPushButton, PushButton, TextEdit
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QPlainTextEdit as TextEdit
    from PySide6.QtWidgets import QPushButton as PrimaryPushButton
    from PySide6.QtWidgets import QPushButton as PushButton


_ACTION_ZH = {
    "detected": "已识别",
    "empty_mask": "未检出",
    "opencv_inpainted": "已去除（快速预览）",
    "iopaint_inpainted": "已去除",
    "copied": "未改动（复制原图）",
    "skipped_no_detection": "跳过（未检出）",
}

_DETECT_ZH = {
    "styles": "水印样式",
    "both": "样式+模型",
    "ai": "水印模型",
}

_BACKEND_ZH = {
    "iopaint": "高质量修补",
    "opencv": "快速预览",
    "none": "仅识别",
}


class ResultsPage(QWidget):
    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self._build_ui()
        self.refresh()

    def app_action_caps(self) -> set[str]:
        return {"export", "run"}

    def toolbar_action_labels(self) -> dict[str, str]:
        return {
            "export": "打开成品图",
            "run": "打开成品图",
        }

    def can_undo(self) -> bool:
        return False

    def can_redo(self) -> bool:
        return False

    def handle_app_action(self, action: str) -> bool:
        if action in {"export", "run"}:
            self._open_sub("output")
            return True
        return False

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_PAGE, SPACE_PAGE, SPACE_PAGE, SPACE_PAGE)
        root.setSpacing(SPACE_ROW)

        body = QHBoxLayout()
        body.setSpacing(SPACE_CARD_GAP)

        left = SectionPanel("历史记录")
        self.job_count = CountBadge("0")
        left.add_header_widget(self.job_count)
        self.refresh_btn = IconToolButton(icon_refresh(), "刷新记录列表", size=28)
        self.refresh_btn.clicked.connect(self.refresh)
        left.add_header_widget(self.refresh_btn)
        self.job_list = QListWidget()
        self.job_list.setToolTip("选择记录查看详情")
        self.job_list.currentItemChanged.connect(self._on_select)
        left.body.addWidget(self.job_list, 1)
        body.addWidget(left, 2)

        right = SectionPanel("详情")
        self.detail = TextEdit()
        self.detail.setReadOnly(True)
        right.body.addWidget(self.detail, 1)
        btns = QHBoxLayout()
        btns.setSpacing(6)
        self.open_output_btn = IconToolButton(icon_open(), "打开输出图", size=34)
        self.open_debug_btn = IconToolButton(icon_debug(), "打开识别预览", size=34)
        self.open_masks_btn = IconToolButton(icon_mask(), "打开遮罩", size=34)
        self.open_job_btn = IconToolButton(icon_folder_open(), "打开处理文件夹", size=34)
        self.open_output_btn.clicked.connect(lambda: self._open_sub("output"))
        self.open_debug_btn.clicked.connect(lambda: self._open_sub("debug"))
        self.open_masks_btn.clicked.connect(lambda: self._open_sub("masks"))
        self.open_job_btn.clicked.connect(lambda: self._open_sub("."))
        for btn in (
            self.open_output_btn,
            self.open_debug_btn,
            self.open_masks_btn,
            self.open_job_btn,
        ):
            btns.addWidget(btn)
        btns.addStretch(1)
        right.body.addLayout(btns)
        body.addWidget(right, 3)
        root.addLayout(body, 1)

    def refresh(self) -> None:
        self.job_list.clear()
        jobs_dir = self.workspace.jobs_dir
        if not jobs_dir.exists():
            self.job_count.set_count(0)
            self._set_detail("暂无处理记录。\n完成批量去除后，记录会出现在这里。")
            return
        count = 0
        for path in sorted(jobs_dir.iterdir(), reverse=True):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(str(path))
            self.job_list.addItem(item)
            count += 1
        self.job_count.set_count(count)
        if count == 0:
            self._set_detail("暂无处理记录。\n完成批量去除后，记录会出现在这里。")

    def select_job(self, job_dir: Path) -> None:
        self.refresh()
        for i in range(self.job_list.count()):
            item = self.job_list.item(i)
            if Path(item.data(Qt.UserRole)) == job_dir:
                self.job_list.setCurrentItem(item)
                return

    def _on_select(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            self._set_detail("")
            return
        job_dir = Path(current.data(Qt.UserRole))
        self._set_detail(self._format_job_detail(job_dir))

    def _format_job_detail(self, job_dir: Path) -> str:
        lines = [
            f"记录编号：{job_dir.name}",
            f"文件夹：{job_dir}",
            "",
        ]
        report = job_dir / "report.json"
        if not report.exists():
            lines.append("尚无摘要。")
            return "\n".join(lines)

        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            lines.append(f"无法读取摘要：{error}")
            return "\n".join(lines)

        summary = data.get("summary", data) if isinstance(data, dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        backend = str(summary.get("backend") or "")
        detect = str(summary.get("detect_mode") or "")
        n_img = summary.get("image_count")
        n_hit = summary.get("detected")
        n_empty = summary.get("empty_mask")
        cancelled = summary.get("cancelled")

        lines.append(f"修补方式：{_BACKEND_ZH.get(backend, backend or '—')}")
        lines.append(f"怎么找水印：{_DETECT_ZH.get(detect, detect or '—')}")
        if n_img is not None:
            lines.append(f"图片数量：{n_img}")
        if n_hit is not None:
            lines.append(f"有识别结果：{n_hit} 张")
        if n_empty is not None:
            lines.append(f"未检出：{n_empty} 张")
        if cancelled:
            lines.append("状态：已中途停止")
        hint = str(summary.get("hint") or "").strip()
        if hint:
            lines.append(f"提示：{hint}")

        profiles = summary.get("profiles") or []
        if profiles:
            lines.append(f"使用样式：{', '.join(str(p) for p in profiles)}")

        images = data.get("images") or []
        if images:
            lines.append("")
            lines.append(f"各图（共 {len(images)}）：")
            for item in images[:40]:
                if not isinstance(item, dict):
                    continue
                name = Path(str(item.get("image") or "")).name or "（未知）"
                action = _ACTION_ZH.get(str(item.get("action") or ""), str(item.get("action") or "—"))
                det = item.get("detections") or []
                n_det = len(det) if isinstance(det, list) else 0
                if n_det > 0:
                    lines.append(f"  · {name}：{action}（{n_det} 处）")
                else:
                    lines.append(f"  · {name}：{action}")
            if len(images) > 40:
                lines.append(f"  … 另有 {len(images) - 40} 张")

        return "\n".join(lines)

    def _set_detail(self, text: str) -> None:
        if hasattr(self.detail, "setPlainText"):
            self.detail.setPlainText(text)
        else:
            self.detail.setText(text)

    def _open_sub(self, name: str) -> None:
        item = self.job_list.currentItem()
        if item is None:
            self._open_path(self.workspace.jobs_dir)
            return
        job_dir = Path(item.data(Qt.UserRole))
        path = job_dir if name == "." else job_dir / name
        self._open_path(path)

    @staticmethod
    def _open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
