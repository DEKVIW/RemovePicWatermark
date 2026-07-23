"""Watermark profile library + ROI capture workbench.

Layout (canvas-first):
  ┌ library ┬ create: name row + canvas  OR  detail: dual canvas ┐

Detail: left editable sample · right recognition template preview.
Zoom / pan / undo / brush / save / open come from main chrome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...device_info import probe_cuda
from ...image_io import read_image
from ...profiles.models import MatchStrategy
from ...services.profile_service import ProfileService, RoiNorm
from ...workspace import Workspace
from ..theme_style import SPACE_PAGE, SPACE_ROW
from ..widgets.icon_tool_button import IconToolButton
from ..widgets.image_canvas import CanvasTool, ImageCanvas
from ..widgets.page_chrome import CountBadge
from ..widgets.profile_card import ProfileCard
from ..widgets.tool_icons import (
    icon_ai,
    icon_back,
    icon_delete,
    icon_reload,
)

try:
    from qfluentwidgets import BodyLabel, LineEdit, SubtitleLabel
except ImportError:  # pragma: no cover
    from PySide6.QtWidgets import QLabel as BodyLabel
    from PySide6.QtWidgets import QLabel as SubtitleLabel
    from PySide6.QtWidgets import QLineEdit as LineEdit


class ProfilesPage(QWidget):
    """Compact structured layout: library rail + canvas-first create/detail."""

    profiles_changed = Signal()
    status_message = Signal(str)
    chrome_state_changed = Signal()

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.service = ProfileService(workspace)
        self._cards: dict[str, ProfileCard] = {}
        self._selected_id: str | None = None
        self._detail_dirty = False
        self._detail_loading = False
        self._device_pref = "auto"
        self._probe = None  # lazy — avoid torch import at app start
        self.setAcceptDrops(True)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_PAGE, SPACE_PAGE, SPACE_PAGE, SPACE_PAGE)
        root.setSpacing(SPACE_ROW)

        body = QHBoxLayout()
        body.setSpacing(SPACE_ROW)
        root.addLayout(body, 1)

        body.addWidget(self._build_library(), 0)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_create_panel())
        self.stack.addWidget(self._build_detail_panel())
        self.stack.currentChanged.connect(lambda _i: self.chrome_state_changed.emit())
        body.addWidget(self.stack, 1)
        self.stack.setCurrentIndex(0)

    def _build_library(self) -> QWidget:
        library = QFrame()
        library.setObjectName("libraryPanel")
        library.setFixedWidth(240)
        lib = QVBoxLayout(library)
        lib.setContentsMargins(12, 12, 12, 12)
        lib.setSpacing(8)

        head = QHBoxLayout()
        lib_title = BodyLabel("样式库")
        lib_title.setProperty("role", "section")
        self.count_label = CountBadge("0")
        head.addWidget(lib_title)
        head.addStretch(1)
        head.addWidget(self.count_label)
        lib.addLayout(head)

        self.library_stack = QStackedWidget()
        empty = QWidget()
        empty_l = QVBoxLayout(empty)
        empty_l.setContentsMargins(12, 24, 12, 24)
        empty_l.setSpacing(8)
        empty_l.addStretch(1)
        empty_icon = BodyLabel("◇")
        empty_icon.setAlignment(Qt.AlignCenter)
        empty_icon.setStyleSheet("font-size: 28px; color: #98a2b3;")
        empty_title = BodyLabel("暂无样式")
        empty_title.setProperty("role", "section")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_l.addWidget(empty_icon)
        empty_l.addWidget(empty_title)
        empty_l.addStretch(2)
        self.library_stack.addWidget(empty)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setObjectName("libraryScroll")
        self.card_host = QWidget()
        self.card_layout = QVBoxLayout(self.card_host)
        self.card_layout.setContentsMargins(0, 0, 2, 0)
        self.card_layout.setSpacing(6)
        self.card_layout.addStretch(1)
        self.scroll.setWidget(self.card_host)
        self.library_stack.addWidget(self.scroll)
        lib.addWidget(self.library_stack, 1)
        return library

    def _build_create_panel(self) -> QWidget:
        """Canvas-first create: compact name row + full-stage canvas (no side inspector)."""
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Compact name field — not a full inspector rail
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_lab = BodyLabel("名称")
        name_lab.setProperty("role", "caption")
        name_row.addWidget(name_lab)
        self.name_edit = LineEdit()
        self.name_edit.setPlaceholderText("样式名称")
        self.name_edit.setClearButtonEnabled(True)
        name_row.addWidget(self.name_edit, 1)
        outer.addLayout(name_row)

        stage = QFrame()
        stage.setObjectName("stagePanel")
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.canvas = ImageCanvas()
        self.canvas.setMinimumHeight(280)
        self.canvas.set_tool(CanvasTool.RECT)
        self.canvas.set_brush_radius(18)
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.canvas.mask_changed.connect(self._on_mask_changed)
        self.canvas.tool_hint.connect(self._on_tool_hint)
        self.canvas.empty_clicked.connect(self._import_sample)
        self.canvas.history_changed.connect(self._on_history_changed)
        self.canvas.tool_changed.connect(self._on_canvas_tool_changed)
        layout.addWidget(self.canvas, 1)

        outer.addWidget(stage, 1)
        # Save via main toolbar; import via main toolbar / drag / empty click
        return panel

    def _build_detail_panel(self) -> QWidget:
        """Inline dual-canvas: left edit sample · right recognition mask preview."""
        panel = QFrame()
        panel.setObjectName("workPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(4)
        self.detail_title = SubtitleLabel("样式详情")
        self.detail_title.setProperty("role", "pageTitle")
        head.addWidget(self.detail_title, 1)

        # Page-local: back / delete / reextract / AI.
        # Save template → main toolbar 保存; new sample → main toolbar 打开
        self.detail_back_btn = IconToolButton(icon_back(), "返回标注", size=32)
        self.detail_back_btn.clicked.connect(self._back_to_create)
        self.detail_delete_btn = IconToolButton(icon_delete(), "删除样式", size=32)
        self.detail_delete_btn.clicked.connect(self._delete_selected)
        self.detail_reextract_btn = IconToolButton(icon_reload(), "重新自动提取", size=32)
        self.detail_reextract_btn.clicked.connect(self._detail_reextract)
        self.detail_ai_btn = IconToolButton(icon_ai(), "AI 抠图", size=32)
        self.detail_ai_btn.clicked.connect(self._detail_ai_matting)

        for b in (
            self.detail_back_btn,
            self.detail_delete_btn,
            self.detail_reextract_btn,
            self.detail_ai_btn,
        ):
            head.addWidget(b)
        layout.addLayout(head)

        # Dual stage
        dual = QHBoxLayout()
        dual.setSpacing(8)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_cap = BodyLabel("样例")
        left_cap.setProperty("role", "section")
        left_col.addWidget(left_cap)
        self.detail_edit_canvas = ImageCanvas()
        self.detail_edit_canvas.setMinimumHeight(220)
        self.detail_edit_canvas.set_tool(CanvasTool.PAINT)
        self.detail_edit_canvas.set_brush_radius(8)
        self.detail_edit_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.detail_edit_canvas.mask_changed.connect(self._on_detail_mask_changed)
        self.detail_edit_canvas.history_changed.connect(self._on_history_changed)
        self.detail_edit_canvas.tool_changed.connect(self._on_detail_tool_changed)
        self.detail_edit_canvas.tool_hint.connect(self._on_tool_hint)
        left_col.addWidget(self.detail_edit_canvas, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        right_cap = BodyLabel("识别模板")
        right_cap.setProperty("role", "section")
        right_col.addWidget(right_cap)
        self.detail_preview_canvas = ImageCanvas()
        self.detail_preview_canvas.setMinimumHeight(220)
        self.detail_preview_canvas.set_tool(CanvasTool.PAN)
        self.detail_preview_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        right_col.addWidget(self.detail_preview_canvas, 1)

        dual.addLayout(left_col, 3)
        dual.addLayout(right_col, 2)
        layout.addLayout(dual, 1)

        # Defer AI status probe until user opens detail (keeps cold start light)
        return panel

    # ----- library -----

    def refresh(self) -> None:
        # Empty library by default — do not auto-seed test/builtin styles.
        while self.card_layout.count():
            item = self.card_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()

        profiles = self.service.list_profiles()
        self.count_label.set_count(len(profiles))
        if len(profiles) == 0:
            self.library_stack.setCurrentIndex(0)
        else:
            self.library_stack.setCurrentIndex(1)

        for profile in profiles:
            card = ProfileCard(profile, self.service.store.profile_dir(profile.id), self.card_host)
            card.selected.connect(self._on_card_selected)
            card.enable_toggled.connect(self._on_enable_toggled)
            card.delete_clicked.connect(self._on_delete)
            self.card_layout.addWidget(card)
            self._cards[profile.id] = card
            if profile.id == self._selected_id:
                card.set_selected(True)
        self.card_layout.addStretch(1)
        self.profiles_changed.emit()

    def _on_card_selected(self, profile_id: str) -> None:
        if self._detail_dirty and self._selected_id and self._selected_id != profile_id:
            reply = QMessageBox.question(
                self,
                "未保存",
                "当前模板有未保存修改，是否放弃？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                # restore selection highlight
                for pid, card in self._cards.items():
                    card.set_selected(pid == self._selected_id)
                return
        self._selected_id = profile_id
        for pid, card in self._cards.items():
            card.set_selected(pid == profile_id)
        self._show_detail(profile_id)

    def _show_detail(self, profile_id: str) -> None:
        try:
            profile = self.service.get(profile_id)
        except FileNotFoundError:
            return
        directory = self.service.store.profile_dir(profile_id)
        self.detail_title.setText(profile.name)
        self._detail_dirty = False
        self._load_detail_canvases(directory)
        self.stack.setCurrentIndex(1)
        self._refresh_ai_btn()
        self.chrome_state_changed.emit()
        self.detail_edit_canvas.setFocus()

    def _load_detail_canvases(self, directory: Path) -> None:
        sample = directory / "sample_crop.png"
        mask_path = directory / "template_mask.png"
        self._detail_loading = True
        try:
            if not sample.is_file():
                self.detail_edit_canvas.clear_image()
                self.detail_preview_canvas.clear_image()
                self.status_message.emit("缺少 sample_crop.png")
                return
            try:
                self.detail_edit_canvas.load_path(sample, keep_view=False)
            except Exception as exc:  # noqa: BLE001
                self.status_message.emit(f"无法打开样例：{exc}")
                return
            if mask_path.is_file():
                try:
                    mask = read_image(mask_path)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    self.detail_edit_canvas.set_paint_mask_from_numpy(mask)
                except Exception:  # noqa: BLE001
                    pass
            self.detail_edit_canvas.set_tool(CanvasTool.PAINT)
            self.detail_edit_canvas.clear_annotation_history()
            self.detail_edit_canvas.clear_prompt_points()
            self.detail_edit_canvas.set_prompt_point_mode(False)
            self.detail_edit_canvas.reset_view()
            self._sync_detail_preview()
            self._detail_dirty = False
        finally:
            self._detail_loading = False

    def _mask_to_preview_qimage(self, mask: np.ndarray) -> QImage:
        """Binary mask → dark-bg white-fg preview image."""
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        h, w = m.shape[:2]
        on = m > 10
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 11  # B
        rgba[:, :, 1] = 18  # G
        rgba[:, :, 2] = 32  # R  # #0b1220
        rgba[:, :, 3] = 255
        rgba[on, 0] = 248
        rgba[on, 1] = 250
        rgba[on, 2] = 252
        return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

    def _sync_detail_preview(self) -> None:
        mask = self.detail_edit_canvas.export_paint_mask_numpy()
        if mask is None:
            # empty mask still show black canvas sized to source
            bgr = self.detail_edit_canvas.source_bgr_numpy()
            if bgr is None:
                self.detail_preview_canvas.clear_image()
                return
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        qimg = self._mask_to_preview_qimage(mask)
        self.detail_preview_canvas.load_qimage(qimg, keep_view=False)
        self.detail_preview_canvas.set_tool(CanvasTool.PAN)
        self.detail_preview_canvas.reset_view()

    def _on_detail_mask_changed(self) -> None:
        if self.stack.currentIndex() != 1 or self._detail_loading:
            return
        self._detail_dirty = True
        self._sync_detail_preview()
        self.chrome_state_changed.emit()

    def _on_detail_tool_changed(self, _tool: CanvasTool) -> None:
        self.chrome_state_changed.emit()

    def set_device_preference(self, pref: str) -> None:
        pref = str(pref).lower()
        if pref in {"cuda", "gpu"}:
            self._device_pref = "gpu"
        elif pref == "cpu":
            self._device_pref = "cpu"
        else:
            self._device_pref = "auto"
        # No CUDA probe at prefs apply (startup). AI btn refreshes on detail open.
        self.chrome_state_changed.emit()

    def current_device_preference(self) -> str:
        return self._device_pref

    def device_status_extra(self) -> str:
        # Avoid importing transformers/torch just for the title chip at startup
        if not self._in_detail() if hasattr(self, "_in_detail") else False:
            return ""
        try:
            from ...services.ai_matting import ai_matting_status

            st = ai_matting_status()
        except Exception:  # noqa: BLE001
            return ""
        if st.available:
            return "AI 抠图就绪" if st.local_ready else "AI 抠图（首次将下载）"
        return "AI 抠图未就绪"

    def _refresh_ai_btn(self) -> None:
        if not hasattr(self, "detail_ai_btn"):
            return
        try:
            from ...services.ai_matting import ai_matting_status

            st = ai_matting_status()
        except Exception:  # noqa: BLE001
            self.detail_ai_btn.setEnabled(False)
            self.detail_ai_btn.setToolTip("AI 抠图不可用")
            return
        self.detail_ai_btn.setEnabled(st.available)
        if st.available:
            tip = "AI 自动识别水印区域"
            if not st.local_ready:
                tip += "（首次使用将下载模型）"
            self.detail_ai_btn.setToolTip(tip)
        else:
            self.detail_ai_btn.setToolTip("AI 抠图不可用")

    def _detail_reextract(self) -> None:
        if not self._selected_id:
            return
        try:
            result = self.service.reextract_template_from_sample(self._selected_id)
            mask = result.get("mask") if isinstance(result, dict) else None
            if mask is None:
                directory = self.service.store.profile_dir(self._selected_id)
                self._load_detail_canvases(directory)
                return
            self.detail_edit_canvas.push_annotation_history()
            self.detail_edit_canvas.set_paint_mask_from_numpy(mask)
            self._detail_dirty = False  # reextract already wrote to disk
            self._sync_detail_preview()
            self.status_message.emit("已重新提取")
            self.chrome_state_changed.emit()
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "提取失败", str(exc))

    def _detail_ai_matting(self) -> None:
        if not self._selected_id:
            return
        bgr = self.detail_edit_canvas.source_bgr_numpy()
        if bgr is None:
            QMessageBox.warning(self, "无法运行", "没有样例图像")
            return
        self.detail_ai_btn.setEnabled(False)
        self.status_message.emit("AI 抠图…")
        QApplication.processEvents()

        def _prog(msg: str) -> None:
            self.status_message.emit(str(msg)[:120])
            QApplication.processEvents()

        try:
            from ...services.ai_matting import extract_mask_ai

            mask, stats = extract_mask_ai(
                bgr,
                dilate=1,
                progress=_prog,
                model_dir=self.workspace.birefnet_dir,
                device_preference=self._device_pref,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "AI 抠图失败", str(exc))
            self._refresh_ai_btn()
            return
        finally:
            self._refresh_ai_btn()

        self.detail_edit_canvas.push_annotation_history()
        self.detail_edit_canvas.set_paint_mask_from_numpy(mask)
        self.detail_edit_canvas.set_tool(CanvasTool.PAINT)
        self._detail_dirty = True
        self._sync_detail_preview()
        self.status_message.emit("AI 抠图完成")
        self.chrome_state_changed.emit()

    def _detail_save_template(self) -> None:
        if not self._selected_id:
            return
        mask = self.detail_edit_canvas.export_paint_mask_numpy()
        if mask is None or int(np.count_nonzero(mask)) < 8:
            QMessageBox.information(self, "提示", "模板为空")
            return
        try:
            self.service.update_template_mask(self._selected_id, mask)
            self._detail_dirty = False
            self._sync_detail_preview()
            self.status_message.emit("识别模板已保存")
            self.refresh()
            self.profiles_changed.emit()
            self.chrome_state_changed.emit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))

    def _on_enable_toggled(self, profile_id: str, enabled: bool) -> None:
        self.service.set_enabled(profile_id, enabled)
        self.status_message.emit(f"{'已启用' if enabled else '已停用'}：{profile_id}")
        self.profiles_changed.emit()
        if self._selected_id == profile_id and self.stack.currentIndex() == 1:
            try:
                profile = self.service.get(profile_id)
                self.detail_title.setText(profile.name)
            except Exception:  # noqa: BLE001
                pass

    def _delete_selected(self) -> None:
        if self._selected_id:
            self._on_delete(self._selected_id)

    def _on_delete(self, profile_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "删除样式",
            "确定删除该样式？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Release canvases first — deleting files while paint layers hold paths can freeze/crash
        was_selected = self._selected_id == profile_id
        if was_selected:
            self._detail_loading = True
            try:
                self.detail_edit_canvas.clear_image()
                self.detail_preview_canvas.clear_image()
            except Exception:  # noqa: BLE001
                pass
            self._selected_id = None
            self._detail_dirty = False
            self.stack.setCurrentIndex(0)
            self._detail_loading = False
            QApplication.processEvents()
        try:
            self.service.delete(profile_id)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "删除失败", str(error))
            self.refresh()
            return
        try:
            self.refresh()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "刷新失败", str(error))
        self.status_message.emit("已删除样式")
        self.chrome_state_changed.emit()

    def _back_to_create(self) -> None:
        if self._detail_dirty:
            reply = QMessageBox.question(
                self,
                "未保存",
                "当前模板有未保存修改，是否放弃？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._selected_id = None
        self._detail_dirty = False
        for card in self._cards.values():
            card.set_selected(False)
        self.stack.setCurrentIndex(0)
        self.status_message.emit("标注水印样式")
        self.chrome_state_changed.emit()

    def _start_create(self) -> None:
        """Reset create workbench and prompt for a sample."""
        if self._detail_dirty:
            reply = QMessageBox.question(
                self,
                "未保存",
                "当前模板有未保存修改，是否放弃？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._selected_id = None
        self._detail_dirty = False
        for card in self._cards.values():
            card.set_selected(False)
        self.stack.setCurrentIndex(0)
        if hasattr(self, "name_edit"):
            self.name_edit.clear()
        if hasattr(self, "canvas"):
            self.canvas.clear_image()
        self._set_tool(CanvasTool.RECT)
        self.status_message.emit("请导入样例图并标注水印")
        self.chrome_state_changed.emit()
        self._import_sample()

    # ----- create flow -----

    def _in_detail(self) -> bool:
        return self.stack.currentIndex() == 1

    def _on_brush(self, value: int) -> None:
        self.canvas.set_brush_radius(int(value))
        if hasattr(self, "detail_edit_canvas"):
            self.detail_edit_canvas.set_brush_radius(int(value))

    def _set_tool(self, tool: CanvasTool) -> None:
        if self._in_detail():
            # Detail: paint/erase/pan only (no rect on template edit)
            if tool == CanvasTool.RECT:
                tool = CanvasTool.PAINT
            self.detail_edit_canvas.set_tool(tool)
        else:
            self.canvas.set_tool(tool)
        self.chrome_state_changed.emit()

    def _on_canvas_tool_changed(self, tool: CanvasTool) -> None:
        self.chrome_state_changed.emit()

    def canvas_tool_caps(self) -> dict:
        if self._in_detail():
            return {
                "rect": False,
                "paint": True,
                "erase": True,
                "pan": True,
                "clear": True,
                "brush": True,
            }
        return {
            "rect": True,
            "paint": True,
            "erase": True,
            "pan": True,
            "clear": True,
            "brush": True,
        }

    def current_canvas_tool(self) -> CanvasTool:
        if self._in_detail():
            return self.detail_edit_canvas.tool
        return self.canvas.tool

    def canvas_brush_radius(self) -> int:
        if self._in_detail():
            return self.detail_edit_canvas.brush_radius()
        return self.canvas.brush_radius()

    def set_canvas_tool(self, tool: CanvasTool) -> None:
        self._set_tool(tool)

    def set_canvas_brush(self, radius: int) -> None:
        self._on_brush(radius)

    def _on_tool_hint(self, text: str) -> None:
        if text:
            self.status_message.emit(text)

    def _on_history_changed(self) -> None:
        self.chrome_state_changed.emit()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            p = Path(local)
            if p.is_file() and p.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                self._load_sample_path(p)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _import_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择含有水印的样例图片",
            str(self.workspace.root),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self._load_sample_path(Path(path))

    def _load_sample_path(self, path: Path) -> None:
        try:
            self.canvas.load_path(path)
            self.canvas.set_tool(self.canvas.tool)
            if not self.name_edit.text().strip():
                self.name_edit.setText(path.stem)
            self.stack.setCurrentIndex(0)
            self.status_message.emit(f"已加载：{path.name}")
            self.chrome_state_changed.emit()
        except ValueError as error:
            QMessageBox.warning(self, "无法打开图片", str(error))

    def _clear_roi(self) -> None:
        if self._in_detail():
            self.detail_edit_canvas.clear_roi()
            self._detail_dirty = True
            self._sync_detail_preview()
        else:
            self.canvas.clear_roi()
        self.chrome_state_changed.emit()

    def app_action_caps(self) -> set[str]:
        caps = {
            "open",
            "export",
            "undo",
            "redo",
            "clear",
            "zoom_in",
            "zoom_out",
            "zoom_fit",
            "zoom_1x",
            "run",
        }
        return caps

    def toolbar_action_labels(self) -> dict[str, str]:
        if self._in_detail():
            return {
                "open": "导入新样例",
                "export": "保存识别模板",
                "run": "保存识别模板",
            }
        return {
            "open": "导入样例",
            "export": "保存样式",
            "run": "保存样式",
        }

    def active_canvas(self) -> ImageCanvas:
        if self._in_detail():
            return self.detail_edit_canvas
        return self.canvas

    def can_undo(self) -> bool:
        return self.active_canvas().can_undo_annotation()

    def can_redo(self) -> bool:
        return self.active_canvas().can_redo_annotation()

    def handle_app_action(self, action: str) -> bool:
        canvas = self.active_canvas()
        if action == "open":
            if self._in_detail():
                self._start_create()
            else:
                self._import_sample()
            return True
        if action == "export" or action == "run":
            if self._in_detail():
                self._detail_save_template()
            else:
                self._save_profile()
            return True
        if action == "undo":
            ok = canvas.undo_annotation()
            if ok:
                if self._in_detail():
                    self._detail_dirty = True
                    self._sync_detail_preview()
                self.status_message.emit("已撤销")
                self.chrome_state_changed.emit()
            return bool(ok)
        if action == "redo":
            ok = canvas.redo_annotation()
            if ok:
                if self._in_detail():
                    self._detail_dirty = True
                    self._sync_detail_preview()
                self.chrome_state_changed.emit()
            return bool(ok)
        if action == "clear":
            self._clear_roi()
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

    def _on_roi_changed(self, roi: RoiNorm | None) -> None:
        if self.canvas.tool == CanvasTool.PAINT:
            return
        if roi is not None:
            w = max(0.0, roi.right - roi.left)
            h = max(0.0, roi.bottom - roi.top)
            ang = self.canvas.roi_angle_deg
            ang_txt = f" · {ang:.0f}°" if abs(ang) > 0.5 else ""
            self.status_message.emit(f"已框选 {w:.0%}×{h:.0%}{ang_txt}")

    def _on_mask_changed(self) -> None:
        if self.canvas.tool in {CanvasTool.PAINT, CanvasTool.ERASE} and self.canvas.has_selection():
            self.status_message.emit("已涂抹")

    def _save_profile(self) -> None:
        if not self.canvas.has_image() or self.canvas.source_path is None:
            QMessageBox.information(self, "提示", "请先导入样例图片。")
            return
        name = self.name_edit.text().strip() or self.canvas.source_path.stem
        # Locate mode is chosen on batch page (固定/附近/全图). Store neutral AUTO.
        strategy = MatchStrategy.AUTO

        self.status_message.emit("正在保存…")
        try:
            if self.canvas.tool == CanvasTool.PAINT:
                mask = self.canvas.export_paint_mask_numpy()
                if mask is None:
                    raise ValueError("请先涂抹水印区域。")
                profile, _directory = self.service.create_from_paint_mask(
                    name=name,
                    image_path=self.canvas.source_path,
                    mask_gray=mask,
                    match_strategy=strategy,
                )
                self.status_message.emit(f"已保存「{profile.name}」")
            else:
                if self.canvas.roi_norm is None:
                    raise ValueError("请拖拽框选水印区域。")
                profile, build, _directory = self.service.create_from_roi(
                    name=name,
                    image_path=self.canvas.source_path,
                    roi=self.canvas.roi_norm,
                    description="",
                    match_strategy=strategy,
                    angle_deg=self.canvas.roi_angle_deg,
                )
                self.status_message.emit(f"已保存「{profile.name}」")
            self.refresh()
            self._on_card_selected(profile.id)
            self.profiles_changed.emit()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(error))
