"""Visual tokens for the watermark workbench GUI.

Design system notes (Fluent-inspired light workbench):
- Surfaces: page bg (window) · white cards · slate toolbars
- Type: pageTitle 16 · section 13 · body 12 · caption 11
- Spacing: page 16 · card pad 12–14 · row 8
- Accent: blue for selected / primary; green for step done
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

FONT_PAGE_TITLE = 16
FONT_SECTION = 13
FONT_BODY = 12
FONT_CAPTION = 11
FONT_MONO = 11

SPACE_PAGE = 12
SPACE_CARD_V = 10
SPACE_CARD_H = 12
SPACE_CARD_GAP = 10
SPACE_ROW = 8
INSPECTOR_WIDTH = 280

COLOR_TEXT = "#101828"
COLOR_TEXT_SECONDARY = "#344054"
COLOR_TEXT_MUTED = "#667085"
COLOR_BORDER = "rgba(16, 24, 40, 0.08)"
COLOR_SURFACE = "#ffffff"
COLOR_SURFACE_MUTED = "#f8fafc"
COLOR_ACCENT = "#1570ef"


def apply_app_typography(app: QApplication) -> None:
    base = QFont()
    base.setPointSize(FONT_BODY)
    app.setFont(base)
    app.setStyleSheet(
        f"""
        QWidget {{
            font-size: {FONT_BODY}pt;
            color: {COLOR_TEXT};
        }}
        QLabel[role="pageTitle"] {{
            font-size: {FONT_PAGE_TITLE}pt;
            font-weight: 600;
            color: {COLOR_TEXT};
        }}
        QLabel[role="section"] {{
            font-size: {FONT_SECTION}pt;
            font-weight: 600;
            color: {COLOR_TEXT_SECONDARY};
        }}
        QLabel[role="caption"], QLabel[role="hint"] {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_MUTED};
        }}
        QStatusBar {{
            font-size: {FONT_CAPTION}pt;
            color: #475467;
        }}

        /* library + work surfaces */
        QFrame#libraryPanel, QFrame#workPanel {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        QFrame#toolbarBar, QFrame#actionBar {{
            background: {COLOR_SURFACE_MUTED};
            border: 1px solid rgba(16, 24, 40, 0.06);
            border-radius: 10px;
        }}
        QFrame#canvasToolBar {{
            background: rgba(248, 250, 252, 0.96);
            border: 1px solid rgba(16, 24, 40, 0.06);
            border-radius: 8px;
        }}
        QFrame#optionGroup {{
            background: {COLOR_SURFACE_MUTED};
            border: 1px solid rgba(16, 24, 40, 0.06);
            border-radius: 10px;
        }}
        QFrame#optionsBar {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        /* PS-style right inspector */
        QFrame#inspectorPanel {{
            background: transparent;
            border: none;
        }}
        QFrame#inspectorBody {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        QFrame#inspectorRail {{
            background: {COLOR_SURFACE_MUTED};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
        }}
        QPushButton#inspectorToggle {{
            background: #f2f4f7;
            border: 1px solid rgba(16, 24, 40, 0.08);
            border-radius: 6px;
            color: {COLOR_TEXT_SECONDARY};
            font-weight: 600;
            padding: 0;
        }}
        QPushButton#inspectorToggle:hover {{
            background: #e8eef9;
            color: {COLOR_ACCENT};
            border-color: rgba(21, 112, 239, 0.35);
        }}
        QFrame#stagePanel {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        QFrame#collapsibleLog {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        QLabel#countBadge {{
            font-size: {FONT_CAPTION}pt;
            font-weight: 600;
            color: {COLOR_TEXT_SECONDARY};
            background: #f2f4f7;
            border-radius: 999px;
            padding: 2px 8px;
            min-height: 18px;
        }}
        QLabel#deviceStatus {{
            font-size: {FONT_CAPTION}pt;
            color: #475467;
        }}
        QFrame#maskStrip QLabel#maskPreview {{
            background: #0f172a;
            border-radius: 8px;
            color: #94a3b8;
            padding: 4px;
        }}
        QPushButton#cardDeleteBtn, QToolButton#cardDeleteBtn {{
            background: transparent;
            color: #98a2b3;
            border: none;
            border-radius: 6px;
            padding: 2px;
            font-size: 14pt;
            font-weight: 600;
        }}
        QPushButton#cardDeleteBtn:hover, QToolButton#cardDeleteBtn:hover {{
            background: #fef3f2;
            color: #d92d20;
        }}
        QPushButton#toolToggle[checked="true"], QPushButton#toolToggle:checked {{
            background: #dbeafe;
            color: #1d4ed8;
            border: 1px solid rgba(37, 99, 235, 0.35);
            font-weight: 600;
        }}

        /* Icon tool buttons + left rail (Inpaint / PS style) */
        QToolButton#iconTool, QToolButton#railTool {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 4px;
            color: {COLOR_TEXT_SECONDARY};
        }}
        QToolButton#iconTool:hover:!checked, QToolButton#railTool:hover:!checked {{
            background: #eef2ff;
            border-color: rgba(37, 99, 235, 0.22);
        }}
        /* Only the active tool — not hover leftovers */
        QToolButton#iconTool:checked, QToolButton#railTool:checked {{
            background: #dbeafe;
            border: 1px solid rgba(37, 99, 235, 0.45);
        }}
        QToolButton#iconTool:!checked, QToolButton#railTool:!checked {{
            background: transparent;
            border: 1px solid transparent;
        }}
        QToolButton#iconTool:disabled, QToolButton#railTool:disabled {{
            background: transparent;
            border-color: transparent;
            opacity: 0.45;
        }}
        QFrame#toolRail {{
            background: {COLOR_SURFACE_MUTED};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        QLabel#railCaption {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_MUTED};
        }}
        QToolBar#mainToolBar {{
            background: {COLOR_SURFACE};
            border: none;
            border-bottom: 1px solid {COLOR_BORDER};
            spacing: 4px;
            padding: 4px 8px;
        }}
        QToolBar#mainToolBar::separator {{
            background: {COLOR_BORDER};
            width: 1px;
            margin: 6px 6px;
        }}
        QLabel#toolbarBrushLabel {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_MUTED};
            min-width: 22px;
        }}
        QSlider#toolbarBrush {{
            max-height: 22px;
        }}
        QSlider#toolbarBrush::groove:horizontal {{
            height: 4px;
            background: {COLOR_BORDER};
            border-radius: 2px;
        }}
        QSlider#toolbarBrush::handle:horizontal {{
            width: 12px;
            margin: -5px 0;
            background: #3b82f6;
            border-radius: 6px;
        }}

        /* profile cards */
        QFrame#profileCard {{
            background: {COLOR_SURFACE_MUTED};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
        }}
        QFrame#profileCard[selected="true"] {{
            background: #eff6ff;
            border: 1px solid rgba(37, 99, 235, 0.45);
        }}
        QFrame#profileCard:hover {{
            border: 1px solid rgba(37, 99, 235, 0.28);
        }}
        QLabel#profileThumb {{
            background: #0f172a;
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
            font-size: 10pt;
        }}
        QLabel#profileTitle {{
            font-size: {FONT_BODY}pt;
            font-weight: 600;
            color: {COLOR_TEXT};
        }}
        QLabel#profileMeta {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_MUTED};
        }}
        QCheckBox#profileEnable {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_SECONDARY};
            spacing: 4px;
        }}
        QWidget#imageCanvas {{
            background: #0b1220;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 10px;
        }}

        /* step chips */
        QLabel#stepChip {{
            font-size: {FONT_CAPTION}pt;
            color: {COLOR_TEXT_MUTED};
            background: #f2f4f7;
            border-radius: 999px;
            padding: 3px 8px;
        }}
        QLabel#stepChip[active="true"] {{
            color: #1d4ed8;
            background: #dbeafe;
            font-weight: 600;
        }}
        QLabel#stepChip[done="true"] {{
            color: #067647;
            background: #dcfae6;
        }}

        /* detail previews */
        QLabel#detailPreview, QLabel#detailTemplate {{
            background: #0b1220;
            border-radius: 12px;
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }}

        QScrollArea#libraryScroll {{
            background: transparent;
            border: none;
        }}
        QScrollArea#libraryScroll > QWidget > QWidget {{
            background: transparent;
        }}

        QGroupBox {{
            font-size: {FONT_SECTION}pt;
            font-weight: 600;
            margin-top: 8px;
            padding: 14px 12px 12px 12px;
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            background: {COLOR_SURFACE};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {COLOR_TEXT_SECONDARY};
        }}
        QListWidget {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            background: {COLOR_SURFACE};
            outline: none;
            padding: 6px;
        }}
        QListWidget::item {{
            padding: 8px 10px;
            border-radius: 8px;
            margin: 2px 0;
        }}
        QListWidget::item:selected {{
            background: rgba(21, 112, 239, 0.12);
            color: {COLOR_TEXT};
        }}
        QListWidget::item:hover {{
            background: rgba(21, 112, 239, 0.06);
        }}
        QPlainTextEdit {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            background: {COLOR_SURFACE};
            color: {COLOR_TEXT};
            padding: 8px;
        }}
        QProgressBar {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            background: #f2f4f7;
            text-align: center;
            min-height: 18px;
            color: {COLOR_TEXT_SECONDARY};
        }}
        QProgressBar::chunk {{
            background: {COLOR_ACCENT};
            border-radius: 5px;
        }}
        """
    )

