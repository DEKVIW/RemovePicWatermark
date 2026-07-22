from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

from remove_pic_watermark import __version__ as _PKG_VERSION

# Internal / package identity (exe, folder, import path) — do not change casually
APP_NAME = "RemovePicWatermark"

# Product display name (window, about, OS app list)
APP_NAME_ZH = "一览清图"
APP_TAGLINE = "批量清除图片水印，智能检测"
APP_VERSION = str(_PKG_VERSION)
# Single title only — do not rely on Qt displayName (Windows appends " - name")
APP_WINDOW_TITLE = f"{APP_NAME_ZH} {APP_VERSION}"

# Author site — shown in 关于 / 帮助, open in browser
AUTHOR_BLOG_URL = "https://blog.yilanapp.com/"
AUTHOR_BLOG_LABEL = "作者博客"

# Plain text fallback (no HTML)
APP_ABOUT = (
    f"{APP_NAME_ZH}  {APP_VERSION}\n"
    f"{APP_TAGLINE}\n\n"
    "· 批量处理多张图片，自动定位并去除水印\n"
    "· 建立水印样式，同类水印可反复复用\n"
    "· 单张精修：框选或涂抹区域，局部补漏\n"
    "· 可选高质量修补，或快速预览\n\n"
    f"{AUTHOR_BLOG_LABEL}：{AUTHOR_BLOG_URL}"
)

# Rich text for clickable link in 关于 dialog
APP_ABOUT_HTML = (
    f"<p style='margin:0 0 4px 0;'><b>{APP_NAME_ZH}</b>  {APP_VERSION}</p>"
    f"<p style='margin:0 0 10px 0; color:#475467;'>{APP_TAGLINE}</p>"
    "<ul style='margin:0 0 12px 18px; padding:0;'>"
    "<li>批量处理多张图片，自动定位并去除水印</li>"
    "<li>建立水印样式，同类水印可反复复用</li>"
    "<li>单张精修：框选或涂抹区域，局部补漏</li>"
    "<li>可选高质量修补，或快速预览</li>"
    "</ul>"
    f"<p style='margin:0;'>{AUTHOR_BLOG_LABEL}："
    f"<a href='{AUTHOR_BLOG_URL}'>{AUTHOR_BLOG_URL.rstrip('/')}</a></p>"
)


def _candidate_icon_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots: list[Path] = [
        here / "resources",
        Path.cwd() / "packaging",
        Path.cwd(),
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        roots = [
            exe_dir,
            meipass,
            meipass / "remove_pic_watermark" / "gui" / "resources",
            meipass / "gui" / "resources",
            *roots,
        ]
    paths: list[Path] = []
    for root in roots:
        paths.append(root / "app.ico")
        paths.append(root / "app.png")
    return paths


def icon_path() -> Path | None:
    for path in _candidate_icon_paths():
        if path.is_file():
            return path
    return None


def app_icon() -> QIcon:
    path = icon_path()
    if path is None:
        return QIcon()
    return QIcon(str(path))
