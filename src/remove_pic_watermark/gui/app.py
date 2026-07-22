"""GUI application entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..paths import app_root
from ..workspace import get_workspace


def _bootstrap_paths() -> Path:
    root = app_root()
    if not getattr(sys, "frozen", False):
        src_dir = root / "src"
        if src_dir.is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
    try:
        os.chdir(root)
    except OSError:
        pass
    return root


def main(workspace: Path | None = None) -> int:
    _bootstrap_paths()
    try:
        from ..stdio_fix import ensure_stdio

        ensure_stdio()
    except Exception:
        pass

    # Cheap six.moves repair without importing yolo/cv2/torch.
    try:
        from ..six_patch import patch_six_meta_path

        patch_six_meta_path()
    except Exception:
        pass

    frozen = bool(getattr(sys, "frozen", False))
    # Frozen: rth hook already warms YOLO before PySide6 when possible.
    # Do NOT re-import YOLO here — keeps first paint fast. Warm after show.

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    # High-DPI friendly
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    from .branding import APP_NAME, APP_NAME_ZH, APP_TAGLINE, APP_VERSION, app_icon
    from .main_window import MainWindow
    from .theme_style import apply_app_typography

    app = QApplication.instance() or QApplication(sys.argv)
    # Keep internal id English; display name empty so Windows title is NOT
    # "一览清图 0.2.13 - 一览清图" (windowTitle + " - " + displayName).
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationDisplayName("")
    app.setApplicationVersion(APP_VERSION)
    try:
        app.setProperty("appTagline", APP_TAGLINE)
        app.setProperty("appNameZh", APP_NAME_ZH)
    except Exception:
        pass

    try:
        from qfluentwidgets import setTheme, Theme

        setTheme(Theme.LIGHT)
    except ImportError:
        pass

    apply_app_typography(app)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    ws = get_workspace(workspace)
    window = MainWindow(ws)
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    window.raise_()
    window.activateWindow()

    # Warm YOLO after first paint (source + frozen). Does not block UI show.
    def _deferred_warm_yolo() -> None:
        try:
            from ..six_patch import patch_six_meta_path
            from ..detectors.yolo_watermark import warm_import_yolo

            patch_six_meta_path()
            warm_import_yolo()
        except Exception:
            pass

    QTimer.singleShot(200, _deferred_warm_yolo)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
