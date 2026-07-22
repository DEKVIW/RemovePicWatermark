"""Launch the desktop GUI without installing the package."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOG = ROOT / "gui_launch.log"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_log(text: str) -> None:
    try:
        LOG.write_text(text, encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    try:
        from remove_pic_watermark.gui.app import main as gui_main

        gui_main()
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except Exception:
        detail = traceback.format_exc()
        _write_log(detail)
        try:
            # Best-effort message box even if Fluent failed
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "一览清图",
                "程序未能启动。\n\n"
                f"详情已写入：\n{LOG}\n\n{detail[-1200:]}",
            )
        except Exception:
            print(detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
