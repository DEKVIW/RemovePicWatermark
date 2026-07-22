"""PyInstaller entry for RemovePicWatermark GUI."""

from __future__ import annotations

import sys


def main() -> int:
    # Windowed freeze: fix None stdout/stderr before torch/diffusers import
    try:
        from remove_pic_watermark.stdio_fix import ensure_stdio

        ensure_stdio()
    except Exception:
        pass

    from remove_pic_watermark.gui.app import main as gui_main

    code = gui_main()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
