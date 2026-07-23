"""App root vs bundled resource root (dev + frozen)."""

from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Writable application root: project dir in dev, folder of exe when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # src/remove_pic_watermark/paths.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    """Read-only bundled resources (PyInstaller _MEIPASS or project root)."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return app_root()
