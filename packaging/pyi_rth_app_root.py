"""Runtime hook: cwd + stdio + freeze fixes + YOLO pre-import for frozen GUI.

Runs BEFORE PyInstaller's pyi_rth_pyside6 (custom hooks are listed first).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "write") and hasattr(stream, "flush"):
            continue
        try:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8", errors="replace"))
        except OSError:
            from io import StringIO

            setattr(sys, name, StringIO())


def _set_cwd() -> None:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path.cwd()
    try:
        os.chdir(root)
    except OSError:
        pass


def _yolo_freeze_env() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("YOLO_AUTOINSTALL", "0")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")


def _patch_six_meta_path() -> None:
    try:
        import six  # noqa: F401
    except Exception:
        return
    for finder in sys.meta_path:
        try:
            if type(finder).__name__ == "_SixMetaPathImporter" and not hasattr(finder, "_path"):
                finder._path = []  # type: ignore[attr-defined]
        except Exception:
            continue
    try:
        from six.moves import urllib  # noqa: F401
    except Exception:
        pass


def _patch_torch_numpy_files_on_disk() -> None:
    """Rewrite shipped _ufuncs.py (idempotent) before any torch._numpy import."""
    if not getattr(sys, "frozen", False):
        return
    try:
        # packaging/ is not always on path in freeze; load by file next to this hook
        hook_dir = Path(__file__).resolve().parent
        patch_path = hook_dir / "patch_torch_ufuncs.py"
        if not patch_path.is_file():
            # also try MEIPASS root
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                patch_path = Path(meipass) / "patch_torch_ufuncs.py"
        if patch_path.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("rpw_patch_torch_ufuncs", patch_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                patch_fn = getattr(mod, "patch_ufuncs_file", None)
            else:
                patch_fn = None
        else:
            patch_fn = None
    except Exception:
        patch_fn = None

    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    try:
        roots.append(Path(sys.executable).resolve().parent / "_internal")
    except Exception:
        pass
    for root in roots:
        ufuncs = root / "torch" / "_numpy" / "_ufuncs.py"
        if not ufuncs.is_file():
            continue
        try:
            if patch_fn is not None:
                patch_fn(ufuncs)
            else:
                # Minimal inline fallback: only if still has broken loop
                text = ufuncs.read_text(encoding="utf-8")
                if "for name in _binary" in text and "RPW_ATTACH" not in text:
                    # Cannot fully restore without TAIL; leave for build-time patch
                    pass
        except Exception:
            continue


def _warm_import_yolo_early() -> None:
    if not getattr(sys, "frozen", False):
        return
    try:
        from remove_pic_watermark.detectors.yolo_watermark import warm_import_yolo

        warm_import_yolo()
        return
    except Exception:
        pass
    try:
        _patch_six_meta_path()
        from ultralytics.models.yolo.model import YOLO  # noqa: F401
    except Exception:
        pass


_ensure_stdio()
_set_cwd()
_yolo_freeze_env()
_patch_six_meta_path()
_patch_torch_numpy_files_on_disk()
_warm_import_yolo_early()
