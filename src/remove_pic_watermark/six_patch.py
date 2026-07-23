"""Tiny six.moves / PyInstaller meta_path fix — no cv2/torch/ultralytics imports.

Importing ``yolo_watermark`` just for this patch pulled OpenCV and slowed startup.
"""

from __future__ import annotations


def patch_six_meta_path() -> None:
    """Repair six.moves under PyInstaller / after Qt load.

    Without this, ultralytics → matplotlib → dateutil can fail with:
      AttributeError: '_SixMetaPathImporter' object has no attribute '_path'
    """
    import sys

    try:
        import six  # noqa: F401
    except Exception:  # noqa: BLE001
        return
    for finder in sys.meta_path:
        try:
            if type(finder).__name__ == "_SixMetaPathImporter" and not hasattr(
                finder, "_path"
            ):
                finder._path = []  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            continue
    try:
        from six.moves import urllib  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
