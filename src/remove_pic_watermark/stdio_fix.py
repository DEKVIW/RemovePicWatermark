"""Fix sys.stdout/stderr for frozen GUI (windowed) processes.

PyInstaller ``runw.exe`` sets stdout/stderr to None. Libraries such as
``diffusers`` configure logging with StreamHandler(sys.stderr) and crash:

    AttributeError: 'NoneType' object has no attribute 'flush'
"""

from __future__ import annotations

import os
import sys
from typing import TextIO, Text


_null_streams: list[TextIO] = []


def ensure_stdio() -> None:
    """Ensure stdout/stderr are writable objects with ``flush``."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "write") and hasattr(stream, "flush"):
            continue
        # Prefer real null device so C extensions that call fileno() may work
        try:
            replacement: TextIO = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
            _null_streams.append(replacement)
        except OSError:
            from io import StringIO

            replacement = StringIO()
        setattr(sys, name, replacement)

    # Some libs cache logging handlers at import time
    try:
        import logging

        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
            root.setLevel(logging.WARNING)
    except Exception:
        pass
