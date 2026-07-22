"""Write VERSION.txt and 使用说明.txt into a dist folder (UTF-8)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: write_dist_meta.py <dist_app_dir> <version>")
        return 2
    out = Path(sys.argv[1])
    version = sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    (out / "VERSION.txt").write_text(version + "\n", encoding="utf-8")
    readme_src = Path(__file__).resolve().parent / "user_README.txt"
    if readme_src.is_file():
        text = readme_src.read_text(encoding="utf-8")
        (out / "使用说明.txt").write_text(text, encoding="utf-8-sig")
        (out / "user_README.txt").write_text(text, encoding="utf-8")
    start = out / "start.bat"
    start.write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "start \"\" \"%~dp0RemovePicWatermark.exe\"\r\n",
        encoding="ascii",
    )
    # Ensure icon sits next to exe for taskbar / explorer fallbacks
    for name in ("app.ico", "app.png"):
        src = Path(__file__).resolve().parent / name
        if src.is_file():
            (out / name).write_bytes(src.read_bytes())
    print("meta written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
