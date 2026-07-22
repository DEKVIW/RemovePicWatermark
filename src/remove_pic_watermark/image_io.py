from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SKIP_DIR_NAMES = {
    "assets",
    "configs",
    "debug",
    "masks",
    "output",
    "src",
    "workspace",
    "profiles",
    "jobs",
    "models",
    "__pycache__",
}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Unable to encode image as {extension}: {path}")
    encoded.tofile(str(path))


def iter_image_files(path: Path, recursive: bool = True) -> list[Path]:
    """List image files under path.

    Nested dirs named masks/debug/output/jobs/… are skipped *relative to the
    scan root*, so a deliberate input folder such as workspace/jobs/_staging
    is still scanned correctly (previously absolute paths containing "jobs"
    were skipped entirely — GUI batch jobs appeared to do nothing).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    root = path.resolve()
    pattern = "**/*" if recursive else "*"
    images: list[Path] = []
    for candidate in path.glob(pattern):
        if not candidate.is_file():
            continue
        if candidate.name.startswith("debug_"):
            continue
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel_parts = candidate.resolve().relative_to(root).parts
        except ValueError:
            rel_parts = candidate.parts
        # Only skip nested generated folders under the scan root, not the root name itself
        if any(part.lower() in SKIP_DIR_NAMES for part in rel_parts[:-1]):
            continue
        images.append(candidate)
    return sorted(images)
