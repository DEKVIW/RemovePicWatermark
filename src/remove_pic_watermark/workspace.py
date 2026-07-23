"""Project workspace layout — user data separate from package code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import app_root, resource_root

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Writable project/exe directory (not PyInstaller _MEIPASS)
PROJECT_ROOT = app_root()
RESOURCE_ROOT = resource_root()

WORKSPACE_DIRNAME = "workspace"
LEGACY_DATA_DIRNAME = "data"


@dataclass(frozen=True)
class Workspace:
    """Resolved paths for profiles, jobs, models, and prefs."""

    root: Path

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def yolo_dir(self) -> Path:
        """YOLO weights directory (watermark.pt)."""
        return self.models_dir / "yolo"

    @property
    def sam_dir(self) -> Path:
        """SAM / EdgeSAM weights (edge_sam.pth)."""
        return self.models_dir / "sam"

    @property
    def birefnet_dir(self) -> Path:
        """BiRefNet weights (HF snapshot under models/birefnet)."""
        return self.models_dir / "birefnet"

    @property
    def yolo_train_dir(self) -> Path:
        """Training workspace: dataset + runs."""
        return self.root / "yolo_train"

    @property
    def prefs_path(self) -> Path:
        return self.root / "gui_prefs.json"

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    def ensure(self) -> "Workspace":
        for path in (
            self.root,
            self.profiles_dir,
            self.jobs_dir,
            self.models_dir,
            self.yolo_dir,
            self.sam_dir,
            self.birefnet_dir,
            self.yolo_train_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def profile_dir(self, profile_id: str) -> Path:
        return self.profiles_dir / profile_id


def default_workspace_root(project_root: Path | None = None) -> Path:
    root = project_root or app_root()
    preferred = root / WORKSPACE_DIRNAME
    if preferred.exists():
        return preferred
    # Fresh installs use workspace/; keep legacy data/ only if already present without workspace.
    legacy = root / LEGACY_DATA_DIRNAME
    if legacy.exists() and not preferred.exists():
        # Prefer new layout; create workspace alongside legacy for migration.
        return preferred
    return preferred


def get_workspace(root: Path | None = None) -> Workspace:
    # Re-bind PROJECT_ROOT each call so frozen exe always uses app_root()
    global PROJECT_ROOT, RESOURCE_ROOT
    PROJECT_ROOT = app_root()
    RESOURCE_ROOT = resource_root()
    workspace_root = root if root is not None else default_workspace_root()
    return Workspace(root=workspace_root).ensure()


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str, fallback: str = "watermark") -> str:
    cleaned = []
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_", ".", "　"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback
