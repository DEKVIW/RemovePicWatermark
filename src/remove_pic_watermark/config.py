from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import app_root, resource_root


def default_config_path() -> Path:
    return resource_root() / "configs" / "default.json"


def load_config(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    config_path = path or default_config_path()
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file), config_path


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    config_relative_path = (config_path.parent / path).resolve()
    if config_relative_path.exists():
        return config_relative_path

    # bundled resources first, then writable app root
    for root in (resource_root(), app_root()):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (resource_root() / path).resolve()


# Back-compat aliases used by older imports
PROJECT_ROOT = app_root()
DEFAULT_CONFIG_PATH = default_config_path()
