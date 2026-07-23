from __future__ import annotations

from typing import Any

from ...workspace import Workspace, get_workspace, load_json, save_json

DEFAULT_PREFS: dict[str, Any] = {
    "window_geometry": None,
    "last_input_dir": "",
    "last_sample_dir": "",
    "backend": "iopaint",
    # UI preference: auto | cpu | gpu  (resolved to cuda/cpu at run time)
    "device_preference": "auto",
    # legacy key kept for older prefs files
    "iopaint_device": "auto",
    # pin | follow | search  (legacy auto → follow)
    "match_strategy": "follow",
    # styles | both | ai  → 水印样式 / 样式+模型 / 水印模型
    "detect_mode": "styles",
    "selected_profiles": [],
    "nav_mode": "profiles",
}


def load_prefs(workspace: Workspace | None = None) -> dict[str, Any]:
    ws = workspace or get_workspace()
    data = load_json(ws.prefs_path, DEFAULT_PREFS)
    merged = dict(DEFAULT_PREFS)
    merged.update(data)
    return merged


def save_prefs(prefs: dict[str, Any], workspace: Workspace | None = None) -> None:
    ws = workspace or get_workspace()
    save_json(ws.prefs_path, prefs)
