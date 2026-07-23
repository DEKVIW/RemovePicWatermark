"""Load / save watermark profiles under workspace/profiles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from ..config import default_config_path, load_config, resolve_config_path
from ..paths import app_root
from ..workspace import Workspace, get_workspace, load_json, save_json, slugify
from .models import Profile, ProfileKind


class ProfileStore:
    def __init__(self, workspace: Workspace | None = None) -> None:
        self.workspace = workspace or get_workspace()

    def list_profiles(self) -> list[Profile]:
        profiles: list[Profile] = []
        for path in sorted(self.workspace.profiles_dir.glob("*/profile.json")):
            try:
                profiles.append(self._read(path.parent))
            except (KeyError, ValueError, OSError):
                continue
        return profiles

    def get(self, profile_id: str) -> Profile:
        path = self.workspace.profile_dir(profile_id)
        if not (path / "profile.json").exists():
            raise FileNotFoundError(f"Profile not found: {profile_id}")
        return self._read(path)

    def exists(self, profile_id: str) -> bool:
        return (self.workspace.profile_dir(profile_id) / "profile.json").exists()

    def profile_dir(self, profile_id: str) -> Path:
        return self.workspace.profile_dir(profile_id)

    def save(self, profile: Profile) -> Path:
        directory = self.workspace.profile_dir(profile.id)
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "profile.json", profile.to_dict())
        return directory

    def delete(self, profile_id: str) -> None:
        directory = self.workspace.profile_dir(profile_id)
        if directory.exists():
            shutil.rmtree(directory)

    def set_enabled(self, profile_id: str, enabled: bool) -> Profile:
        profile = self.get(profile_id)
        profile.enabled = enabled
        self.save(profile)
        return profile

    def enabled_profiles(self, profile_ids: Iterable[str] | None = None) -> list[Profile]:
        if profile_ids is None:
            return [item for item in self.list_profiles() if item.enabled]
        selected = set(profile_ids)
        return [item for item in self.list_profiles() if item.id in selected and item.enabled]

    def allocate_id(self, name: str) -> str:
        base = slugify(name)
        candidate = base
        index = 2
        while self.exists(candidate):
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _read(self, directory: Path) -> Profile:
        data = load_json(directory / "profile.json")
        if "id" not in data:
            data["id"] = directory.name
        return Profile.from_dict(data)


def bootstrap_builtin_profiles(
    workspace: Workspace | None = None,
    config_path: Path | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Import legacy configs/default.json + assets templates into workspace profiles."""
    store = ProfileStore(workspace)
    config, resolved = load_config(config_path or default_config_path())
    created: list[str] = []

    for item in config.get("fixed_watermarks", []):
        profile_id = slugify(item.get("label", "fixed_box"))
        if store.exists(profile_id) and not overwrite:
            continue
        profile = Profile(
            id=profile_id,
            name=item.get("label", profile_id),
            kind=ProfileKind.FIXED_BOX,
            enabled=bool(item.get("enabled", True)),
            description="Migrated from configs/default.json fixed_watermarks",
            detector={
                "box": item.get("box", {}),
                "mask_mode": item.get("mask_mode", "rectangle"),
                "bright_threshold": item.get("bright_threshold", 190),
                "low_saturation_threshold": item.get("low_saturation_threshold", 130),
                "min_mask_ratio": item.get("min_mask_ratio", 0.0),
                "min_span_ratio": item.get("min_span_ratio", 0.0),
                "dilate": item.get("dilate", 0),
                "fallback_to_rectangle": item.get("fallback_to_rectangle", True),
            },
            created_from={"source": "legacy_config", "config": str(resolved)},
        )
        store.save(profile)
        created.append(profile_id)

    for item in config.get("template_watermarks", []):
        profile_id = slugify(item.get("label", "template"))
        if store.exists(profile_id) and not overwrite:
            continue
        template_src = resolve_config_path(item["template_path"], resolved)
        directory = store.profile_dir(profile_id)
        directory.mkdir(parents=True, exist_ok=True)
        template_name = "template_mask.png"
        if template_src.exists():
            shutil.copy2(template_src, directory / template_name)
        detector = {
            key: value
            for key, value in item.items()
            if key not in {"enabled", "label", "template_path"}
        }
        profile = Profile(
            id=profile_id,
            name=item.get("label", profile_id),
            kind=ProfileKind.TEMPLATE,
            enabled=bool(item.get("enabled", True)),
            description="Migrated from configs/default.json template_watermarks",
            template_file=template_name,
            detector=detector,
            created_from={
                "source": "legacy_config",
                "config": str(resolved),
                "template": str(template_src),
            },
        )
        store.save(profile)
        # keep sample of original asset if present under project assets
        samples = directory / "samples"
        samples.mkdir(exist_ok=True)
        created.append(profile_id)

    _ = app_root()
    return created
