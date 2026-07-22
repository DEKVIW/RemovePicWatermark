from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .backends.opencv import inpaint
from .config import load_config
from .detectors import FixedBoxDetector, TemplateStampDetector
from .image_io import iter_image_files, read_image, write_image
from .masking import combine_masks, draw_debug_overlay
from .models import Detection
from .profiles.models import Profile, ProfileKind
from .profiles.store import ProfileStore, bootstrap_builtin_profiles
from .workspace import get_workspace


def build_detectors(config: dict[str, Any], config_path: Path) -> list[Any]:
    detectors: list[Any] = []
    for item in config.get("fixed_watermarks", []):
        if item.get("enabled", True):
            detectors.append(FixedBoxDetector.from_config(item))
    for item in config.get("template_watermarks", []):
        if item.get("enabled", True):
            detectors.append(TemplateStampDetector.from_config(item, config_path))
    return detectors


def build_detectors_from_profiles(
    profiles: list[Profile],
    store: ProfileStore | None = None,
) -> list[Any]:
    store = store or ProfileStore(get_workspace())
    detectors: list[Any] = []
    for profile in profiles:
        if not profile.enabled:
            continue
        if profile.kind == ProfileKind.FIXED_BOX:
            detectors.append(FixedBoxDetector.from_config({"label": profile.id, **profile.detector}))
        elif profile.kind == ProfileKind.TEMPLATE:
            template_path = profile.template_path(store.profile_dir(profile.id))
            if template_path is None or not template_path.exists():
                continue
            config = {"label": profile.id, "template_path": str(template_path), **profile.detector}
            detectors.append(TemplateStampDetector.from_config(config, store.profile_dir(profile.id) / "profile.json"))
    return detectors


def detect_image(image: np.ndarray, detectors: list[Any]) -> list[Detection]:
    detections: list[Detection] = []
    for detector in detectors:
        detections.extend(detector.detect(image))
    return detections


def run_detection_batch(
    input_path: Path,
    mask_dir: Path,
    debug_dir: Path | None = None,
    report_path: Path | None = None,
    config_path: Path | None = None,
    profile_ids: list[str] | None = None,
    use_profiles: bool = True,
) -> list[dict[str, Any]]:
    """Detect watermarks and write masks.

    By default prefers workspace profiles (after bootstrapping builtins).
    Pass use_profiles=False or an explicit legacy config_path-only workflow via use_profiles=False.
    """
    if use_profiles and config_path is None:
        workspace = get_workspace()
        bootstrap_builtin_profiles(workspace)
        store = ProfileStore(workspace)
        if profile_ids:
            profiles = [store.get(pid) for pid in profile_ids]
        else:
            profiles = [p for p in store.list_profiles() if p.enabled]
        detectors = build_detectors_from_profiles(profiles, store)
    else:
        config, resolved_config_path = load_config(config_path)
        detectors = build_detectors(config, resolved_config_path)

    images = iter_image_files(input_path)
    if not images:
        raise ValueError(f"No supported image files found: {input_path}")

    report: list[dict[str, Any]] = []

    for image_path in images:
        image = read_image(image_path)
        detections = detect_image(image, detectors)
        combined_mask = combine_masks(detections, image.shape[:2])

        mask_path = mask_dir / f"{image_path.stem}.png"
        write_image(mask_path, combined_mask)

        debug_path = None
        if debug_dir is not None:
            debug = draw_debug_overlay(image, combined_mask, detections)
            debug_path = debug_dir / f"{image_path.stem}.jpg"
            write_image(debug_path, debug)

        report.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "debug": str(debug_path) if debug_path else None,
                "detections": [detection.to_report() for detection in detections],
            }
        )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def run_opencv_preview(input_path: Path, mask_dir: Path, output_dir: Path, radius: int = 3) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    images = iter_image_files(input_path)
    if not images:
        raise ValueError(f"No supported image files found: {input_path}")

    for image_path in images:
        output_path = output_dir / image_path.name
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            copy_original(image_path, output_path)
            results.append({"image": str(image_path), "mask": "", "output": str(output_path), "action": "copied"})
            continue

        mask = read_image(mask_path)
        mask_gray = mask[:, :, 0] if mask.ndim == 3 else mask
        if not np.any(mask_gray > 0):
            copy_original(image_path, output_path)
            results.append({"image": str(image_path), "mask": str(mask_path), "output": str(output_path), "action": "copied"})
            continue

        image = read_image(image_path)
        output = inpaint(image, mask_gray, radius=radius)
        write_image(output_path, output)
        results.append({"image": str(image_path), "mask": str(mask_path), "output": str(output_path), "action": "inpainted"})
    return results


def copy_original(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
