"""Application service for creating and managing watermark profiles."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..image_io import read_image, write_image
from ..models import BBox
from ..profiles.models import MatchStrategy, Profile, ProfileKind
from ..profiles.store import ProfileStore, bootstrap_builtin_profiles
from ..workspace import Workspace, get_workspace
from .template_builder import (
    TemplateBuildResult,
    build_solid_roi_template,
    build_template_from_roi,
)


@dataclass(frozen=True)
class RoiNorm:
    """Normalized ROI in [0, 1] relative coordinates (left, top, right, bottom)."""

    left: float
    top: float
    right: float
    bottom: float

    def to_bbox(self, width: int, height: int) -> BBox:
        x1 = int(round(self.left * width))
        y1 = int(round(self.top * height))
        x2 = int(round(self.right * width))
        y2 = int(round(self.bottom * height))
        return BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    @classmethod
    def from_bbox(cls, bbox: BBox, width: int, height: int) -> "RoiNorm":
        return cls(
            left=bbox.x / width,
            top=bbox.y / height,
            right=bbox.right / width,
            bottom=bbox.bottom / height,
        )


class ProfileService:
    def __init__(self, workspace: Workspace | None = None) -> None:
        self.workspace = workspace or get_workspace()
        self.store = ProfileStore(self.workspace)

    @property
    def _seed_marker(self) -> Path:
        return self.workspace.root / ".builtins_seeded"

    def ensure_builtins(self, overwrite: bool = False) -> list[str]:
        """Seed built-in profiles once.

        Important: do NOT re-create after the user deletes them.
        Re-run only when marker missing, or overwrite=True (CLI bootstrap --overwrite).
        """
        if self._seed_marker.exists() and not overwrite:
            return []
        created = bootstrap_builtin_profiles(self.workspace, overwrite=overwrite)
        try:
            self._seed_marker.write_text("1\n", encoding="utf-8")
        except OSError:
            pass
        return created

    def list_profiles(self) -> list[Profile]:
        """List profiles on disk only — do not auto-seed builtins into empty libraries.

        Built-ins are created only via ``ensure_builtins`` / CLI ``profile bootstrap``.
        Fresh installs and packaged apps start with an empty style library.
        """
        return self.store.list_profiles()

    def get(self, profile_id: str) -> Profile:
        return self.store.get(profile_id)

    def set_enabled(self, profile_id: str, enabled: bool) -> Profile:
        return self.store.set_enabled(profile_id, enabled)

    def delete(self, profile_id: str) -> None:
        # Ensure seed marker exists so ensure_builtins won't resurrect deleted builtins
        if not self._seed_marker.exists():
            try:
                self._seed_marker.write_text("1\n", encoding="utf-8")
            except OSError:
                pass
        self.store.delete(profile_id)

    def create_from_roi(
        self,
        *,
        name: str,
        image_path: Path,
        roi: RoiNorm | BBox,
        description: str = "",
        profile_id: str | None = None,
        dilate: int = 1,
        detector_overrides: dict[str, Any] | None = None,
        match_strategy: MatchStrategy | str = MatchStrategy.AUTO,
        angle_deg: float = 0.0,
    ) -> tuple[Profile, TemplateBuildResult, Path]:
        """Create a unified watermark style from a user box selection."""
        image = read_image(image_path)
        height, width = image.shape[:2]
        if isinstance(roi, RoiNorm):
            bbox = roi.to_bbox(width, height)
            roi_norm = roi
        else:
            bbox = roi
            roi_norm = RoiNorm.from_bbox(bbox, width, height)

        if bbox.width < 8 or bbox.height < 8:
            raise ValueError("选区过小，请框选更大的水印区域。")

        # Default: solid filled ROI (no OpenCV/AI extract). Detail page can re-extract.
        # Honor canvas rotation so diagonal boxes still crop upright.
        build = build_solid_roi_template(
            image, bbox, dilate=0, angle_deg=float(angle_deg or 0.0)
        )
        allocated_id = profile_id or self.store.allocate_id(name)
        directory = self.store.profile_dir(allocated_id)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        template_name = "template_mask.png"
        sample_name = "sample_crop.png"
        preview_name = "preview_overlay.png"
        write_image(directory / template_name, build.template_mask)
        write_image(directory / sample_name, build.sample_crop_bgr)
        write_image(directory / preview_name, build.preview_overlay_bgr)

        samples_dir = directory / "samples"
        samples_dir.mkdir(exist_ok=True)
        shutil.copy2(image_path, samples_dir / image_path.name)

        detector = dict(build.suggested_detector)
        if detector_overrides:
            detector.update(detector_overrides)

        strategy = (
            match_strategy
            if isinstance(match_strategy, MatchStrategy)
            else MatchStrategy(str(match_strategy))
        )
        # Apply strategy defaults into detector search regions when explicit
        if strategy == MatchStrategy.SEARCH:
            detector["search_regions"] = [{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}]
            detector["min_bottom_ratio"] = 0.0
            # Full-frame single logo by default; paint 满屏 can set multi_instance True
            detector.setdefault("multi_instance", False)
            detector.setdefault("max_instances", 32)
            detector.setdefault("candidate_limit", max(int(detector.get("candidate_limit") or 12), 24))
            detector.setdefault("feature_mode", "fused")
            detector.setdefault("feature_adaptive", True)
            detector.setdefault("feature_threshold", 12)
            detector.setdefault("multi_score_ratio", 0.82)
            detector["min_confidence"] = max(float(detector.get("min_confidence") or 0.0), 0.38)
        elif strategy == MatchStrategy.FOLLOW:
            pad = 0.08
            detector["search_regions"] = [
                {
                    "left": max(0.0, roi_norm.left - pad),
                    "top": max(0.0, roi_norm.top - pad),
                    "right": min(1.0, roi_norm.right + pad),
                    "bottom": min(1.0, roi_norm.bottom + pad),
                }
            ]
            detector.setdefault("multi_instance", False)
            detector["min_confidence"] = max(float(detector.get("min_confidence") or 0.0), 0.35)

        profile = Profile(
            id=allocated_id,
            name=name.strip() or allocated_id,
            kind=ProfileKind.TEMPLATE,
            enabled=True,
            description=description,
            template_file=template_name,
            detector=detector,
            match_strategy=strategy,
            created_from={
                "mode": "roi",
                "image": str(image_path),
                "solid_region": True,
                "angle_deg": round(float(angle_deg), 2),
                "roi_norm": {
                    "left": round(roi_norm.left, 6),
                    "top": round(roi_norm.top, 6),
                    "right": round(roi_norm.right, 6),
                    "bottom": round(roi_norm.bottom, 6),
                },
                "stats": build.stats,
            },
        )
        self.store.save(profile)
        if not self._seed_marker.exists():
            try:
                self._seed_marker.write_text("1\n", encoding="utf-8")
            except OSError:
                pass
        return profile, build, directory

    def create_from_paint_mask(
        self,
        *,
        name: str,
        image_path: Path,
        mask_gray,  # np.ndarray HxW
        description: str = "",
        match_strategy: MatchStrategy | str = MatchStrategy.SEARCH,
    ) -> tuple[Profile, Path]:
        """Create style from freehand paint mask on a sample image."""
        import cv2
        import numpy as np

        image = read_image(image_path)
        mask = np.asarray(mask_gray)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        if int(np.count_nonzero(mask)) < 32:
            raise ValueError("涂抹区域过小，请扩大笔刷范围。")

        ys, xs = np.where(mask > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad = 4
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1] - 1, x2 + pad)
        y2 = min(image.shape[0] - 1, y2 + pad)
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
        # Solid bounding rect of the stroke — region for pin + stable bbox for match
        crop_mask = np.full((y2 - y1 + 1, x2 - x1 + 1), 255, dtype=np.uint8)
        overlay = crop.copy()
        red = overlay.copy()
        red[:, :] = (0, 0, 255)
        overlay = cv2.addWeighted(crop, 0.45, red, 0.55, 0)

        allocated_id = self.store.allocate_id(name)
        directory = self.store.profile_dir(allocated_id)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_image(directory / "template_mask.png", crop_mask)
        write_image(directory / "sample_crop.png", crop)
        write_image(directory / "preview_overlay.png", overlay)
        samples = directory / "samples"
        samples.mkdir(exist_ok=True)
        shutil.copy2(image_path, samples / image_path.name)

        h, w = image.shape[:2]
        roi_norm = RoiNorm(left=x1 / w, top=y1 / h, right=(x2 + 1) / w, bottom=(y2 + 1) / h)
        detector = {
            "reference_width": int(w),
            "scale_factors": [0.7, 0.85, 1.0, 1.2, 1.4],
            "min_confidence": 0.28,
            "feature_mode": "fused",
            "feature_adaptive": True,
            "feature_threshold": 12,
            "output_mask_mode": "bbox",
            "search_regions": [{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}],
            "min_bottom_ratio": 0.0,
            "dilate": 3,
            "mask_expand_ratio": 0.08,
            "edge_expand_ratio": 0.05,
            # 涂抹建档：外接矩形区域；批量全图时可多实例
            "multi_instance": True,
            "max_instances": 16,
            "candidate_limit": 20,
            "dual_feature_search": False,
            "multi_score_ratio": 0.82,
            "prefer_larger_score_margin": 0.0,
        }
        strategy = (
            match_strategy
            if isinstance(match_strategy, MatchStrategy)
            else MatchStrategy(str(match_strategy))
        )
        if strategy in {MatchStrategy.FOLLOW, MatchStrategy.AUTO, MatchStrategy.PIN}:
            detector["multi_instance"] = False
        profile = Profile(
            id=allocated_id,
            name=name.strip() or allocated_id,
            kind=ProfileKind.TEMPLATE,
            enabled=True,
            description=description,
            template_file="template_mask.png",
            detector=detector,
            match_strategy=strategy,
            created_from={
                "mode": "paint",
                "image": str(image_path),
                "solid_region": True,
                "roi_norm": {
                    "left": round(roi_norm.left, 6),
                    "top": round(roi_norm.top, 6),
                    "right": round(roi_norm.right, 6),
                    "bottom": round(roi_norm.bottom, 6),
                },
            },
        )
        self.store.save(profile)
        return profile, directory

    def create_from_crop_image(
        self,
        *,
        name: str,
        crop_path: Path,
        reference_width: int | None = None,
        description: str = "",
        profile_id: str | None = None,
    ) -> tuple[Profile, TemplateBuildResult, Path]:
        """Create a profile from an already-cropped watermark image."""
        import numpy as np

        crop = read_image(crop_path)
        height, width = crop.shape[:2]
        ref_w = reference_width or max(width * 4, 1024)
        ref_h = max(height * 4, 1024)
        canvas = np.zeros((ref_h, ref_w, 3), dtype=crop.dtype)
        x = (ref_w - width) // 2
        y = int(ref_h * 0.72)
        canvas[y : y + height, x : x + width] = crop
        build = build_template_from_roi(canvas, BBox(x, y, width, height))

        allocated_id = profile_id or self.store.allocate_id(name)
        directory = self.store.profile_dir(allocated_id)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        template_name = "template_mask.png"
        write_image(directory / template_name, build.template_mask)
        write_image(directory / "sample_crop.png", crop)
        write_image(directory / "preview_overlay.png", build.preview_overlay_bgr)
        samples = directory / "samples"
        samples.mkdir(exist_ok=True)
        shutil.copy2(crop_path, samples / crop_path.name)

        profile = Profile(
            id=allocated_id,
            name=name.strip() or allocated_id,
            kind=ProfileKind.TEMPLATE,
            enabled=True,
            description=description,
            template_file=template_name,
            detector=build.suggested_detector,
            created_from={"mode": "crop", "image": str(crop_path), "stats": build.stats},
        )
        self.store.save(profile)
        return profile, build, directory

    def preview_roi(
        self,
        image_path: Path,
        roi: RoiNorm | BBox,
        dilate: int = 1,
        angle_deg: float = 0.0,
    ) -> TemplateBuildResult:
        image = read_image(image_path)
        height, width = image.shape[:2]
        bbox = roi.to_bbox(width, height) if isinstance(roi, RoiNorm) else roi
        # Preview uses solid region by default (same as save)
        return build_template_from_roi(
            image,
            bbox,
            dilate=dilate,
            angle_deg=float(angle_deg or 0.0),
            solid=True,
        )

    def update_template_mask(
        self,
        profile_id: str,
        mask_gray,
        *,
        rebuild_preview: bool = True,
    ) -> Path:
        """Replace template_mask.png (and preview) after user edit / re-extract.

        ``mask_gray`` must match sample_crop size (or will be resized).
        """
        import cv2
        import numpy as np

        from ..services.template_builder import _preview_overlay

        directory = self.store.profile_dir(profile_id)
        if not directory.is_dir():
            raise FileNotFoundError(f"样式不存在: {profile_id}")
        sample_path = directory / "sample_crop.png"
        if not sample_path.is_file():
            raise FileNotFoundError("缺少 sample_crop.png，无法保存模板")

        sample = read_image(sample_path)
        mask = np.asarray(mask_gray)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        sh, sw = sample.shape[:2]
        if mask.shape[0] != sh or mask.shape[1] != sw:
            mask = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        if int(np.count_nonzero(mask)) < 8:
            raise ValueError("模板为空，请保留水印区域或重新提取。")

        write_image(directory / "template_mask.png", mask)
        if rebuild_preview:
            preview = _preview_overlay(sample, mask)
            write_image(directory / "preview_overlay.png", preview)

        # Touch profile metadata
        try:
            profile = self.get(profile_id)
            cf = dict(profile.created_from or {})
            cf["template_edited"] = True
            cf["template_fill"] = round(float(np.count_nonzero(mask) / mask.size), 4)
            profile.created_from = cf
            self.store.save(profile)
        except Exception:  # noqa: BLE001
            pass
        return directory / "template_mask.png"

    def reextract_template_from_sample(
        self, profile_id: str, *, dilate: int = 1
    ):
        """Re-run automatic extraction on sample_crop (same size) and overwrite mask.

        Default dilate is small (1): template matching wants letter-like shape,
        not a fat capsule. Inpaint coverage is handled later by hit-box expand.
        """
        from ..services.template_builder import extract_mask_from_crop

        directory = self.store.profile_dir(profile_id)
        sample_path = directory / "sample_crop.png"
        if not sample_path.is_file():
            raise FileNotFoundError("缺少 sample_crop.png")
        sample = read_image(sample_path)
        mask, stats = extract_mask_from_crop(sample, dilate=dilate)
        self.update_template_mask(profile_id, mask, rebuild_preview=True)
        return {"mask": mask, "stats": stats}
