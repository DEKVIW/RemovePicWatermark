"""Batch detection and inpaint jobs with structured workspace outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..backends.iopaint import run_iopaint
from ..backends.opencv import inpaint
from ..detectors import FixedBoxDetector, TemplateStampDetector
from ..detectors.orchestrator import DetectOrchestrator, normalize_detect_mode
from ..detectors.residual_ai import ResidualAiDetector
from ..detectors.yolo_watermark import (
    YoloWatermarkDetector,
    ensure_yolo_dir,
    probe_yolo,
)
from ..image_io import iter_image_files, read_image, write_image
from ..masking import combine_masks, draw_debug_overlay
from ..profiles.models import MatchStrategy, Profile, ProfileKind
from ..profiles.store import ProfileStore
from ..workspace import Workspace, get_workspace


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class JobCancelled(Exception):
    """Raised when the user requests stop mid-job (optional control flow)."""


@dataclass
class JobSpec:
    input_path: Path
    profile_ids: list[str] | None = None  # None = all enabled; ignored for pure AI if empty
    backend: str = "opencv"  # none | opencv | iopaint
    job_id: str | None = None
    opencv_radius: int = 7
    iopaint_model: str = "lama"
    iopaint_device: str = "cpu"
    iopaint_model_dir: Path | None = None
    iopaint_executable: str | None = None
    copy_inputs: bool = True
    # batch locate strategy override: pin | follow | search | auto | None(=per-profile)
    match_strategy: str | None = None
    # styles | ai | both  (product: 水印样式 / 水印模型 / 样式+模型)
    detect_mode: str = "styles"
    # optional YOLO weights; residual AI always available for ai/both
    yolo_weights: Path | str | None = None
    enable_residual_ai: bool = True
    enable_yolo: bool = True


@dataclass
class JobResult:
    job_id: str
    job_dir: Path
    input_dir: Path
    mask_dir: Path
    debug_dir: Path
    output_dir: Path
    report_path: Path
    images: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class JobService:
    def __init__(self, workspace: Workspace | None = None) -> None:
        self.workspace = workspace or get_workspace()
        self.store = ProfileStore(self.workspace)

    def create_job_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(
        self,
        spec: JobSpec,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> JobResult:
        detect_mode = normalize_detect_mode(spec.detect_mode)
        strategy_key = (spec.match_strategy or "").strip().lower()
        # Fixed pin requires styles only (no model scan)
        if strategy_key == "pin":
            detect_mode = "styles"
            from dataclasses import replace

            spec = replace(
                spec,
                detect_mode="styles",
                enable_yolo=False,
                match_strategy="pin",
            )
        profiles = self._resolve_profiles(spec.profile_ids)
        if detect_mode == "styles" and not profiles:
            raise ValueError("请至少勾选一个水印样式。")
        if detect_mode == "both" and not profiles:
            # degrade to model-only rather than fail hard
            detect_mode = "ai"
        if detect_mode == "ai" and not spec.enable_yolo and not spec.enable_residual_ai:
            raise ValueError("水印模型不可用，请先在「训练检测」页训练，或改用「水印样式」。")

        job_id = spec.job_id or self.create_job_id()
        job_dir = self.workspace.job_dir(job_id)
        input_dir = job_dir / "input"
        mask_dir = job_dir / "masks"
        debug_dir = job_dir / "debug"
        output_dir = job_dir / "output"
        for path in (job_dir, input_dir, mask_dir, debug_dir, output_dir):
            path.mkdir(parents=True, exist_ok=True)

        source_images = iter_image_files(spec.input_path)
        if not source_images:
            raise ValueError(f"No supported image files found: {spec.input_path}")

        # Stage inputs into job folder for stable relative paths / GUI browsing
        staged: list[Path] = []
        for source in source_images:
            target = input_dir / source.name
            if target.exists() and target.resolve() != source.resolve():
                stem, suffix = source.stem, source.suffix
                n = 2
                while target.exists():
                    target = input_dir / f"{stem}_{n}{suffix}"
                    n += 1
            if spec.copy_inputs or target.resolve() != source.resolve():
                if not target.exists():
                    shutil.copy2(source, target)
            staged.append(target)

        style_detectors = (
            self._build_detectors(
                profiles,
                strategy_override=spec.match_strategy,
                enable_yolo_propose=bool(spec.enable_yolo),
                yolo_weights=spec.yolo_weights,
                yolo_device=spec.iopaint_device or "cpu",
            )
            if detect_mode in {"styles", "both"}
            else []
        )
        yolo_probe = None
        # Probe whenever YOLO might be used (styles propose and/or ai chain)
        if spec.enable_yolo or detect_mode in {"ai", "both"}:
            ensure_yolo_dir(self.workspace.models_dir)
            yolo_probe = probe_yolo(
                self.workspace.models_dir,
                spec.yolo_weights,
                try_load=False,
                device=spec.iopaint_device,
            )

        ai_detectors = (
            self._build_ai_detectors(spec, detect_mode=detect_mode, has_styles=bool(profiles))
            if detect_mode in {"ai", "both"}
            else []
        )
        if detect_mode == "ai" and not ai_detectors:
            raise ValueError(
                "水印模型不可用。请先在「训练检测」页训练，或改用「水印样式」。"
            )
        if detect_mode == "both" and not style_detectors and not ai_detectors:
            raise ValueError("请勾选样式，或改用「水印模型」。")

        style_size_hints = self._style_size_hints(profiles) if profiles else []
        # both = cascade (style first; AI only if style empty) — see DetectOrchestrator
        orchestrator = DetectOrchestrator(
            style_detectors=style_detectors,
            ai_detectors=ai_detectors,
            mode=detect_mode if (style_detectors or detect_mode == "ai") else "ai",
            style_size_hints=style_size_hints,
            filter_ai_to_style_scale=bool(
                detect_mode in {"both", "ai"} and (profiles or style_size_hints)
            ),
            ai_fill_min_confidence=0.22,
            ai_fill_yolo_min_confidence=0.25,
            ai_max_image_area_ratio=0.05,
        )
        # If both requested but styles empty, degrade to AI-only fill path
        if detect_mode == "both" and not style_detectors:
            orchestrator.mode = "ai"
            orchestrator.filter_ai_to_style_scale = bool(style_size_hints)
        elif detect_mode == "both" and not ai_detectors:
            orchestrator.mode = "styles"

        report_items: list[dict[str, Any]] = []
        n = len(staged)
        # Work units: each image is "detect" (+ "repair" for LaMa). OpenCV repairs inline.
        # completed is reported AFTER each unit finishes so the bar never sits at 100% early.
        if spec.backend == "iopaint":
            total_units = max(1, n * 2)  # detect n + repair n
        else:
            total_units = max(1, n)
        done_units = 0

        def _tick(message: str) -> None:
            nonlocal done_units
            done_units = min(done_units + 1, total_units)
            if progress:
                progress(done_units, total_units, message)

        if progress:
            progress(0, total_units, f"开始处理 {n} 张图…")

        cancelled = False
        for index, image_path in enumerate(staged):
            if should_cancel and should_cancel():
                cancelled = True
                if progress:
                    progress(done_units, total_units, f"已停止（完成 {index}/{n} 张检测）")
                break
            if progress:
                # status only — value advances after work completes
                progress(done_units, total_units, f"检测中 ({index + 1}/{n}) {image_path.name}")
            image = read_image(image_path)
            detections = orchestrator.detect(image)
            yolo_trace = dict(getattr(orchestrator, "last_run_stats", None) or {})
            combined = combine_masks(detections, image.shape[:2])
            mask_path = mask_dir / f"{image_path.stem}.png"
            write_image(mask_path, combined)
            debug = draw_debug_overlay(image, combined, detections)
            debug_path = debug_dir / f"{image_path.stem}.jpg"
            write_image(debug_path, debug)

            action = "detected" if np.any(combined > 0) else "empty_mask"
            if action == "empty_mask":
                if progress:
                    progress(
                        done_units,
                        total_units,
                        f"{image_path.name} 未识别到水印",
                    )
            output_path = None
            if spec.backend == "opencv" and np.any(combined > 0):
                repaired = inpaint(image, combined, radius=spec.opencv_radius)
                output_path = output_dir / image_path.name
                write_image(output_path, repaired)
                action = "opencv_inpainted"
            elif spec.backend == "opencv":
                output_path = output_dir / image_path.name
                shutil.copy2(image_path, output_path)
                action = "copied"

            report_items.append(
                {
                    "image": str(image_path),
                    "yolo_trace": yolo_trace,
                    "mask": str(mask_path),
                    "debug": str(debug_path),
                    "output": str(output_path) if output_path else None,
                    "action": action,
                    "detections": [item.to_report() for item in detections],
                }
            )
            hit = len(detections)
            _tick(f"已检测 ({index + 1}/{n}) {image_path.name} · {hit} 处")

        if cancelled:
            # skip repair phase; write report for partial work
            if progress:
                progress(done_units, total_units, "已停止，跳过后续修补")
        elif should_cancel and should_cancel():
            cancelled = True
            if progress:
                progress(done_units, total_units, "已停止，跳过修补阶段")
        elif spec.backend == "iopaint":
            nonempty = []
            for item in report_items:
                mask_file = Path(item["mask"])
                mask_img = read_image(mask_file)
                mask_gray = mask_img[:, :, 0] if mask_img.ndim == 3 else mask_img
                if np.any(mask_gray > 0):
                    nonempty.append(item)
                else:
                    out = output_dir / Path(item["image"]).name
                    shutil.copy2(item["image"], out)
                    item["output"] = str(out)
                    item["action"] = "skipped_no_detection"

            if not nonempty:
                while done_units < total_units:
                    _tick("未检测到水印，跳过修补")
            else:
                logs: list[str] = []

                def _iopaint_log(message: str) -> None:
                    logs.append(message)

                def _iopaint_progress(completed_repairs: int, repair_total: int, message: str) -> None:
                    # Second half of the bar: after n detects, then repairs 0..n
                    nonlocal done_units
                    value = min(total_units, n + completed_repairs)
                    done_units = max(done_units, value)
                    if progress:
                        progress(done_units, total_units, message)

                project_root = (
                    self.workspace.root.parent
                    if self.workspace.root.name == "workspace"
                    else Path.cwd()
                )
                run_iopaint(
                    input_dir,
                    mask_dir,
                    output_dir,
                    model=spec.iopaint_model,
                    device=spec.iopaint_device,
                    model_dir=spec.iopaint_model_dir or self.workspace.models_dir,
                    executable=spec.iopaint_executable,
                    project_root=project_root,
                    search_model_roots=[
                        self.workspace.models_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt",
                        project_root / "data" / "iopaint-models" / "torch" / "hub" / "checkpoints" / "big-lama.pt",
                        project_root / "workspace" / "models" / "torch" / "hub" / "checkpoints" / "big-lama.pt",
                    ],
                    log=_iopaint_log,
                    progress=_iopaint_progress,
                )
                try:
                    (job_dir / "iopaint.log").write_text("\n".join(logs), encoding="utf-8")
                except OSError:
                    pass

                for item in nonempty:
                    stem = Path(item["image"]).stem
                    candidates = list(output_dir.glob(f"{stem}.*"))
                    if candidates:
                        item["output"] = str(candidates[0])
                        item["action"] = "iopaint_inpainted"

                while done_units < total_units:
                    _tick("修补阶段收尾")

        report_path = job_dir / "report.json"
        empty_n = sum(1 for i in report_items if not i.get("detections"))
        ai_stats = _aggregate_ai_stats(report_items)
        cascade_stats = _aggregate_cascade_stats(report_items)
        hint = ""
        if empty_n:
            if orchestrator.mode == "styles":
                hint = "部分图未识别到水印：可改查找为「样式+模型」，或到「单张精修」手涂"
            elif orchestrator.mode == "both":
                hint = "部分图样式与检测均未命中：可补标训练、完善识别模板，或单张精修"
            else:
                hint = "部分图检测未命中：可到「训练检测」补数据重训，或单张精修"
        summary = {
            "job_id": job_id,
            "backend": spec.backend,
            "detect_mode": orchestrator.mode,
            "detectors": orchestrator.describe(),
            "yolo": yolo_probe.to_dict() if yolo_probe is not None else {"status": "not_requested"},
            "ai_stats": ai_stats,
            "cascade_stats": cascade_stats,
            "profiles": [p.id for p in profiles],
            "image_count": len(report_items),
            "cancelled": cancelled,
            "detected": sum(1 for i in report_items if i.get("detections")),
            "empty_mask": empty_n,
            "actions": _count_actions(report_items),
            "hint": hint,
        }
        payload = {"summary": summary, "images": report_items}
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if progress:
            progress(total_units, total_units, "全部完成")

        return JobResult(
            job_id=job_id,
            job_dir=job_dir,
            input_dir=input_dir,
            mask_dir=mask_dir,
            debug_dir=debug_dir,
            output_dir=output_dir,
            report_path=report_path,
            images=report_items,
            summary=summary,
        )

    def _resolve_profiles(self, profile_ids: list[str] | None) -> list[Profile]:
        """Resolve profiles for a job.

        When profile_ids is provided (e.g. GUI multi-select), use them as-is —
        do not re-filter by the library's enabled flag. The caller's selection
        is the source of truth for this run.
        """
        if profile_ids is None:
            return [p for p in self.store.list_profiles() if p.enabled]
        selected: list[Profile] = []
        missing: list[str] = []
        for profile_id in profile_ids:
            try:
                selected.append(self.store.get(profile_id))
            except FileNotFoundError:
                missing.append(profile_id)
        if missing:
            raise FileNotFoundError(f"Profile not found: {', '.join(missing)}")
        return selected

    def _style_size_hints(self, profiles: list[Profile]) -> list[tuple[int, int]]:
        """Template mask sizes (w, h) for scale-aware AI filtering in both mode."""
        hints: list[tuple[int, int]] = []
        for profile in profiles:
            if profile.kind != ProfileKind.TEMPLATE:
                continue
            path = profile.template_path(self.store.profile_dir(profile.id))
            if path is None or not path.is_file():
                continue
            try:
                mask = read_image(path)
                h, w = mask.shape[:2]
                if w >= 4 and h >= 4:
                    hints.append((int(w), int(h)))
            except Exception:  # noqa: BLE001
                continue
            # also sample_crop if present (often closer to on-image stamp size)
            crop = self.store.profile_dir(profile.id) / "sample_crop.png"
            if not crop.is_file():
                crop = self.store.profile_dir(profile.id) / "sample_crop.jpg"
            if crop.is_file():
                try:
                    im = read_image(crop)
                    ch, cw = im.shape[:2]
                    if cw >= 4 and ch >= 4:
                        hints.append((int(cw), int(ch)))
                except Exception:  # noqa: BLE001
                    pass
        return hints

    def _build_ai_detectors(
        self,
        spec: JobSpec,
        *,
        detect_mode: str = "ai",
        has_styles: bool = False,
    ) -> list[Any]:
        """AI fill chain: residual + YOLO.

        For「样式+模型」(both) with styles selected this list only runs when
        style matching found nothing (cascade). Residual is off in that case —
        it caused multi-strip false positives; YOLO propose→confirm already
        lives inside the style detector.
        """
        detectors: list[Any] = []
        cascade_fill = detect_mode == "both" and has_styles

        if spec.enable_residual_ai and not cascade_fill:
            # Pure AI mode: residual for offline scan. Cascade fill: YOLO only.
            residual_cfg: dict[str, Any] = {
                "label": "ai_residual",
                "feature_mode": "fused",
                "feature_adaptive": True,
                "feature_threshold": 10,
                "max_instances": 16,
                "max_area_ratio": 0.05,
                "min_area": 100,
                "reject_border_full": True,
            }
            detectors.append(ResidualAiDetector.from_config(residual_cfg))
        elif spec.enable_residual_ai and cascade_fill:
            # Very conservative residual only as last resort if YOLO missing
            pass

        if spec.enable_yolo:
            # Cascade empty-fill needs higher conf; pure AI can be slightly lower
            if cascade_fill:
                conf = 0.25
            elif detect_mode == "ai":
                conf = 0.18
            else:
                conf = 0.22
            yolo = YoloWatermarkDetector.try_create(
                workspace_models=self.workspace.models_dir,
                weights=spec.yolo_weights,
                device=spec.iopaint_device,
                conf=conf,
                refine_with_residual=False,
            )
            if yolo is not None:
                detectors.append(yolo)
            elif cascade_fill and spec.enable_residual_ai:
                # No YOLO weights: allow tight residual as cascade fill
                residual_cfg = {
                    "label": "ai_residual",
                    "feature_mode": "fused",
                    "feature_adaptive": True,
                    "feature_threshold": 14,
                    "max_instances": 4,
                    "max_area_ratio": 0.03,
                    "min_area": 150,
                    "reject_border_full": True,
                }
                detectors.append(ResidualAiDetector.from_config(residual_cfg))
        return detectors

    def _build_detectors(
        self,
        profiles: list[Profile],
        strategy_override: str | None = None,
        *,
        enable_yolo_propose: bool = True,
        yolo_weights: Path | str | None = None,
        yolo_device: str = "cpu",
    ) -> list[Any]:
        detectors: list[Any] = []
        # Shared YOLO proposer for style templates: only used when primary match
        # misses, then still must pass template+ORB gates (no raw YOLO masks).
        yolo_proposer = None
        if enable_yolo_propose:
            # Propose-only path inside style matcher (must still pass template gates).
            # Slightly lower conf than AI fill — proposals are confirmed by style match.
            yolo_proposer = YoloWatermarkDetector.try_create(
                workspace_models=self.workspace.models_dir,
                weights=yolo_weights,
                device=yolo_device,
                conf=0.18,
                refine_with_residual=False,
            )
        for profile in profiles:
            strategy = self._effective_strategy(profile, strategy_override)
            if strategy == MatchStrategy.PIN or profile.kind == ProfileKind.FIXED_BOX:
                pin = self._build_pin_detector(profile)
                if pin is None:
                    raise ValueError(
                        f"样式「{profile.name}」缺少位置信息，无法使用固定位置。"
                        "请用框选或涂抹重新保存样式。"
                    )
                detectors.append(pin)
                continue
            if profile.kind == ProfileKind.TEMPLATE:
                template_path = profile.template_path(self.store.profile_dir(profile.id))
                if template_path is None or not template_path.exists():
                    raise FileNotFoundError(f"样式模板缺失：{profile.id}")
                detector = dict(profile.detector)
                detector = self._apply_match_strategy(profile, detector, strategy)
                config = {
                    "label": profile.id,
                    "template_path": str(template_path),
                    **detector,
                }
                stamp = TemplateStampDetector.from_config(
                    config, template_path.parent / "profile.json"
                )
                if yolo_proposer is not None:
                    stamp.attach_yolo_proposer(yolo_proposer)
                detectors.append(stamp)
        return detectors

    def _build_pin_detector(self, profile: Profile) -> FixedBoxDetector | None:
        """Hard-paste a solid rectangle at recorded position — no matching, no template shape."""
        box = self._pin_box_for_profile(profile)
        if box is None:
            return None
        dilate = int((profile.detector or {}).get("dilate") or 3)
        dilate = min(max(dilate, 2), 5)
        # Always solid ROI; never paste recognition-template contours
        config: dict[str, Any] = {
            "label": profile.id,
            "box": box,
            "mask_mode": "rectangle",
            "dilate": dilate,
            "template_mask": None,
            "fallback_to_rectangle": True,
        }
        return FixedBoxDetector.from_config(config)

    def _pin_box_for_profile(self, profile: Profile) -> dict[str, float] | None:
        """Normalized box for pin: detector.box or created_from.roi_norm."""
        det = profile.detector or {}
        raw = det.get("box")
        if isinstance(raw, dict) and all(k in raw for k in ("left", "top", "right", "bottom")):
            return {
                "left": float(raw["left"]),
                "top": float(raw["top"]),
                "right": float(raw["right"]),
                "bottom": float(raw["bottom"]),
            }
        roi = (profile.created_from or {}).get("roi_norm") or {}
        if isinstance(roi, dict) and all(k in roi for k in ("left", "top", "right", "bottom")):
            return {
                "left": float(roi["left"]),
                "top": float(roi["top"]),
                "right": float(roi["right"]),
                "bottom": float(roi["bottom"]),
            }
        # Fallback: full-frame if only template exists (rare)
        return None

    def _effective_strategy(self, profile: Profile, override: str | None) -> MatchStrategy:
        if override and override not in {"", "auto", "None"}:
            try:
                if override == "auto":
                    # legacy auto → nearby match
                    if profile.match_strategy == MatchStrategy.AUTO:
                        return (
                            MatchStrategy.FOLLOW
                            if profile.kind == ProfileKind.FIXED_BOX
                            else MatchStrategy.FOLLOW
                        )
                    return profile.match_strategy
                return MatchStrategy(override)
            except ValueError:
                pass
        if profile.match_strategy == MatchStrategy.AUTO:
            return MatchStrategy.FOLLOW
        return profile.match_strategy

    def _apply_match_strategy(
        self,
        profile: Profile,
        detector: dict[str, Any],
        strategy: MatchStrategy,
    ) -> dict[str, Any]:
        out = dict(detector)
        created_mode = str((profile.created_from or {}).get("mode") or "").lower()
        # PIN is handled by FixedBoxDetector path — should not reach here.
        if strategy == MatchStrategy.PIN:
            return out
        # ROI single-logo styles must never flood with multi-instance hits.
        # Paint / 满屏 may keep multi_instance=True when the profile opts in.
        if strategy == MatchStrategy.SEARCH:
            # Full-frame locate. Multi-instance only if profile opts in (满屏重复单元).
            out["search_regions"] = [{"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}]
            out["min_bottom_ratio"] = 0.0
            if created_mode == "roi" or "multi_instance" not in out:
                out["multi_instance"] = False
            multi = bool(out.get("multi_instance"))
            # Cap work: multi-instance + many scales is O(scales × peaks × ORB) per image
            out.setdefault("max_instances", 16 if multi else 8)
            out["max_instances"] = min(int(out.get("max_instances") or 16), 20 if multi else 12)
            out.setdefault("candidate_limit", 20 if multi else 12)
            out["candidate_limit"] = min(int(out.get("candidate_limit") or 20), 24 if multi else 16)
            out.setdefault("nms_iou", 0.3)
            out.setdefault("feature_mode", "fused")
            out.setdefault("feature_adaptive", True)
            out.setdefault("feature_threshold", 12)
            # Soft floor only — do not raise profile confidence (hurts recall)
            out["min_confidence"] = float(out.get("min_confidence") or 0.28)
            out.setdefault("multi_score_ratio", 0.78)
            out["prefer_larger_score_margin"] = 0.0
            # Dual feature path doubles TM cost — keep for single-logo only
            if multi:
                out["dual_feature_search"] = False
            # Tighter scale pyramid (was up to 8) — main cost of batch detect
            scales = out.get("scale_factors") or []
            if not isinstance(scales, list) or len(scales) < 4:
                out["scale_factors"] = (
                    [0.7, 0.85, 1.0, 1.2, 1.4] if multi else [0.7, 0.85, 1.0, 1.15, 1.35]
                )
            elif len(scales) > 6:
                # Keep mid scales only when profile has a long list
                mid = sorted(float(s) for s in scales)
                step = max(1, len(mid) // 5)
                out["scale_factors"] = mid[::step][:6]
            out.setdefault("topk_validate", 12 if multi else 16)
            out["topk_validate"] = min(int(out.get("topk_validate") or 12), 16)
        elif strategy == MatchStrategy.FOLLOW:
            # Position-biased: single hit near the sample ROI (wider pad for drift)
            roi = (profile.created_from or {}).get("roi_norm") or {}
            if roi:
                pad = 0.18
                out["search_regions"] = [
                    {
                        "left": max(0.0, float(roi.get("left", 0)) - pad),
                        "top": max(0.0, float(roi.get("top", 0)) - pad),
                        "right": min(1.0, float(roi.get("right", 1)) + pad),
                        "bottom": min(1.0, float(roi.get("bottom", 1)) + pad),
                    }
                ]
            out["multi_instance"] = False
            out["dual_feature_search"] = True
            out["min_confidence"] = float(out.get("min_confidence") or 0.28)
            out["prefer_larger_score_margin"] = 0.0
            # Nearby match: fewer scales needed
            scales = out.get("scale_factors") or []
            if not isinstance(scales, list) or len(scales) > 5:
                out["scale_factors"] = [0.85, 1.0, 1.15, 1.3]
            out.setdefault("candidate_limit", 10)
            out.setdefault("topk_validate", 12)

        # Product clamps: locate + solid bbox for LaMa (not contour paste)
        if created_mode == "roi":
            out["multi_instance"] = False
        conf = float(out.get("min_confidence") or 0.28)
        # Soft floor for garbage scores; never force 0.38+ (was killing recall)
        out["min_confidence"] = max(0.22, min(conf, 0.55))
        # Always solid rectangle for inpaint coverage (legacy template/envelope → bbox)
        mode = str(out.get("output_mask_mode") or "bbox").strip().lower()
        if mode in {"template", "envelope", "footprint", ""}:
            out["output_mask_mode"] = "bbox"
        else:
            out["output_mask_mode"] = mode
        out["min_bottom_ratio"] = 0.0
        out["dilate"] = min(max(int(out.get("dilate") or 4), 3), 6)
        # Localization quality defaults (can be overridden by profile)
        out.setdefault("refine_enabled", True)
        out.setdefault("min_match_side", 48)
        out.setdefault("min_side_ratio", 0.04)
        out.setdefault("max_side_ratio", 0.48)
        out.setdefault("min_scale_vs_expected", 0.55)
        out.setdefault("max_scale_vs_expected", 1.55)
        out.setdefault("min_residual_density", 0.06)
        out.setdefault("min_structure_alignment", 0.10)
        out.setdefault("weak_score_threshold", 0.38)
        out.setdefault("weak_min_structure", 0.16)
        out.setdefault("min_orb_matches", 8)
        out.setdefault("strong_score_threshold", 0.45)
        out.setdefault("topk_validate", 28)
        out.setdefault("secondary_search_enabled", True)
        out.setdefault("dual_feature_search", True)
        out.setdefault("yolo_propose_enabled", True)
        for key, cap in (
            ("mask_expand_ratio", 0.16),
            ("mask_expand_top_ratio", 0.16),
            ("mask_expand_bottom_ratio", 0.16),
            ("mask_expand_x_ratio", 0.16),
            ("edge_expand_ratio", 0.10),
        ):
            if key in out and out[key] is not None:
                try:
                    # Floor for complete stamp coverage; cap to avoid huge boxes
                    val = float(out[key])
                    if key.startswith("mask_expand"):
                        val = max(val, 0.12)
                    out[key] = min(val, cap)
                except (TypeError, ValueError):
                    pass
            elif key.startswith("mask_expand"):
                out[key] = 0.12
        # Dense solid templates (bad extraction) need a slightly stricter gate
        try:
            fill = float(((profile.created_from or {}).get("stats") or {}).get("fill_ratio") or 0.0)
        except (TypeError, ValueError):
            fill = 0.0
        if fill >= 0.92:
            out["min_confidence"] = max(float(out.get("min_confidence") or 0.0), 0.35)
            out["multi_instance"] = False
            out["prefer_larger_score_margin"] = 0.0
        return out


def _count_actions(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        action = str(item.get("action") or "unknown")
        counts[action] = counts.get(action, 0) + 1
    return counts


def _aggregate_cascade_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Count cascade_path per image for product/debug (style hit vs AI fill)."""
    paths: dict[str, int] = {}
    for item in items:
        trace = item.get("yolo_trace") or {}
        path = str(trace.get("cascade_path") or "")
        if not path:
            # fallback: first detection metadata
            dets = item.get("detections") or []
            if dets and isinstance(dets[0], dict):
                meta = dets[0].get("metadata") or {}
                path = str(meta.get("cascade_path") or "")
        if not path:
            path = "unknown"
        paths[path] = paths.get(path, 0) + 1
    return {
        "by_path": paths,
        "style_hit_skip_ai": int(paths.get("style_hit_skip_ai", 0)),
        "style_miss_ai_fill": int(paths.get("style_miss_ai_fill", 0)),
        "style_weak_ai_fill": int(paths.get("style_weak_ai_fill", 0)),
        "style_weak_no_ai": int(paths.get("style_weak_no_ai", 0)),
        "style_miss_no_ai": int(paths.get("style_miss_no_ai", 0)),
        "styles_only": int(paths.get("styles_only", 0)),
        "ai_only": int(paths.get("ai_only", 0)),
    }


def _aggregate_ai_stats(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count final detections + YOLO raw/filter so users can see if weights fired."""
    style_n = 0
    yolo_n = 0
    residual_n = 0
    yolo_pass_n = 0  # style hit via search_pass containing yolo
    yolo_raw = 0
    yolo_kept = 0
    yolo_dropped = 0
    images_yolo_proposed = 0
    images_yolo_kept = 0
    for item in items:
        trace = item.get("yolo_trace") or {}
        raw = int(trace.get("yolo_raw_proposals") or 0)
        kept = int(trace.get("yolo_kept_after_filter") or 0)
        dropped = int(trace.get("yolo_dropped_by_filter") or 0)
        yolo_raw += raw
        yolo_kept += kept
        yolo_dropped += dropped
        if raw > 0:
            images_yolo_proposed += 1
        if kept > 0:
            images_yolo_kept += 1
        for det in item.get("detections") or []:
            md = det.get("metadata") or {}
            tag = str(md.get("detector") or "").lower()
            sp = str(md.get("search_pass") or "").lower()
            if "yolo" in tag:
                yolo_n += 1
            elif "residual" in tag:
                residual_n += 1
            else:
                style_n += 1
                if "yolo" in sp:
                    yolo_pass_n += 1
    return {
        "style_detections": style_n,
        "yolo_detections": yolo_n,
        "residual_detections": residual_n,
        "style_via_yolo_propose": yolo_pass_n,
        "yolo_raw_proposals": yolo_raw,
        "yolo_kept_after_filter": yolo_kept,
        "yolo_dropped_by_filter": yolo_dropped,
        "images_with_yolo_proposals": images_yolo_proposed,
        "images_with_yolo_kept": images_yolo_kept,
    }
