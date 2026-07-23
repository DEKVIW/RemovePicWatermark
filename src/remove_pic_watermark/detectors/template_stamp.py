from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import resolve_config_path
from ..image_io import read_image
from ..masking import dilate_mask
from ..models import BBox, Detection
from .features import (
    FeatureParams,
    build_template_feature_edges,
    compute_feature_edges,
    tophat_response,
    blackhat_response,
)
from .matching import match_template_peaks, nms_by_iou, refine_match_local


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TemplateStampDetector:
    """Template-stamp detector with optional multi-instance full-frame matching.

    multi_instance=True: extract many peaks per scale, NMS, emit one Detection each.
    multi_instance=False: legacy single best hit (corner logos / one stamp).
    """

    label: str
    template_path: Path
    reference_width: int
    scale_factors: list[float]
    min_confidence: float = 0.22
    feature_kernel: int = 31
    feature_threshold: int = 12
    feature_mode: str = "fused"  # tophat | blackhat | laplacian | fused | adaptive
    feature_adaptive: bool = False  # percentile thresholds (pale watermarks)
    # bbox = solid rectangle (preferred for LaMa); template/footprint/envelope = legacy shapes
    output_mask_mode: str = "bbox"
    footprint_close_kernel: int = 31
    footprint_min_area: int = 200
    dilate: int = 0
    mask_expand_ratio: float = 0.0
    mask_expand_top_ratio: float | None = None
    mask_expand_bottom_ratio: float | None = None
    mask_expand_x_ratio: float | None = None
    edge_expand_ratio: float = 0.0
    search_regions: list[dict[str, float]] | None = None
    candidate_limit: int = 12
    min_hough_circles: int = 0
    min_watermark_color_ratio: float = 0.0
    min_bottom_ratio: float = 0.0
    prefer_larger_score_margin: float = 0.0
    hough_param2: int = 18
    # --- multi-instance (full-frame repeating units) ---
    multi_instance: bool = False
    max_instances: int = 64
    nms_iou: float = 0.35
    # slightly lower gate for secondary peaks vs the global best
    multi_score_ratio: float = 0.72
    # --- localization quality (reject tiny/huge FPs, refine after coarse match) ---
    # Absolute pixel floor for matched template side (also ratio-based below)
    min_match_side: int = 48
    # Match side must be within [min_side_ratio, max_side_ratio] of min(image_w, image_h)
    min_side_ratio: float = 0.04
    max_side_ratio: float = 0.48
    # Relative to expected size from reference_width (base_scale * template)
    min_scale_vs_expected: float = 0.55
    max_scale_vs_expected: float = 1.55
    # Local re-match around coarse peak
    refine_enabled: bool = True
    refine_pad_ratio: float = 0.40
    # Soft residual density gate (0 = off); pale stamps still pass ~0.08+
    min_residual_density: float = 0.06
    # NCC between residual map and template shape; rejects faces/hands that only
    # match edge energy but not stamp geometry. Typical true hits ~0.15–0.45.
    min_structure_alignment: float = 0.10
    # Below this match score, require stronger structure (avoid weak FPs)
    weak_score_threshold: float = 0.38
    weak_min_structure: float = 0.16
    # ORB inliers vs sample_crop; weak scores need this to reject faces/hands
    # Empirically: true NiceDay ≥8–400; faces/hands/shoes often 0–7
    min_orb_matches: int = 8
    # Scores at/above this skip the ORB gate (strong template peak)
    strong_score_threshold: float = 0.45
    # How many coarse candidates to fully refine+validate (single-instance)
    topk_validate: int = 28
    # If primary pass finds nothing: residual-hotspot second pass (same gates)
    secondary_search_enabled: bool = True
    # Also match on pure tophat map (helps pale white stamps; validation unchanged)
    dual_feature_search: bool = True
    # YOLO only proposes ROIs when primary+hotspot miss; never paints mask alone
    yolo_propose_enabled: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any], config_path: Path) -> "TemplateStampDetector":
        return cls(
            label=config["label"],
            template_path=resolve_config_path(config["template_path"], config_path),
            reference_width=int(config.get("reference_width", 2640)),
            scale_factors=[float(scale) for scale in config.get("scale_factors", [1.0])],
            min_confidence=float(config.get("min_confidence", 0.22)),
            feature_kernel=int(config.get("feature_kernel", 31)),
            feature_threshold=int(config.get("feature_threshold", 12)),
            feature_mode=str(config.get("feature_mode", "fused")),
            feature_adaptive=_as_bool(config.get("feature_adaptive"), False)
            or str(config.get("feature_mode", "")).lower() == "adaptive",
            output_mask_mode=str(config.get("output_mask_mode", "bbox") or "bbox").strip().lower(),
            footprint_close_kernel=int(config.get("footprint_close_kernel", 31)),
            footprint_min_area=int(config.get("footprint_min_area", 200)),
            dilate=int(config.get("dilate", 0)),
            mask_expand_ratio=float(config.get("mask_expand_ratio", 0.0)),
            mask_expand_top_ratio=_optional_float(config.get("mask_expand_top_ratio")),
            mask_expand_bottom_ratio=_optional_float(config.get("mask_expand_bottom_ratio")),
            mask_expand_x_ratio=_optional_float(config.get("mask_expand_x_ratio")),
            edge_expand_ratio=float(config.get("edge_expand_ratio", 0.0)),
            search_regions=config.get("search_regions"),
            candidate_limit=int(config.get("candidate_limit", 12)),
            min_hough_circles=int(config.get("min_hough_circles", 0)),
            min_watermark_color_ratio=float(config.get("min_watermark_color_ratio", 0.0)),
            min_bottom_ratio=float(config.get("min_bottom_ratio", 0.0)),
            prefer_larger_score_margin=float(config.get("prefer_larger_score_margin", 0.0)),
            hough_param2=int(config.get("hough_param2", 18)),
            multi_instance=_as_bool(config.get("multi_instance"), False),
            max_instances=int(config.get("max_instances", 64)),
            nms_iou=float(config.get("nms_iou", 0.35)),
            multi_score_ratio=float(config.get("multi_score_ratio", 0.72)),
            min_match_side=int(config.get("min_match_side", 48)),
            min_side_ratio=float(config.get("min_side_ratio", 0.04)),
            max_side_ratio=float(config.get("max_side_ratio", 0.48)),
            min_scale_vs_expected=float(config.get("min_scale_vs_expected", 0.55)),
            max_scale_vs_expected=float(config.get("max_scale_vs_expected", 1.55)),
            refine_enabled=_as_bool(config.get("refine_enabled"), True),
            refine_pad_ratio=float(config.get("refine_pad_ratio", 0.40)),
            min_residual_density=float(config.get("min_residual_density", 0.06)),
            min_structure_alignment=float(config.get("min_structure_alignment", 0.10)),
            weak_score_threshold=float(config.get("weak_score_threshold", 0.38)),
            weak_min_structure=float(config.get("weak_min_structure", 0.16)),
            min_orb_matches=int(config.get("min_orb_matches", 8)),
            strong_score_threshold=float(config.get("strong_score_threshold", 0.45)),
            topk_validate=int(config.get("topk_validate", 28)),
            secondary_search_enabled=_as_bool(config.get("secondary_search_enabled"), True),
            dual_feature_search=_as_bool(config.get("dual_feature_search"), True),
            yolo_propose_enabled=_as_bool(config.get("yolo_propose_enabled"), True),
        )

    def __post_init__(self) -> None:
        template = read_image(self.template_path)
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, self.template_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        self._feature_params = FeatureParams.from_detector_fields(
            feature_mode=self.feature_mode,
            feature_kernel=self.feature_kernel,
            feature_threshold=self.feature_threshold,
            feature_adaptive=self.feature_adaptive,
        )
        # Dense envelope masks need sample_crop edges; shared helper in features.py
        sample_path = self.template_path.parent / "sample_crop.png"
        self.sample_bgr = read_image(sample_path) if sample_path.exists() else None
        self.template_edges = build_template_feature_edges(
            self.sample_bgr, self.template_mask, self._feature_params
        )
        self.output_template_mask = self._build_output_template_mask()
        self.output_template_bounds = self._mask_bounds(self.output_template_mask)
        # Precompute sample feature edges for secondary verification
        if self.sample_bgr is not None:
            self.sample_edges = compute_feature_edges(self.sample_bgr, self._feature_params)
            sample_gray = cv2.cvtColor(self.sample_bgr, cv2.COLOR_BGR2GRAY)
            self._orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)
            self._sample_kp, self._sample_des = self._orb.detectAndCompute(sample_gray, None)
        else:
            self.sample_edges = self.template_edges
            self._orb = None
            self._sample_kp, self._sample_des = None, None
        # Optional YOLO box proposer (set by JobService when weights ready)
        self._yolo_proposer: Any = None

    def attach_yolo_proposer(self, proposer: Any | None) -> None:
        """Attach a YOLO detector used only as ROI proposer (must expose propose_boxes)."""
        self._yolo_proposer = proposer

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        base_scale = width / float(max(1, self.reference_width))
        expected_w = max(1.0, float(self.template_edges.shape[1]) * base_scale)
        expected_h = max(1.0, float(self.template_edges.shape[0]) * base_scale)

        # Single-instance: more peaks per scale so the true stamp is not buried
        peak_cap = self.max_instances if self.multi_instance else max(5, min(10, self.candidate_limit))
        regions = self.search_regions or [{"left": 0, "top": 0, "right": 1, "bottom": 1}]

        # Full-frame + large images: coarse pyramid first, then refine locally.
        # Nearby / small search regions keep full-res path (accuracy first).
        use_pyramid = self._should_use_pyramid_search(width, height, regions)
        if use_pyramid:
            candidates = self._pyramid_collect_candidates(
                image,
                regions,
                width=width,
                height=height,
                base_scale=base_scale,
                expected_w=expected_w,
                expected_h=expected_h,
                peak_cap=peak_cap,
            )
            # Features for validation / secondary — only full-res once
            target_edges = self._feature_edges(image)
        else:
            target_edges = self._feature_edges(image)
            # Collect slightly below min_confidence so pale true stamps enter the pool;
            # validation still enforces score / ORB / structure (accuracy preserved).
            collect_thr = max(0.18, float(self.min_confidence) * 0.90)
            candidates = self._collect_candidates(
                target_edges,
                regions,
                width=width,
                height=height,
                base_scale=base_scale,
                expected_w=expected_w,
                expected_h=expected_h,
                peak_cap=peak_cap,
                score_threshold=collect_thr,
                feature_tag="fused",
            )

            # Pale white stamps: dual feature only when not multi-instance and
            # search area is limited (nearby). Full-frame dual is too expensive.
            if self.dual_feature_search and not self.multi_instance:
                tophat_params = FeatureParams(
                    mode="tophat",
                    kernel=self._feature_params.kernel,
                    threshold=max(6, int(self._feature_params.threshold * 0.85)),
                    adaptive=True,
                )
                tophat_edges = compute_feature_edges(image, tophat_params)
                if self.sample_bgr is not None:
                    tpl_tophat = compute_feature_edges(self.sample_bgr, tophat_params)
                    tpl_tophat = cv2.bitwise_and(tpl_tophat, self.template_mask)
                else:
                    tpl_tophat = self.template_edges
                candidates.extend(
                    self._collect_candidates(
                        tophat_edges,
                        regions,
                        width=width,
                        height=height,
                        base_scale=base_scale,
                        expected_w=expected_w,
                        expected_h=expected_h,
                        peak_cap=max(3, peak_cap // 2),
                        score_threshold=collect_thr,
                        feature_tag="tophat",
                        template_edges_override=tpl_tophat,
                    )
                )

        if not candidates and not (
            self.secondary_search_enabled and not self.multi_instance
        ):
            return []

        candidates = self._dedupe_candidates(candidates)

        if self.multi_instance:
            candidates.sort(
                key=lambda item: float(item["score"]) * float(item.get("size_weight", 1.0)),
                reverse=True,
            )
            selected = self._select_multi_instances(
                image, candidates, width, height, target_edges=target_edges
            )
        else:
            # Per-channel top-K so tophat noise cannot starve fused true hits
            validate_pool = self._topk_validate_pool(candidates)
            best = self._best_valid_candidate(
                image,
                validate_pool,
                width,
                height,
                target_edges=target_edges,
            )
            search_pass = "primary"
            # Second pass: residual hotspots — only when primary found nothing.
            # Same validation gates → accuracy preserved; recall improves on pale/occluded stamps.
            if best is None and self.secondary_search_enabled:
                secondary = self._secondary_hotspot_candidates(
                    image,
                    target_edges,
                    width=width,
                    height=height,
                    base_scale=base_scale,
                    expected_w=expected_w,
                    expected_h=expected_h,
                )
                if secondary:
                    secondary = self._dedupe_candidates(secondary)
                    best = self._best_valid_candidate(
                        image,
                        self._topk_validate_pool(secondary),
                        width,
                        height,
                        target_edges=target_edges,
                    )
                    if best is not None:
                        search_pass = "secondary_hotspot"
            # Third pass: YOLO proposals → local template match → same gates.
            # Never emits YOLO box alone (prevents regression on already-good hits).
            if (
                best is None
                and self.yolo_propose_enabled
                and self._yolo_proposer is not None
                and not self.multi_instance
            ):
                yolo_cands = self._candidates_from_yolo_proposals(
                    image,
                    target_edges,
                    width=width,
                    height=height,
                    base_scale=base_scale,
                    expected_w=expected_w,
                    expected_h=expected_h,
                )
                if yolo_cands:
                    yolo_cands = self._dedupe_candidates(yolo_cands)
                    best = self._best_valid_candidate(
                        image,
                        self._topk_validate_pool(yolo_cands),
                        width,
                        height,
                        target_edges=target_edges,
                    )
                    if best is not None:
                        search_pass = "yolo_propose"
            if best is not None:
                best["search_pass"] = search_pass
            selected = [best] if best is not None else []

        detections: list[Detection] = []
        for index, hit in enumerate(selected):
            det = self._candidate_to_detection(
                image, hit, height, width, instance_index=index
            )
            if det is not None:
                detections.append(det)
        return detections

    def _search_area_ratio(self, regions: list[dict[str, float]]) -> float:
        """Union area of search regions as fraction of full frame (approx sum, capped)."""
        total = 0.0
        for r in regions or []:
            w = max(0.0, float(r.get("right", 1)) - float(r.get("left", 0)))
            h = max(0.0, float(r.get("bottom", 1)) - float(r.get("top", 0)))
            total += w * h
        return min(1.0, total)

    def _should_use_pyramid_search(
        self, width: int, height: int, regions: list[dict[str, float]]
    ) -> bool:
        """Use coarse→fine when search is large (full-frame / big images)."""
        area = float(width * height)
        ratio = self._search_area_ratio(regions)
        # Nearby match (small pad): keep full-res for accuracy
        if ratio < 0.35:
            return False
        # Full-ish search on medium+ images
        if ratio >= 0.85 and area >= 480 * 480:
            return True
        # Large images even with moderate regions
        if area >= 1280 * 720 and ratio >= 0.5:
            return True
        return False

    def _pyramid_collect_candidates(
        self,
        image: np.ndarray,
        regions: list[dict[str, float]],
        *,
        width: int,
        height: int,
        base_scale: float,
        expected_w: float,
        expected_h: float,
        peak_cap: int,
    ) -> list[dict[str, Any]]:
        """Coarse match on downscaled image, then local full-res refine.

        Preserves validation gates (caller still runs ORB/structure checks).
        Typical speedup: 4–10× on full-frame HD+ vs multi-scale full-res TM.
        """
        # Target long side ~720 for coarse pass
        long_side = max(width, height)
        if long_side <= 800:
            down = 1.5
        elif long_side <= 1400:
            down = 2.0
        else:
            down = min(3.0, long_side / 640.0)
        inv = 1.0 / down
        small_w = max(32, int(round(width * inv)))
        small_h = max(32, int(round(height * inv)))
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Fewer scales on coarse level
        scales = list(self.scale_factors or [1.0])
        if len(scales) > 4:
            # keep mid-range scales
            scales = sorted(float(s) for s in scales)
            step = max(1, len(scales) // 4)
            scales = scales[::step][:4]
        saved_scales = self.scale_factors
        self.scale_factors = scales

        collect_thr = max(0.16, float(self.min_confidence) * 0.85)
        try:
            small_edges = self._feature_edges(small)
            small_base = small_w / float(max(1, self.reference_width))
            # expected size on coarse image
            exp_w_s = max(1.0, float(self.template_edges.shape[1]) * small_base)
            exp_h_s = max(1.0, float(self.template_edges.shape[0]) * small_base)
            coarse = self._collect_candidates(
                small_edges,
                regions,
                width=small_w,
                height=small_h,
                base_scale=small_base,
                expected_w=exp_w_s,
                expected_h=exp_h_s,
                peak_cap=max(3, min(peak_cap, 6)),
                score_threshold=collect_thr,
                feature_tag="fused_coarse",
            )
        finally:
            self.scale_factors = saved_scales

        if not coarse:
            # Fallback: single-scale full-res fused (still better than full multi-scale dual)
            full_edges = self._feature_edges(image)
            mid_scales = sorted(float(s) for s in (self.scale_factors or [1.0]))
            if len(mid_scales) > 3:
                mid_scales = mid_scales[len(mid_scales) // 4 : -len(mid_scales) // 4 or None] or mid_scales
                mid_scales = mid_scales[:3]
            saved = self.scale_factors
            self.scale_factors = mid_scales or [1.0]
            try:
                return self._collect_candidates(
                    full_edges,
                    regions,
                    width=width,
                    height=height,
                    base_scale=base_scale,
                    expected_w=expected_w,
                    expected_h=expected_h,
                    peak_cap=peak_cap,
                    score_threshold=max(0.18, float(self.min_confidence) * 0.9),
                    feature_tag="fused",
                )
            finally:
                self.scale_factors = saved

        # Map coarse peaks → full-res local windows and re-match
        full_edges = self._feature_edges(image)
        refined: list[dict[str, Any]] = []
        # Rank coarse hits
        coarse.sort(
            key=lambda c: float(c["score"]) * float(c.get("size_weight", 1.0)),
            reverse=True,
        )
        max_refine = max(4, min(10, peak_cap * 2 if self.multi_instance else 6))
        pad_frac = 0.55  # local window margin relative to template size

        for cand in coarse[:max_refine]:
            lx, ly = cand["location"]
            tw, th = cand["size"]
            # upscale to full-res coords
            fx = int(round(lx * down))
            fy = int(round(ly * down))
            ftw = max(8, int(round(tw * down)))
            fth = max(8, int(round(th * down)))
            pad_x = max(16, int(ftw * pad_frac))
            pad_y = max(16, int(fth * pad_frac))
            x0 = max(0, fx - pad_x)
            y0 = max(0, fy - pad_y)
            x1 = min(width, fx + ftw + pad_x)
            y1 = min(height, fy + fth + pad_y)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            local_region = [
                {
                    "left": x0 / width,
                    "top": y0 / height,
                    "right": x1 / width,
                    "bottom": y1 / height,
                }
            ]
            # Local multi-scale around expected size only
            local_scales = [0.85, 1.0, 1.15]
            saved = self.scale_factors
            self.scale_factors = local_scales
            try:
                local_hits = self._collect_candidates(
                    full_edges,
                    local_region,
                    width=width,
                    height=height,
                    base_scale=base_scale,
                    expected_w=expected_w,
                    expected_h=expected_h,
                    peak_cap=3 if self.multi_instance else 2,
                    score_threshold=max(0.18, float(self.min_confidence) * 0.88),
                    feature_tag="fused",
                )
            finally:
                self.scale_factors = saved
            for h in local_hits:
                h["feature_tag"] = "fused"
                refined.append(h)

        if not refined:
            # Promote best coarse hit mapped to full-res as weak candidate
            for cand in coarse[: max(2, peak_cap)]:
                lx, ly = cand["location"]
                tw, th = cand["size"]
                refined.append(
                    {
                        "score": float(cand["score"]) * 0.95,
                        "location": (
                            int(round(lx * down)),
                            int(round(ly * down)),
                        ),
                        "size": (
                            max(8, int(round(tw * down))),
                            max(8, int(round(th * down))),
                        ),
                        "scale": float(cand.get("scale", base_scale)) * down,
                        "size_weight": float(cand.get("size_weight", 1.0)),
                        "feature_tag": "fused",
                    }
                )
        return refined

    def _collect_candidates(
        self,
        target_edges: np.ndarray,
        regions: list[dict[str, float]],
        *,
        width: int,
        height: int,
        base_scale: float,
        expected_w: float,
        expected_h: float,
        peak_cap: int,
        score_threshold: float,
        feature_tag: str,
        template_edges_override: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-scale matchTemplate peaks over search regions."""
        saved_tpl = self.template_edges
        if template_edges_override is not None:
            self.template_edges = template_edges_override
        try:
            out: list[dict[str, Any]] = []
            for region_config in regions:
                region = self._absolute_region(region_config, width, height)
                if region.width < 8 or region.height < 8:
                    continue
                roi_edges = target_edges[region.y : region.bottom, region.x : region.right]
                for scale_factor in self.scale_factors:
                    scale = base_scale * scale_factor
                    if not self._scale_allowed(scale, base_scale):
                        continue
                    peaks = self._match_scale_peaks(
                        roi_edges,
                        scale,
                        peak_cap=peak_cap,
                        image_wh=(width, height),
                        score_threshold=score_threshold,
                    )
                    for score, location, size in peaks:
                        absolute_location = (region.x + location[0], region.y + location[1])
                        if not self._size_allowed(
                            size[0], size[1], width, height, expected_w, expected_h
                        ):
                            continue
                        out.append(
                            {
                                "score": float(score),
                                "location": absolute_location,
                                "size": size,
                                "scale": float(scale),
                                "size_weight": self._size_weight(
                                    size[0], size[1], expected_w, expected_h
                                ),
                                "feature_tag": feature_tag,
                            }
                        )
            return out
        finally:
            self.template_edges = saved_tpl

    def _dedupe_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop near-duplicate peaks within each feature channel.

        NMS is **not** cross-feature: a high-score tophat FP must not suppress a
        slightly lower fused true hit at the same place (that caused recall regressions).
        """
        if len(candidates) <= 1:
            return candidates
        by_tag: dict[str, list[dict[str, Any]]] = {}
        for cand in candidates:
            tag = str(cand.get("feature_tag") or "fused")
            by_tag.setdefault(tag, []).append(cand)
        out: list[dict[str, Any]] = []
        per_tag = max(24, int(self.topk_validate))
        for group in by_tag.values():
            out.extend(nms_by_iou(group, iou_threshold=0.45, max_keep=per_tag))
        return out

    def _topk_validate_pool(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build validation pool with fair quota per feature channel."""
        if not candidates:
            return []
        by_tag: dict[str, list[dict[str, Any]]] = {}
        for cand in candidates:
            tag = str(cand.get("feature_tag") or "fused")
            by_tag.setdefault(tag, []).append(cand)
        pool: list[dict[str, Any]] = []
        base_n = max(int(self.topk_validate), int(self.candidate_limit), 16)
        for tag, group in by_tag.items():
            group = sorted(
                group,
                key=lambda item: float(item["score"]) * float(item.get("size_weight", 1.0)),
                reverse=True,
            )
            # fused gets full quota; auxiliary channels (tophat/hotspot) get half
            n = base_n if tag == "fused" else max(12, base_n // 2)
            pool.extend(group[:n])
        # Stable unique order for validation: rank score first
        pool.sort(
            key=lambda item: float(item["score"]) * float(item.get("size_weight", 1.0)),
            reverse=True,
        )
        return pool

    def _secondary_hotspot_candidates(
        self,
        image: np.ndarray,
        target_edges: np.ndarray,
        *,
        width: int,
        height: int,
        base_scale: float,
        expected_w: float,
        expected_h: float,
    ) -> list[dict[str, Any]]:
        """Propose ROIs from pale/dark residual blobs, then re-match locally.

        Does not relax validation — only changes *where* we look when global peak failed.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        k = max(15, int(self.feature_kernel) | 1)
        residual = cv2.max(tophat_response(gray, k), blackhat_response(gray, k))
        if residual.size == 0:
            return []
        # Adaptive bright residual mask
        nz = residual[residual > 0]
        if nz.size < 64:
            return []
        thr = float(np.percentile(nz, 85.0))
        thr = max(thr * 0.42, float(np.percentile(nz, 68.0)) * 0.30, 2.5)
        _, binary = cv2.threshold(residual, thr, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        # Connect broken stamp strokes
        side = max(3, int(round(min(expected_w, expected_h) * 0.12)))
        if side % 2 == 0:
            side += 1
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)
        binary = cv2.dilate(binary, close_k, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        exp_area = max(1.0, expected_w * expected_h)
        blobs: list[tuple[float, BBox]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area < exp_area * 0.08 or area > exp_area * 6.5:
                continue
            # Aspect roughly stamp-like (not thin lines)
            aspect = w / max(1.0, float(h))
            if aspect < 0.35 or aspect > 2.8:
                continue
            # Residual mass inside box
            mass = float(np.count_nonzero(binary[y : y + h, x : x + w]))
            if mass < 16:
                continue
            blobs.append((mass, BBox(x, y, w, h)))
        if not blobs:
            return []
        blobs.sort(key=lambda t: t[0], reverse=True)
        blobs = blobs[:16]

        # Slightly softer peak floor only for *collecting* candidates in hotspots
        soft_thr = max(0.15, float(self.min_confidence) * 0.78)
        candidates: list[dict[str, Any]] = []
        for _mass, box in blobs:
            pad_x = int(round(max(box.width, expected_w) * 0.55))
            pad_y = int(round(max(box.height, expected_h) * 0.55))
            x0 = max(0, box.x - pad_x)
            y0 = max(0, box.y - pad_y)
            x1 = min(width, box.right + pad_x)
            y1 = min(height, box.bottom + pad_y)
            region = {
                "left": x0 / width,
                "top": y0 / height,
                "right": x1 / width,
                "bottom": y1 / height,
            }
            # Focus scales near expected size (±40%)
            local_scales = []
            for sf in self.scale_factors:
                scale = base_scale * sf
                tw = self.template_edges.shape[1] * scale
                if expected_w * 0.55 <= tw <= expected_w * 1.55:
                    local_scales.append(sf)
            if not local_scales:
                local_scales = list(self.scale_factors)

            saved_scales = self.scale_factors
            self.scale_factors = local_scales
            try:
                candidates.extend(
                    self._collect_candidates(
                        target_edges,
                        [region],
                        width=width,
                        height=height,
                        base_scale=base_scale,
                        expected_w=expected_w,
                        expected_h=expected_h,
                        peak_cap=4,
                        score_threshold=soft_thr,
                        feature_tag="hotspot",
                    )
                )
            finally:
                self.scale_factors = saved_scales
        return candidates

    def _candidates_from_yolo_proposals(
        self,
        image: np.ndarray,
        target_edges: np.ndarray,
        *,
        width: int,
        height: int,
        base_scale: float,
        expected_w: float,
        expected_h: float,
    ) -> list[dict[str, Any]]:
        """Turn YOLO boxes into template-match candidates (still need full validation)."""
        proposer = self._yolo_proposer
        if proposer is None or not hasattr(proposer, "propose_boxes"):
            return []
        try:
            proposals = proposer.propose_boxes(image) or []
        except Exception:  # noqa: BLE001 — YOLO must never break style path
            return []
        if not proposals:
            return []

        soft_thr = max(0.15, float(self.min_confidence) * 0.78)
        candidates: list[dict[str, Any]] = []
        # Prefer mid scales around expected stamp size
        local_scales = []
        for sf in self.scale_factors:
            scale = base_scale * sf
            tw = self.template_edges.shape[1] * scale
            if expected_w * 0.50 <= tw <= expected_w * 1.65:
                local_scales.append(sf)
        if not local_scales:
            local_scales = list(self.scale_factors)

        for prop in proposals[:16]:
            box = prop.get("bbox")
            if box is None:
                continue
            if not isinstance(box, BBox):
                try:
                    x, y, w, h = box  # type: ignore[misc]
                    box = BBox(int(x), int(y), int(w), int(h))
                except Exception:  # noqa: BLE001
                    continue
            # Pad YOLO box so multi-scale template match can lock on
            pad_x = int(round(max(box.width, expected_w) * 0.65))
            pad_y = int(round(max(box.height, expected_h) * 0.65))
            x0 = max(0, box.x - pad_x)
            y0 = max(0, box.y - pad_y)
            x1 = min(width, box.right + pad_x)
            y1 = min(height, box.bottom + pad_y)
            region = {
                "left": x0 / max(1, width),
                "top": y0 / max(1, height),
                "right": x1 / max(1, width),
                "bottom": y1 / max(1, height),
            }
            saved_scales = self.scale_factors
            self.scale_factors = local_scales
            try:
                local = self._collect_candidates(
                    target_edges,
                    [region],
                    width=width,
                    height=height,
                    base_scale=base_scale,
                    expected_w=expected_w,
                    expected_h=expected_h,
                    peak_cap=5,
                    score_threshold=soft_thr,
                    feature_tag="yolo_propose",
                )
            finally:
                self.scale_factors = saved_scales
            for cand in local:
                cand["yolo_conf"] = float(prop.get("confidence") or 0.0)
                candidates.append(cand)

            # Also seed a size-normalized candidate at YOLO center with expected scale
            # (helps when matchTemplate peak is weak but geometry is right)
            cx = box.x + box.width // 2
            cy = box.y + box.height // 2
            tw = max(8, int(round(expected_w)))
            th = max(8, int(round(expected_h)))
            lx = max(0, cx - tw // 2)
            ly = max(0, cy - th // 2)
            if self._size_allowed(tw, th, width, height, expected_w, expected_h):
                # Score via one-shot match at this location (not free pass)
                seed_score = self._score_at_location(
                    target_edges, lx, ly, tw, th, width, height
                )
                if seed_score >= soft_thr:
                    candidates.append(
                        {
                            "score": float(seed_score),
                            "location": (lx, ly),
                            "size": (tw, th),
                            "scale": float(base_scale),
                            "size_weight": self._size_weight(tw, th, expected_w, expected_h),
                            "feature_tag": "yolo_seed",
                            "yolo_conf": float(prop.get("confidence") or 0.0),
                        }
                    )
        return candidates

    def _score_at_location(
        self,
        target_edges: np.ndarray,
        x: int,
        y: int,
        tw: int,
        th: int,
        width: int,
        height: int,
    ) -> float:
        """TM_CCOEFF_NORMED of template at a fixed top-left (single cell)."""
        if tw < 8 or th < 8:
            return 0.0
        if x < 0 or y < 0 or x + tw > width or y + th > height:
            return 0.0
        patch = target_edges[y : y + th, x : x + tw]
        if patch.shape[0] != th or patch.shape[1] != tw:
            return 0.0
        tpl = cv2.resize(self.template_edges, (tw, th), interpolation=cv2.INTER_AREA)
        if np.count_nonzero(tpl) == 0 or np.count_nonzero(patch) == 0:
            return 0.0
        # Same-size NCC (faster than matchTemplate for one cell)
        a = patch.astype(np.float32)
        b = tpl.astype(np.float32)
        a_std = float(a.std())
        b_std = float(b.std())
        if a_std < 1e-3 or b_std < 1e-3:
            return 0.0
        a = (a - float(a.mean())) / a_std
        b = (b - float(b.mean())) / b_std
        # Map NCC [-1,1] → [0,1] so thresholds match TM_CCOEFF_NORMED scale
        return float(np.clip((float((a * b).mean()) + 1.0) * 0.5, 0.0, 1.0))

    def _scale_allowed(self, scale: float, base_scale: float) -> bool:
        """Reject absolute scales far from the reference-size expectation."""
        if base_scale <= 0:
            return True
        # scale / base_scale == scale_factor; also guard absolute tiny scales
        rel = scale / base_scale
        if rel < self.min_scale_vs_expected * 0.85 or rel > self.max_scale_vs_expected * 1.15:
            return False
        return True

    def _min_allowed_side(
        self,
        image_w: int,
        image_h: int,
        expected_w: float,
        expected_h: float,
    ) -> int:
        """Adaptive pixel floor: absolute config, but never larger than ~expected size.

        Hard-coding 48px rejected legitimate small templates (unit tests / small tiles).
        Real NiceDay stamps have expected side ≫ 48 after reference scaling, so tiny
        foot-floor FPs (~50px while expected ~200+) still fail via expected-ratio gate.
        """
        img_min = max(1, min(int(image_w), int(image_h)))
        ratio_floor = int(round(img_min * float(self.min_side_ratio)))
        expected_side = min(float(expected_w), float(expected_h))
        # Allow down to min_scale_vs_expected of the reference-scaled template
        expected_floor = int(round(expected_side * float(self.min_scale_vs_expected))) if expected_side > 1 else 0
        absolute = int(self.min_match_side)
        # min(absolute, max(expected_floor, ratio_floor)) keeps small real templates alive
        # while still requiring at least 16px of structure
        floor = max(16, min(absolute, max(expected_floor, ratio_floor, 16)))
        if self.multi_instance:
            floor = max(12, floor // 2)
        return int(floor)

    def _size_allowed(
        self,
        match_w: int,
        match_h: int,
        image_w: int,
        image_h: int,
        expected_w: float,
        expected_h: float,
    ) -> bool:
        """Hard size gates: kill extreme tiny/huge FPs relative to expected stamp size."""
        side = min(int(match_w), int(match_h))
        long_side = max(int(match_w), int(match_h))
        img_min = max(1, min(int(image_w), int(image_h)))
        min_side = self._min_allowed_side(image_w, image_h, expected_w, expected_h)
        max_side = max(min_side + 1, int(round(img_min * self.max_side_ratio)))
        if side < min_side or long_side > max_side:
            return False
        # Relative to expected reference size (primary FP killer for wrong scales)
        if expected_w > 1 and expected_h > 1:
            wr = float(match_w) / expected_w
            hr = float(match_h) / expected_h
            if wr < self.min_scale_vs_expected or hr < self.min_scale_vs_expected:
                return False
            if wr > self.max_scale_vs_expected or hr > self.max_scale_vs_expected:
                return False
        return True

    def _size_weight(
        self,
        match_w: int,
        match_h: int,
        expected_w: float,
        expected_h: float,
    ) -> float:
        """Soft prior: prefer sizes near the reference-scaled template."""
        if expected_w <= 1 or expected_h <= 1:
            return 1.0
        ratio = 0.5 * (float(match_w) / expected_w + float(match_h) / expected_h)
        # Peak at 1.0; gentle falloff
        if 0.75 <= ratio <= 1.30:
            return 1.15
        if 0.60 <= ratio <= 1.50:
            return 1.0
        if 0.50 <= ratio <= 1.70:
            return 0.72
        return 0.40

    def _match_scale_peaks(
        self,
        roi_edges: np.ndarray,
        scale: float,
        *,
        peak_cap: int,
        image_wh: tuple[int, int] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[float, tuple[int, int], tuple[int, int]]]:
        template_width = max(1, int(round(self.template_edges.shape[1] * scale)))
        template_height = max(1, int(round(self.template_edges.shape[0] * scale)))
        if template_width > roi_edges.shape[1] or template_height > roi_edges.shape[0]:
            return []
        # Adaptive floor from expected reference size (see _min_allowed_side)
        if image_wh is not None:
            base_scale = image_wh[0] / float(max(1, self.reference_width))
            expected_w = max(1.0, float(self.template_edges.shape[1]) * base_scale)
            expected_h = max(1.0, float(self.template_edges.shape[0]) * base_scale)
            min_side = self._min_allowed_side(image_wh[0], image_wh[1], expected_w, expected_h)
        else:
            min_side = 12 if self.multi_instance else 24
        if template_width < min_side or template_height < min_side:
            return []

        resized_template = cv2.resize(
            self.template_edges,
            (template_width, template_height),
            interpolation=cv2.INTER_AREA,
        )
        thr = float(self.min_confidence if score_threshold is None else score_threshold)
        return match_template_peaks(
            roi_edges,
            resized_template,
            threshold=thr,
            max_peaks=max(1, peak_cap),
            multi_instance=True if peak_cap > 1 else self.multi_instance,
        )

    def _select_multi_instances(
        self,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        width: int,
        height: int,
        *,
        target_edges: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Filter by relative score, validate, then NMS → up to max_instances."""
        if not candidates:
            return []
        best_score = float(candidates[0]["score"])
        # Secondary peaks must be both above absolute min_confidence and close to best
        floor = max(self.min_confidence, best_score * self.multi_score_ratio)
        pool = [c for c in candidates if float(c["score"]) >= floor]
        pool = pool[: max(self.candidate_limit, self.max_instances * 2)]

        validated: list[dict[str, Any]] = []
        for candidate in pool:
            refined = self._maybe_refine(candidate, target_edges, width, height)
            ok = self._validate_candidate(image, refined, width, height)
            if ok is not None:
                validated.append(ok)

        kept = nms_by_iou(
            validated,
            iou_threshold=self.nms_iou,
            max_keep=self.max_instances,
        )
        # Drop weak leftovers after NMS (scale clutter often survives at 0.4 while best is 0.8)
        if kept:
            top = float(kept[0]["score"])
            kept = [c for c in kept if float(c["score"]) >= max(self.min_confidence, top * self.multi_score_ratio)]
        return kept

    def _maybe_refine(
        self,
        candidate: dict[str, Any],
        target_edges: np.ndarray | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Local re-match for tighter box; keeps coarse hit if refine fails."""
        if not self.refine_enabled or target_edges is None:
            return candidate
        x, y = candidate["location"]
        w, h = candidate["size"]
        base_scale = width / float(max(1, self.reference_width))
        expected_w = max(1.0, float(self.template_edges.shape[1]) * base_scale)
        expected_h = max(1.0, float(self.template_edges.shape[0]) * base_scale)
        refine_min_side = max(
            12,
            self._min_allowed_side(width, height, expected_w, expected_h) // (2 if self.multi_instance else 1),
        )
        refined = refine_match_local(
            target_edges,
            self.template_edges,
            location=(int(x), int(y)),
            size=(int(w), int(h)),
            scale=float(candidate["scale"]),
            pad_ratio=self.refine_pad_ratio,
            min_score=max(0.18, self.min_confidence * 0.85),
            min_side=refine_min_side,
        )
        if refined is None:
            out = dict(candidate)
            out["refined"] = False
            return out
        score, loc, size, scale = refined
        # Accept refine if score is not worse than ~3% below coarse (prefer tighter geometry)
        if float(score) + 0.02 < float(candidate["score"]) * 0.97:
            out = dict(candidate)
            out["refined"] = False
            return out
        if not self._size_allowed(size[0], size[1], width, height, expected_w, expected_h):
            out = dict(candidate)
            out["refined"] = False
            return out
        out = dict(candidate)
        out["score"] = float(score)
        out["location"] = loc
        out["size"] = size
        out["scale"] = float(scale)
        out["size_weight"] = self._size_weight(size[0], size[1], expected_w, expected_h)
        out["refined"] = True
        return out

    def _validate_candidate(
        self,
        image: np.ndarray,
        candidate: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any] | None:
        bbox = BBox(
            candidate["location"][0],
            candidate["location"][1],
            candidate["size"][0],
            candidate["size"][1],
        ).clamp(width, height)
        if bbox.width <= 0 or bbox.height <= 0:
            return None
        if bbox.bottom / height < self.min_bottom_ratio:
            return None

        # Size gate again after refine / clamp
        base_scale = width / float(max(1, self.reference_width))
        expected_w = max(1.0, float(self.template_edges.shape[1]) * base_scale)
        expected_h = max(1.0, float(self.template_edges.shape[0]) * base_scale)
        if not self._size_allowed(bbox.width, bbox.height, width, height, expected_w, expected_h):
            return None

        hough_circles = self._count_hough_circles(image, bbox)
        watermark_color_ratio = self._watermark_color_ratio(image, bbox)
        residual_density = self._residual_density(image, bbox)
        structure = self._structure_alignment(image, bbox)
        sample_score = self._sample_verify_score(image, bbox)
        if hough_circles < self.min_hough_circles:
            return None
        if watermark_color_ratio < self.min_watermark_color_ratio:
            return None
        # Residual / structure gates: sparse stamp masks need geometry checks;
        # solid disks (fill≥0.55) skip structure (residual is often uniform).
        tpl_fill = float(np.count_nonzero(self.template_mask > 10)) / max(1, self.template_mask.size)
        solid_template = tpl_fill >= 0.55
        residual_gate = float(self.min_residual_density)
        if solid_template:
            residual_gate = min(residual_gate, 0.02)
        if residual_gate > 0 and residual_density < residual_gate:
            return None

        score = float(candidate["score"])
        orb_matches = self._orb_match_count(image, bbox)
        if not solid_template:
            # Structure alignment: faces/hands often beat pale stamps on raw score alone
            structure_gate = float(self.min_structure_alignment)
            if structure_gate > 0 and structure < structure_gate:
                return None
            if score < float(self.weak_score_threshold) and structure < float(self.weak_min_structure):
                return None
            # Sample-crop secondary check for weak scores (when sample exists)
            if (
                self.sample_bgr is not None
                and score < float(self.weak_score_threshold)
                and sample_score < 0.22
                and structure < 0.22
            ):
                return None
            # ORB vs sample: decisive for pale stamps vs faces/hands/mesh.
            # Keep hard threshold — soft ORB (orb=7) was accepting hair FPs.
            if (
                self.min_orb_matches > 0
                and self._sample_des is not None
                and score < float(self.strong_score_threshold)
                and orb_matches >= 0
                and orb_matches < int(self.min_orb_matches)
            ):
                return None
            # Extra-weak peaks: fabric/skin can get mid ORB without stamp geometry.
            # True low-score stamps (e.g. conference c≈0.26) usually have ORB≥35 or structure≥0.30.
            if (
                score < 0.30
                and orb_matches >= 0
                and orb_matches < 35
                and structure < 0.30
            ):
                return None
            if (
                score < 0.32
                and orb_matches >= 0
                and orb_matches < 25
                and structure < 0.28
            ):
                return None
            # Blurry background patches can accumulate spurious ORB matches.
            sharpness = self._patch_sharpness(image, bbox)
            if score < 0.36 and sharpness < 18.0 and structure < 0.30:
                return None
            # Tophat auxiliary channel: stricter gates (shoe/fabric FPs score well on tophat alone)
            feat = str(candidate.get("feature_tag") or "fused")
            if feat in {"tophat", "yolo_propose", "yolo_seed"} and score < float(self.strong_score_threshold):
                if (
                    score < 0.36
                    or structure < 0.36
                    or orb_matches < max(int(self.min_orb_matches) + 8, 16)
                ):
                    return None
            # Mid-score fused hair/neck FPs: need stronger ORB or structure than weak true stamps
            # True weak stamps (luggage/conference) usually have ORB≥25 or structure≥0.35
            if (
                feat == "fused"
                and score < 0.38
                and score >= 0.30
                and orb_matches >= 0
                and orb_matches < 18
                and structure < 0.34
            ):
                return None
        else:
            sharpness = self._patch_sharpness(image, bbox)

        out = dict(candidate)
        out["hough_circles"] = hough_circles
        out["watermark_color_ratio"] = watermark_color_ratio
        out["residual_density"] = residual_density
        out["structure"] = structure
        out["sample_score"] = sample_score
        out["orb_matches"] = orb_matches
        out["sharpness"] = sharpness
        out["area"] = bbox.width * bbox.height
        # Rank: match score × size × structure × sample × ORB boost × sharpness
        orb_term = 1.0
        if orb_matches >= 0:
            orb_term = 0.65 + 0.35 * float(np.clip(orb_matches / 40.0, 0.0, 1.0))
        sharp_term = 0.85 + 0.15 * float(np.clip(sharpness / 80.0, 0.0, 1.0))
        out["rank"] = (
            score
            * float(out.get("size_weight", 1.0))
            * (0.45 + 0.55 * float(np.clip((structure + 1.0) * 0.5, 0.0, 1.0)))
            * (0.70 + 0.30 * float(np.clip(sample_score, 0.0, 1.0)))
            * (0.80 + 0.20 * float(np.clip(watermark_color_ratio, 0.0, 1.0)))
            * orb_term
            * sharp_term
        )
        return out

    def _patch_sharpness(self, image: np.ndarray, bbox: BBox) -> float:
        """Laplacian variance inside the box — low on out-of-focus FPs."""
        patch = image[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if patch.size == 0:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _residual_density(self, image: np.ndarray, bbox: BBox) -> float:
        """Fraction of watermark-like residual pixels inside the match box."""
        residual, _tpl = self._box_residual_and_template(image, bbox)
        if residual is None:
            return 0.0
        region = _tpl > 10
        region_px = int(np.count_nonzero(region))
        if region_px < 16:
            return float(np.count_nonzero(residual > 0) / max(1, residual.size))
        return float(np.count_nonzero((residual > 0) & region) / region_px)

    def _box_residual_and_template(
        self, image: np.ndarray, bbox: BBox
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        patch = image[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if patch.size == 0:
            return None, None
        soft = FeatureParams(
            mode=self._feature_params.mode,
            kernel=self._feature_params.kernel,
            threshold=max(6, int(self._feature_params.threshold * 0.7)),
            adaptive=True,
        )
        residual = compute_feature_edges(patch, soft)
        tpl = cv2.resize(
            self.template_mask,
            (bbox.width, bbox.height),
            interpolation=cv2.INTER_AREA,
        )
        return residual, tpl

    def _structure_alignment(self, image: np.ndarray, bbox: BBox) -> float:
        """NCC of residual map vs template shape — true stamps align, faces/hands don't."""
        residual, tpl = self._box_residual_and_template(image, bbox)
        if residual is None or tpl is None or residual.size < 64:
            return 0.0
        r = residual.astype(np.float32)
        t = tpl.astype(np.float32)
        r_std = float(r.std())
        t_std = float(t.std())
        if r_std < 1e-3 or t_std < 1e-3:
            return 0.0
        r = (r - float(r.mean())) / r_std
        t = (t - float(t.mean())) / t_std
        return float(np.clip((r * t).mean(), -1.0, 1.0))

    def _sample_verify_score(self, image: np.ndarray, bbox: BBox) -> float:
        """Secondary TM score: sample feature edges vs in-box edges (same size)."""
        if self.sample_edges is None or self.sample_edges.size == 0:
            return 0.5
        patch = image[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if patch.size == 0 or bbox.width < 12 or bbox.height < 12:
            return 0.0
        box_edges = compute_feature_edges(patch, self._feature_params)
        sample = cv2.resize(
            self.sample_edges,
            (bbox.width, bbox.height),
            interpolation=cv2.INTER_AREA,
        )
        if np.count_nonzero(sample) == 0 or np.count_nonzero(box_edges) == 0:
            return 0.0
        # Same-size normalized correlation (not matchTemplate sliding)
        a = box_edges.astype(np.float32)
        b = sample.astype(np.float32)
        a_std = float(a.std())
        b_std = float(b.std())
        if a_std < 1e-3 or b_std < 1e-3:
            return 0.0
        a = (a - float(a.mean())) / a_std
        b = (b - float(b.mean())) / b_std
        # Map NCC [-1,1] → [0,1]
        return float(np.clip((float((a * b).mean()) + 1.0) * 0.5, 0.0, 1.0))

    def _orb_match_count(self, image: np.ndarray, bbox: BBox) -> int:
        """Good ORB matches between sample_crop and the candidate box.

        Returns -1 when ORB is unavailable (no sample / no descriptors).
        True NiceDay stamps typically yield 8–300 inliers; faces/hands ≤4.
        """
        if self._orb is None or self._sample_des is None or self._sample_kp is None:
            return -1
        if len(self._sample_kp) < 8:
            return -1
        # Slight pad so partial stamps still get keypoints
        pad = int(round(max(bbox.width, bbox.height) * 0.12))
        height, width = image.shape[:2]
        x0 = max(0, bbox.x - pad)
        y0 = max(0, bbox.y - pad)
        x1 = min(width, bbox.right + pad)
        y1 = min(height, bbox.bottom + pad)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0 or patch.shape[0] < 16 or patch.shape[1] < 16:
            return 0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Mild contrast stretch helps pale overlays
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des2 is None or kp2 is None or len(kp2) < 4:
            return 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            pairs = matcher.knnMatch(self._sample_des, des2, k=2)
        except cv2.error:
            return 0
        good = 0
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < 0.75 * n.distance:
                good += 1
        return int(good)

    def _candidate_to_detection(
        self,
        image: np.ndarray,
        hit: dict[str, Any],
        height: int,
        width: int,
        *,
        instance_index: int,
    ) -> Detection | None:
        x, y = hit["location"]
        match_width, match_height = hit["size"]
        raw_bbox = self._expanded_output_bbox(x, y, match_width, match_height, width, height)
        bbox = raw_bbox.clamp(width, height)
        if bbox.width <= 0 or bbox.height <= 0:
            return None

        mode = (self.output_mask_mode or "bbox").strip().lower()
        # Product default: solid rectangle covers the hit for LaMa (not stamp contour).
        if mode in {"bbox", "box", "rect", "rectangle", "filled"}:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
            # Ensure complete coverage even with tiny dilate configs
            dilate_px = max(int(self.dilate), 2)
            mask = dilate_mask(mask, dilate_px)
            mask_refine = "bbox_fill"
        else:
            patch = cv2.resize(
                self.output_template_mask,
                (raw_bbox.width, raw_bbox.height),
                interpolation=cv2.INTER_AREA,
            )
            _, patch = cv2.threshold(patch, 10, 255, cv2.THRESH_BINARY)
            patch_x = bbox.x - raw_bbox.x
            patch_y = bbox.y - raw_bbox.y
            patch = patch[patch_y : patch_y + bbox.height, patch_x : patch_x + bbox.width]
            if mode == "envelope" and self._touches_image_edge(bbox, width, height):
                patch = cv2.bitwise_or(
                    patch, self._build_edge_envelope_patch(bbox.width, bbox.height)
                )
            # Legacy contour modes only: optional residual trim
            patch, mask_refine = self._refine_patch_with_residual(image, bbox, patch)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = np.maximum(
                mask[bbox.y : bbox.bottom, bbox.x : bbox.right],
                patch,
            )
            mask = dilate_mask(mask, self.dilate)

        return Detection(
            label=self.label,
            bbox=bbox,
            confidence=float(hit["score"]),
            mask=mask,
            metadata={
                "detector": "template_stamp",
                "feature_mode": self.feature_mode,
                "feature_adaptive": self.feature_adaptive,
                "output_mask_mode": mode,
                "mask_refine": mask_refine,
                "scale": round(float(hit["scale"]), 4),
                "mask_expand_ratio": round(float(self.mask_expand_ratio), 4),
                "edge_expand_ratio": round(float(self.edge_expand_ratio), 4),
                "template": str(self.template_path),
                "hough_circles": hit.get("hough_circles", 0),
                "watermark_color_ratio": round(float(hit.get("watermark_color_ratio", 0.0)), 4),
                "residual_density": round(float(hit.get("residual_density", 0.0)), 4),
                "structure": round(float(hit.get("structure", 0.0)), 4),
                "sample_score": round(float(hit.get("sample_score", 0.0)), 4),
                "orb_matches": int(hit.get("orb_matches", -1) or -1),
                "sharpness": round(float(hit.get("sharpness", 0.0)), 2),
                "refined": bool(hit.get("refined", False)),
                "size_weight": round(float(hit.get("size_weight", 1.0)), 4),
                "feature_tag": str(hit.get("feature_tag") or "fused"),
                "search_pass": str(hit.get("search_pass") or "primary"),
                "multi_instance": self.multi_instance,
                "instance_index": instance_index,
            },
        )

    def _refine_patch_with_residual(
        self,
        image: np.ndarray,
        bbox: BBox,
        patch: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Intersect template patch with in-box residual; fall back if too empty.

        Skeleton = style template (recognizes shape). Residual trims background
        that the scaled template incorrectly covers. If intersection is too
        sparse (pale/low-contrast), keep pure template so LaMa still gets coverage.
        """
        if patch.size == 0 or int(np.count_nonzero(patch)) == 0:
            return patch, "template_empty"
        crop = image[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if crop.size == 0:
            return patch, "template"
        # Slightly softer thresholds so intersection does not erase thin strokes
        soft = FeatureParams(
            mode=self._feature_params.mode,
            kernel=self._feature_params.kernel,
            threshold=max(6, int(self._feature_params.threshold * 0.75)),
            adaptive=True,
        )
        residual = compute_feature_edges(crop, soft)
        if residual.shape[:2] != patch.shape[:2]:
            residual = cv2.resize(
                residual,
                (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        _, residual_bin = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        # Dilate residual so thin watermark strokes still intersect template
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        residual_bin = cv2.dilate(residual_bin, kernel, iterations=1)
        refined = cv2.bitwise_and(patch, residual_bin)
        tpl_px = int(np.count_nonzero(patch))
        ref_px = int(np.count_nonzero(refined))
        # Keep refine only if enough mass remains
        if ref_px >= max(32, int(tpl_px * 0.12)):
            return refined, "template_x_residual"
        return patch, "template_fallback"

    def _match_scale(
        self, roi_edges: np.ndarray, scale: float
    ) -> tuple[float, tuple[int, int], tuple[int, int]] | None:
        """Legacy single-peak API (tests / external)."""
        peaks = self._match_scale_peaks(roi_edges, scale, peak_cap=1)
        return peaks[0] if peaks else None

    def _feature_edges(self, image: np.ndarray) -> np.ndarray:
        return compute_feature_edges(image, self._feature_params)

    def _absolute_region(self, region: dict[str, float], width: int, height: int) -> BBox:
        left = int(round(width * float(region.get("left", 0))))
        top = int(round(height * float(region.get("top", 0))))
        right = int(round(width * float(region.get("right", 1))))
        bottom = int(round(height * float(region.get("bottom", 1))))
        return BBox(left, top, right - left, bottom - top).clamp(width, height)

    def _expanded_output_bbox(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
    ) -> BBox:
        x_ratio = self.mask_expand_x_ratio if self.mask_expand_x_ratio is not None else self.mask_expand_ratio
        top_ratio = self.mask_expand_top_ratio if self.mask_expand_top_ratio is not None else self.mask_expand_ratio
        bottom_ratio = (
            self.mask_expand_bottom_ratio if self.mask_expand_bottom_ratio is not None else self.mask_expand_ratio
        )
        pad_x = int(round(width * x_ratio))
        pad_top = int(round(height * top_ratio))
        pad_bottom = int(round(height * bottom_ratio))
        # Pixel floor so slightly undersized matches still fully cover the stamp
        min_pad = max(4, int(round(min(width, height) * 0.06)))
        pad_x = max(pad_x, min_pad)
        pad_top = max(pad_top, min_pad)
        pad_bottom = max(pad_bottom, min_pad)
        left = x - pad_x
        top = y - pad_top
        output_width = width + 2 * pad_x
        output_height = height + pad_top + pad_bottom

        template_height, template_width = self.output_template_mask.shape[:2]
        content_left, content_top, content_width, content_height = self.output_template_bounds.to_list()
        content_right_margin = template_width - content_left - content_width
        content_bottom_margin = template_height - content_top - content_height
        left_margin = int(round(output_width * content_left / template_width))
        right_margin = int(round(output_width * content_right_margin / template_width))
        top_margin = int(round(output_height * content_top / template_height))
        bottom_margin = int(round(output_height * content_bottom_margin / template_height))
        edge_expand_x = int(round(width * self.edge_expand_ratio))
        edge_expand_y = int(round(height * self.edge_expand_ratio))

        if x <= left_margin:
            left -= left_margin
            output_width += left_margin + edge_expand_x
        if image_width - (x + width) <= right_margin:
            left -= edge_expand_x
            output_width += right_margin
        if y <= top_margin:
            top -= top_margin
            output_height += top_margin + edge_expand_y
        if image_height - (y + height) <= bottom_margin:
            top -= edge_expand_y
            output_height += bottom_margin
        return BBox(left, top, output_width, output_height)

    def _mask_bounds(self, mask: np.ndarray) -> BBox:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return BBox(0, 0, mask.shape[1], mask.shape[0])
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return BBox(x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    def _touches_image_edge(self, bbox: BBox, image_width: int, image_height: int) -> bool:
        edge_margin = max(2, int(round(min(bbox.width, bbox.height) * 0.03)))
        return (
            bbox.x <= edge_margin
            or bbox.y <= edge_margin
            or image_width - bbox.right <= edge_margin
            or image_height - bbox.bottom <= edge_margin
        )

    def _build_edge_envelope_patch(self, width: int, height: int) -> np.ndarray:
        patch = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(1, int(round(width * 0.5))), max(1, int(round(height * 0.48))))
        cv2.ellipse(patch, center, axes, 0, 0, 360, 255, -1)

        band_height = max(1, int(round(height * 0.34)))
        band_top = max(0, center[1] - band_height // 2)
        band_bottom = min(height - 1, center[1] + band_height // 2)
        cv2.rectangle(patch, (0, band_top), (width - 1, band_bottom), 255, -1)
        return patch

    def _best_valid_candidate(
        self,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        width: int,
        height: int,
        *,
        target_edges: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        valid_candidates: list[dict[str, Any]] = []
        accept_floor = max(0.18, float(self.min_confidence) * 0.92)
        for candidate in candidates:
            if float(candidate["score"]) < accept_floor * 0.95:
                # List is score×size ranked, not pure score — keep scanning a bit
                continue
            refined = self._maybe_refine(candidate, target_edges, width, height)
            # After refine, allow a hair under min_confidence if later ORB/structure are strong
            if float(refined["score"]) < accept_floor:
                continue
            ok = self._validate_candidate(image, refined, width, height)
            if ok is not None:
                valid_candidates.append(ok)

        if not valid_candidates:
            return None

        # Prefer composite rank (score × size × color × residual)
        if self.prefer_larger_score_margin <= 0:
            return max(valid_candidates, key=lambda item: float(item.get("rank", item["score"])))
        best_score = max(float(c["score"]) for c in valid_candidates)
        near_best = [
            candidate
            for candidate in valid_candidates
            if float(candidate["score"]) >= best_score - self.prefer_larger_score_margin
        ]
        return max(near_best, key=lambda item: (float(item.get("rank", 0.0)), item["area"]))

    def _count_hough_circles(self, image: np.ndarray, bbox: BBox) -> int:
        if self.min_hough_circles <= 0:
            return 0

        patch = image[bbox.y:bbox.bottom, bbox.x:bbox.right]
        if patch.size == 0:
            return 0

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        min_dimension = min(bbox.width, bbox.height)
        min_radius = max(5, int(round(min_dimension * 0.18)))
        max_radius = max(min_radius + 1, int(round(min_dimension * 0.48)))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, min_dimension // 2),
            param1=80,
            param2=self.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        return 0 if circles is None else int(circles.shape[1])

    def _watermark_color_ratio(self, image: np.ndarray, bbox: BBox) -> float:
        patch = image[bbox.y:bbox.bottom, bbox.x:bbox.right]
        if patch.size == 0:
            return 0.0

        resized_template = cv2.resize(
            self.template_mask,
            (bbox.width, bbox.height),
            interpolation=cv2.INTER_AREA,
        )
        template_region = resized_template > 10
        template_pixels = int(np.count_nonzero(template_region))
        if template_pixels == 0:
            return 0.0

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        pale = (saturation < 90) & (value > 135)
        dark = (saturation < 130) & (value < 95)
        red = ((hue < 14) | (hue > 166)) & (saturation > 55) & (value > 70)
        watermark_like = (pale | dark | red) & template_region
        return float(np.count_nonzero(watermark_like) / template_pixels)

    def _build_output_template_mask(self) -> np.ndarray:
        mode = (self.output_mask_mode or "bbox").strip().lower()
        # bbox mode does not paste this mask at detect time; keep edges for matching only
        if mode in {"bbox", "box", "rect", "rectangle", "filled"}:
            return self.template_mask
        if mode == "envelope":
            return self._build_envelope_mask()
        if mode != "footprint":
            return self.template_mask

        kernel_size = self.footprint_close_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(self.template_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        footprint = np.zeros_like(self.template_mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.footprint_min_area:
                cv2.drawContours(footprint, [contour], -1, 255, -1)
        return footprint

    def _build_envelope_mask(self) -> np.ndarray:
        cleaned = self.template_mask.copy()
        if self.footprint_close_kernel > 1:
            kernel_size = self.footprint_close_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        ys, xs = np.where(cleaned > 0)
        if len(xs) == 0 or len(ys) == 0:
            return self.template_mask

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        center = (x1 + width // 2, y1 + height // 2)
        pad = max(2, int(round(max(width, height) * 0.035)))

        envelope = np.zeros_like(self.template_mask)
        axes = (max(1, width // 2 + pad), max(1, height // 2 + pad))
        cv2.ellipse(envelope, center, axes, 0, 0, 360, 255, -1)

        band_height = max(1, int(round(height * 0.34)))
        band_top = max(0, center[1] - band_height // 2)
        band_bottom = min(envelope.shape[0], center[1] + band_height // 2)
        cv2.rectangle(
            envelope,
            (max(0, x1 - pad), band_top),
            (min(envelope.shape[1] - 1, x2 + pad), band_bottom),
            255,
            -1,
        )
        return envelope
