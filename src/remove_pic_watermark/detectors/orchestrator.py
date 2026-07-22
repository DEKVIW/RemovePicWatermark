"""Compose style detectors + optional AI detectors into one detect pass.

Product modes:
  - styles: style matching only (may use YOLO *inside* style as propose→confirm)
  - ai:     detection model (+ residual) only
  - both:   **cascade** — styles first; AI fill-in only when style finds nothing
            (not a union of all boxes — that caused multi-hit false positives)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

import numpy as np

from ..models import Detection
from .base import detector_label

DetectMode = Literal["styles", "ai", "both"]

# AI detectors that paint free-form proposals (not style-confirmed)
_AI_DETECTOR_TAGS = frozenset({"residual_ai", "yolo_watermark", "ai_residual", "ai_yolo"})


def normalize_detect_mode(value: str | None) -> DetectMode:
    raw = (value or "styles").strip().lower()
    if raw in {"ai", "ai_only", "residual", "yolo"}:
        return "ai"
    if raw in {"both", "all", "union", "styles+ai", "styles_ai"}:
        return "both"
    return "styles"


def _is_ai_detection(det: Detection) -> bool:
    meta = det.metadata or {}
    tag = str(meta.get("detector") or meta.get("source_detector") or det.label or "").lower()
    if tag in _AI_DETECTOR_TAGS:
        return True
    if tag.startswith("ai_"):
        return True
    return False


def _bbox_area(det: Detection) -> float:
    b = det.bbox
    if b is None:
        return 0.0
    return float(max(0, b.width) * max(0, b.height))


def _iou_xywh(a: Detection, b: Detection) -> float:
    ba, bb = a.bbox, b.bbox
    if ba is None or bb is None:
        return 0.0
    ax2, ay2 = ba.x + ba.width, ba.y + ba.height
    bx2, by2 = bb.x + bb.width, bb.y + bb.height
    ix1, iy1 = max(ba.x, bb.x), max(ba.y, bb.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = ba.width * ba.height + bb.width * bb.height - inter
    return float(inter / max(1, union))


@dataclass
class DetectOrchestrator:
    """Run style and/or AI detectors.

    ``both`` is cascade (style → if empty then AI), not union.
    """

    style_detectors: list[Any] = field(default_factory=list)
    ai_detectors: list[Any] = field(default_factory=list)
    mode: DetectMode = "styles"
    # Template (w, h) from selected styles — reject absurd AI boxes
    style_size_hints: list[tuple[int, int]] = field(default_factory=list)
    filter_ai_to_style_scale: bool = True
    ai_min_area_scale: float = 0.25
    ai_max_area_scale: float = 2.8
    # Absolute cap: AI box area / image area
    ai_max_image_area_ratio: float = 0.05
    # Drop AI box that heavily overlaps a style hit (union path / safety)
    ai_style_iou_suppress: float = 0.25
    # Minimum confidence for AI *fill* boxes (cascade empty → AI)
    ai_fill_min_confidence: float = 0.22
    # YOLO-only floor (stricter than residual when both present)
    ai_fill_yolo_min_confidence: float = 0.25
    # Style hit must clear this conf to skip AI fill (blocks weak TM false positives)
    style_accept_min_confidence: float = 0.38
    # Style box area / image area above this is treated as weak (e.g. half-frame miss)
    style_max_image_area_ratio: float = 0.10
    last_run_stats: dict[str, Any] = field(default_factory=dict)

    def active_detectors(self) -> list[Any]:
        """Detectors that may run (for describe UI). both lists both chains."""
        if self.mode == "styles":
            return list(self.style_detectors)
        if self.mode == "ai":
            return list(self.ai_detectors)
        return list(self.style_detectors) + list(self.ai_detectors)

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        yolo_raw = 0
        yolo_emitted = 0

        def _run(dets: list[Any]) -> list[Detection]:
            nonlocal yolo_raw, yolo_emitted
            out: list[Detection] = []
            for detector in dets:
                hits = detector.detect(image) or []
                if hasattr(detector, "last_raw_proposals"):
                    try:
                        yolo_raw += int(getattr(detector, "last_raw_proposals") or 0)
                    except (TypeError, ValueError):
                        pass
                if hasattr(detector, "last_emitted"):
                    try:
                        yolo_emitted += int(getattr(detector, "last_emitted") or 0)
                    except (TypeError, ValueError):
                        pass
                name = detector_label(detector).lower()
                if "yolo" not in name and hasattr(detector, "_yolo_proposer"):
                    prop = getattr(detector, "_yolo_proposer", None)
                    if prop is not None and hasattr(prop, "last_raw_proposals"):
                        try:
                            yolo_raw += int(getattr(prop, "last_raw_proposals") or 0)
                        except (TypeError, ValueError):
                            pass
                for hit in hits:
                    meta = dict(hit.metadata or {})
                    meta.setdefault("source_detector", detector_label(detector))
                    meta.setdefault("detect_mode", self.mode)
                    hit.metadata = meta
                    out.append(hit)
            return out

        cascade_path = "n/a"
        style_raw_n = 0
        style_strong_n = 0
        if self.mode == "styles":
            detections = _run(self.style_detectors)
            cascade_path = "styles_only"
            style_raw_n = len(detections)
            style_strong_n = len(
                self._strong_style_hits(detections, (height, width))
            )
        elif self.mode == "ai":
            detections = _run(self.ai_detectors)
            if self.filter_ai_to_style_scale and self.style_size_hints:
                detections = self._filter_ai_boxes(
                    detections, (height, width), style_hits=[], force_fill=True
                )
            cascade_path = "ai_only"
        else:
            # both = cascade (style first; AI only if no *strong* style hit)
            style_hits = _run(self.style_detectors)
            style_raw_n = len(style_hits)
            strong = self._strong_style_hits(style_hits, (height, width))
            style_strong_n = len(strong)
            if strong:
                # Strong style (incl. YOLO propose→confirm). Do NOT union AI paint.
                detections = strong
                cascade_path = "style_hit_skip_ai"
            elif self.ai_detectors:
                ai_hits = _run(self.ai_detectors)
                detections = self._filter_ai_boxes(
                    ai_hits, (height, width), style_hits=[], force_fill=True
                )
                cascade_path = (
                    "style_weak_ai_fill" if style_hits else "style_miss_ai_fill"
                )
            elif style_hits:
                # No AI available: keep best weak style rather than empty
                detections = self._keep_best_style_fallback(style_hits)
                cascade_path = "style_weak_no_ai"
            else:
                detections = []
                cascade_path = "style_miss_no_ai"

        yolo_kept = sum(
            1
            for d in detections
            if "yolo" in str((d.metadata or {}).get("detector") or "").lower()
        )
        self.last_run_stats = {
            "yolo_raw_proposals": yolo_raw,
            "yolo_emitted_before_filter": yolo_emitted,
            "yolo_kept_after_filter": yolo_kept,
            "yolo_dropped_by_filter": max(0, yolo_emitted - yolo_kept),
            "cascade_path": cascade_path,
            "style_raw_hits": style_raw_n,
            "style_strong_hits": style_strong_n,
            "style_hits": sum(1 for d in detections if not _is_ai_detection(d)),
            "ai_hits": sum(1 for d in detections if _is_ai_detection(d)),
        }
        if detections:
            meta = dict(detections[0].metadata or {})
            meta["yolo_trace"] = dict(self.last_run_stats)
            meta["cascade_path"] = cascade_path
            detections[0].metadata = meta
        return detections

    def _strong_style_hits(
        self, hits: list[Detection], image_hw: tuple[int, int]
    ) -> list[Detection]:
        """Style hits good enough to skip AI fill (confidence + size sanity)."""
        if not hits:
            return []
        height, width = int(image_hw[0]), int(image_hw[1])
        image_area = float(max(1, height * width))
        max_area = image_area * float(self.style_max_image_area_ratio)
        conf_floor = float(self.style_accept_min_confidence)

        ref = 0.0
        if self.style_size_hints:
            areas = [float(max(1, w) * max(1, h)) for w, h in self.style_size_hints]
            if areas:
                ref = float(median(areas))

        strong: list[Detection] = []
        for d in hits:
            if _is_ai_detection(d):
                continue
            conf = float(getattr(d, "confidence", 0.0) or 0.0)
            if conf < conf_floor:
                continue
            area = _bbox_area(d)
            if area <= 0 or area > max_area:
                continue
            # Vs template size: reject absurd scales (half-frame TM false hits)
            if ref > 0 and (area > ref * 4.5 or area < ref * 0.10):
                continue
            strong.append(d)
        return strong

    @staticmethod
    def _keep_best_style_fallback(hits: list[Detection]) -> list[Detection]:
        """When AI unavailable, keep highest-confidence style hit only."""
        style = [d for d in hits if not _is_ai_detection(d)]
        if not style:
            return []
        best = max(style, key=lambda d: float(d.confidence or 0.0))
        return [best]

    def _filter_ai_boxes(
        self,
        detections: list[Detection],
        image_hw: tuple[int, int],
        *,
        style_hits: list[Detection],
        force_fill: bool = False,
    ) -> list[Detection]:
        """Keep AI boxes near style/template scale; drop low conf / huge boxes."""
        ai_hits = [d for d in detections if _is_ai_detection(d)]
        non_ai = [d for d in detections if not _is_ai_detection(d)]
        if not ai_hits:
            return detections

        height, width = int(image_hw[0]), int(image_hw[1])
        image_area = float(max(1, height * width))

        ref_areas = [_bbox_area(d) for d in style_hits if _bbox_area(d) >= 64]
        if not ref_areas and self.style_size_hints:
            ref_areas = [float(max(1, w) * max(1, h)) for w, h in self.style_size_hints]
        ref = float(median(ref_areas)) if ref_areas else 0.0
        style_miss = len(style_hits) == 0 or force_fill

        kept_ai: list[Detection] = []
        dropped = 0
        dropped_yolo = 0
        dropped_residual = 0
        kept_yolo = 0
        kept_residual = 0

        for d in ai_hits:
            area = _bbox_area(d)
            meta0 = d.metadata or {}
            det_tag = str(meta0.get("detector") or "").lower()
            is_yolo = "yolo" in det_tag
            conf = float(getattr(d, "confidence", 0.0) or 0.0)

            # Confidence floor (cascade fill is strict — few-shot YOLO loves 0.1 noise)
            conf_floor = (
                float(self.ai_fill_yolo_min_confidence)
                if is_yolo
                else max(0.18, float(self.ai_fill_min_confidence) * 0.85)
            )
            if conf < conf_floor:
                dropped += 1
                if is_yolo:
                    dropped_yolo += 1
                else:
                    dropped_residual += 1
                continue

            if is_yolo:
                min_scale = 0.18 if style_miss else 0.20
                max_scale = 3.5 if style_miss else 3.0
                abs_ratio = 0.06 if style_miss else 0.05
            else:
                min_scale = float(self.ai_min_area_scale)
                max_scale = float(self.ai_max_area_scale)
                abs_ratio = min(float(self.ai_max_image_area_ratio), 0.04)

            abs_cap = image_area * abs_ratio

            if area <= 0:
                dropped += 1
                if is_yolo:
                    dropped_yolo += 1
                else:
                    dropped_residual += 1
                continue
            if area > abs_cap:
                dropped += 1
                if is_yolo:
                    dropped_yolo += 1
                else:
                    dropped_residual += 1
                continue
            if ref > 0:
                if area < ref * min_scale or area > ref * max_scale:
                    dropped += 1
                    if is_yolo:
                        dropped_yolo += 1
                    else:
                        dropped_residual += 1
                    continue
            if style_hits and any(
                _iou_xywh(d, s) >= self.ai_style_iou_suppress for s in style_hits
            ):
                dropped += 1
                if is_yolo:
                    dropped_yolo += 1
                else:
                    dropped_residual += 1
                continue

            meta = dict(meta0)
            meta["style_scale_filter"] = "kept"
            meta["cascade_fill"] = bool(force_fill or style_miss)
            if ref > 0:
                meta["style_ref_area"] = round(ref, 1)
                meta["area_scale"] = round(area / ref, 3)
            d.metadata = meta
            kept_ai.append(d)
            if is_yolo:
                kept_yolo += 1
            else:
                kept_residual += 1

        # NMS among AI fills so residual doesn't stack 4 strips
        kept_ai = self._nms_ai(kept_ai, iou_thr=0.35, max_keep=6)

        stats = {
            "ai_filtered_out": dropped,
            "ai_kept": len(kept_ai),
            "yolo_kept": kept_yolo,
            "yolo_dropped": dropped_yolo,
            "residual_kept": kept_residual,
            "residual_dropped": dropped_residual,
            "style_miss_on_image": style_miss,
        }
        host = non_ai[:1] or kept_ai[:1]
        for s in host:
            meta = dict(s.metadata or {})
            meta.update(stats)
            s.metadata = meta
            break
        return non_ai + kept_ai

    @staticmethod
    def _nms_ai(
        dets: list[Detection], *, iou_thr: float = 0.35, max_keep: int = 6
    ) -> list[Detection]:
        if len(dets) <= 1:
            return dets
        ordered = sorted(dets, key=lambda d: float(d.confidence or 0.0), reverse=True)
        keep: list[Detection] = []
        for d in ordered:
            if any(_iou_xywh(d, k) >= iou_thr for k in keep):
                continue
            keep.append(d)
            if len(keep) >= max_keep:
                break
        return keep

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cascade": self.mode == "both",
            "styles": [detector_label(d) for d in self.style_detectors],
            "ai": [detector_label(d) for d in self.ai_detectors],
            "active": [detector_label(d) for d in self.active_detectors()],
            "style_size_hints": list(self.style_size_hints),
            "filter_ai_to_style_scale": bool(
                self.filter_ai_to_style_scale and self.mode in {"both", "ai"}
            ),
        }
