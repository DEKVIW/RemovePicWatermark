from .features import FeatureParams, compute_feature_edges
from .fixed_box import FixedBoxDetector
from .matching import extract_peaks, match_template_peaks, nms_by_iou
from .orchestrator import DetectOrchestrator, normalize_detect_mode
from .residual_ai import ResidualAiDetector
from .template_stamp import TemplateStampDetector
from .yolo_watermark import (
    YoloProbe,
    YoloWatermarkDetector,
    ensure_yolo_dir,
    normalize_yolo_device,
    probe_yolo,
    resolve_yolo_weights,
    ultralytics_available,
)

__all__ = [
    "DetectOrchestrator",
    "FeatureParams",
    "FixedBoxDetector",
    "ResidualAiDetector",
    "TemplateStampDetector",
    "YoloProbe",
    "YoloWatermarkDetector",
    "compute_feature_edges",
    "ensure_yolo_dir",
    "extract_peaks",
    "match_template_peaks",
    "nms_by_iou",
    "normalize_detect_mode",
    "normalize_yolo_device",
    "probe_yolo",
    "resolve_yolo_weights",
    "ultralytics_available",
]
