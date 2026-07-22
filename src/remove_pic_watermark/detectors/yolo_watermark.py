"""Optional YOLO watermark detector (ultralytics).

Product role: part of 「自动扫描」 when weights + ultralytics are available.
Never required — residual scan and style matching work without it.

## How to land (usage)

1. Install: ``pip install ultralytics`` or ``pip install -e ".[yolo]"``
2. Place a **watermark-trained** weight file (not generic COCO) at one of:
   - ``workspace/models/yolo/watermark.pt``  (preferred)
   - ``watermark.onnx`` / ``best.pt`` / ``yolo_watermark.pt`` in the same folder
   - env ``REMOVE_PIC_YOLO_WEIGHTS`` = absolute path
   - ``JobSpec.yolo_weights``
3. Batch 「怎么找水印」 = 样式+自动扫描 / 仅自动扫描 → job auto-attaches YOLO
4. Device follows batch device (cpu / cuda)

Full GUI freeze (0.2.3+) ships ultralytics + torch for detect/train.
Minimal builds may still omit ultralytics; residual/style paths work without it.

Mask strategy (product default — no user tuning):
- Always solid **bbox fill** for LaMa coverage (not stamp contour)
- Optional residual refine is off by default (can carve holes and miss ink)
- conf / iou / imgsz are fixed internals; batch UI only picks detect mode + device
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from ..masking import dilate_mask
from ..models import BBox, Detection
from .features import FeatureParams, compute_feature_edges

YoloStatusKind = Literal[
    "ready",
    "missing_ultralytics",
    "missing_weights",
    "load_error",
]

WEIGHT_NAMES = (
    "watermark.pt",
    "watermark.onnx",
    "best.pt",
    "yolo_watermark.pt",
    "yolov8n-watermark.pt",
    "yolov8s-watermark.pt",
)


_YOLO_CLS: Any | None = None
_YOLO_IMPORT_ERROR: str | None = None


def _yolo_env_bootstrap() -> None:
    """Freeze-friendly env before any ultralytics import."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Never try pip install from inside a frozen GUI
    os.environ.setdefault("YOLO_AUTOINSTALL", "0")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")


def _patch_six_meta_path() -> None:
    """Delegate to lightweight module (avoids circular import cost)."""
    from ..six_patch import patch_six_meta_path

    patch_six_meta_path()


def _log_yolo_import(message: str) -> None:
    """Append one line to app-root log (dev + frozen)."""
    try:
        from ..paths import app_root

        path = app_root() / "yolo_import.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(message.rstrip() + "\n")
    except Exception:  # noqa: BLE001
        pass


def _import_yolo_class_candidates() -> Any:
    """Try import paths that avoid heavy optional models (FastSAM/NAS).

    ``from ultralytics import YOLO`` goes through ``ultralytics.models`` which
    eagerly imports FastSAM/NAS/RTDETR — those often break under PyInstaller.
    Prefer the direct model class path used by training/detect.
    """
    errors: list[str] = []

    # 1) Direct class (best for freeze)
    try:
        from ultralytics.models.yolo.model import YOLO  # type: ignore

        return YOLO
    except Exception as exc:  # noqa: BLE001
        errors.append(f"models.yolo.model: {type(exc).__name__}: {exc}")

    # 2) Package lazy attr (ultralytics 8.4+)
    try:
        import ultralytics as _ultra  # type: ignore

        yolo_cls = getattr(_ultra, "YOLO", None)
        if yolo_cls is not None:
            return yolo_cls
        errors.append("ultralytics.YOLO: attribute missing")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ultralytics package: {type(exc).__name__}: {exc}")

    # 3) Classic from-import
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as exc:  # noqa: BLE001
        errors.append(f"from ultralytics import YOLO: {type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors))


def _patch_torch_numpy_ufuncs_if_needed() -> None:
    """Mitigate PyInstaller NameError in torch._numpy._ufuncs (vars()[name]).

    Under freeze, ``for name in ...: vars()[name] = ...`` loses ``name``.
    Prefer rewriting on-disk module next to the frozen app; also safe no-op in dev.
    """
    try:
        import sys
        from pathlib import Path

        candidates: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "torch" / "_numpy" / "_ufuncs.py")
        if getattr(sys, "frozen", False):
            candidates.append(
                Path(sys.executable).resolve().parent
                / "_internal"
                / "torch"
                / "_numpy"
                / "_ufuncs.py"
            )
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if "vars()[name]" not in text:
                return
            fixed = text.replace(
                "vars()[name] = deco_binary_ufunc(ufunc)",
                "globals()[name] = deco_binary_ufunc(ufunc)",
            ).replace(
                "vars()[name] = deco_unary_ufunc(ufunc)",
                "globals()[name] = deco_unary_ufunc(ufunc)",
            )
            if fixed != text:
                path.write_text(fixed, encoding="utf-8")
                _log_yolo_import(f"patched torch._numpy._ufuncs: {path}")
            return
    except Exception as exc:  # noqa: BLE001
        _log_yolo_import(f"torch._numpy patch skip: {exc}")


def warm_import_yolo() -> Any | None:
    """Import the YOLO class **before** PySide6 when possible.

    GUI apps load PySide6/shiboken first; later ultralytics import can trip
    matplotlib/dateutil/six + shiboken conflicts. Pre-import once before Qt.
    Also uses freeze-safe import paths (see ``_import_yolo_class_candidates``).
    Safe to call multiple times; returns the class or None.
    """
    global _YOLO_CLS, _YOLO_IMPORT_ERROR
    if _YOLO_CLS is not None:
        return _YOLO_CLS
    _yolo_env_bootstrap()
    _patch_six_meta_path()
    _patch_torch_numpy_ufuncs_if_needed()
    try:
        # Touch torch first so CUDA libs are loaded in a predictable order
        try:
            import torch  # noqa: F401
        except Exception as torch_exc:  # noqa: BLE001
            _log_yolo_import(f"torch preload: {type(torch_exc).__name__}: {torch_exc}")

        # six may break during first attempt if Qt already loaded; patch again and retry
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                if attempt:
                    _patch_six_meta_path()
                yolo_cls = _import_yolo_class_candidates()
                _YOLO_CLS = yolo_cls
                _YOLO_IMPORT_ERROR = None
                _log_yolo_import(f"YOLO import OK (attempt {attempt + 1})")
                return _YOLO_CLS
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        assert last_exc is not None
        raise last_exc
    except Exception as exc:  # noqa: BLE001
        import traceback

        _YOLO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        _log_yolo_import(f"YOLO import FAIL: {_YOLO_IMPORT_ERROR}")
        _log_yolo_import(traceback.format_exc())
        return None


def import_yolo_class() -> Any:
    """Return YOLO class or raise RuntimeError with a clear message."""
    cls = warm_import_yolo()
    if cls is not None:
        return cls
    # Package present but class broken (typical after PySide6 without warm import)
    try:
        import ultralytics  # noqa: F401

        raise RuntimeError("检测组件加载失败，请重启应用后重试。")
    except ImportError as exc:
        raise RuntimeError("检测组件不可用。") from exc


def ultralytics_package_present() -> bool:
    """Cheap check: ultralytics is importable without loading torch/YOLO."""
    if _YOLO_CLS is not None:
        return True
    import importlib.util

    return importlib.util.find_spec("ultralytics") is not None


def ultralytics_available() -> bool:
    """True if the YOLO class can be used (not merely that the package exists).

    For UI status labels prefer ``ultralytics_package_present()`` or
    ``probe_yolo(..., try_load=False)`` which uses the package check only.
    Full import is deferred until train/run.
    """
    if _YOLO_CLS is not None:
        return True
    if not ultralytics_package_present():
        return False
    # Only warm-import when caller needs a real class (train / run paths)
    if warm_import_yolo() is not None:
        return True
    return False


def yolo_import_error() -> str | None:
    """Last YOLO import failure detail, if any."""
    return _YOLO_IMPORT_ERROR


def normalize_yolo_device(device: str | None) -> str:
    """Map app device prefs to ultralytics device string."""
    raw = (device or "cpu").strip().lower()
    if raw in {"", "auto"}:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    if raw in {"cuda", "gpu", "cuda:0"}:
        return "0"
    if raw.startswith("cuda:"):
        return raw.split(":", 1)[1] or "0"
    if raw.isdigit():
        return raw
    return "cpu"


def resolve_yolo_weights(
    workspace_models: Path | None = None,
    explicit: Path | str | None = None,
) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    env = os.environ.get("REMOVE_PIC_YOLO_WEIGHTS", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path
    if workspace_models is not None:
        yolo_dir = Path(workspace_models) / "yolo"
        for name in WEIGHT_NAMES:
            cand = yolo_dir / name
            if cand.is_file():
                return cand
        # Any single .pt / .onnx in folder (user-named)
        if yolo_dir.is_dir():
            pts = sorted(yolo_dir.glob("*.pt")) + sorted(yolo_dir.glob("*.onnx"))
            for cand in pts:
                if cand.is_file():
                    return cand
    return None


def ensure_yolo_dir(workspace_models: Path) -> Path:
    """Create models/yolo and drop a short README if missing."""
    yolo_dir = Path(workspace_models) / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    readme = yolo_dir / "README.txt"
    if not readme.is_file():
        readme.write_text(
            "YOLO 水印检测权重目录\n"
            "====================\n"
            "\n"
            "推荐文件名：watermark.pt\n"
            "也支持：watermark.onnx / best.pt / yolo_watermark.pt\n"
            "或环境变量 REMOVE_PIC_YOLO_WEIGHTS=完整路径\n"
            "\n"
            "依赖（源码）：\n"
            "  pip install ultralytics\n"
            "  或 pip install -e \".[yolo]\"\n"
            "\n"
            "GUI：\n"
            "  - 按样式找：漏检时自动 YOLO 提案（需库+权重），无单独开关\n"
            "  - 样式+自动扫描 / 仅自动扫描：残差 ± YOLO 检测器\n"
            "\n"
            "默认发布 exe 不内置 ultralytics；要 YOLO 请用源码环境。\n"
            "完整步骤见 docs/YOLO自动扫描落地.md\n"
            "\n"
            "注意：需要水印专用权重，通用 COCO 检测效果差。\n",
            encoding="utf-8",
        )
    return yolo_dir


# Public watermark detector (HF) — optional auto-fetch, no user conf knobs
_DEFAULT_YOLO_URLS: tuple[tuple[str, str], ...] = (
    # Prefer smaller-ish ONNX if available; fall back to common .pt names
    (
        "watermark.pt",
        "https://huggingface.co/qfisch/yolov8n-watermark-detection/resolve/main/best.pt",
    ),
    (
        "watermark.pt",
        "https://huggingface.co/Eugeoter/yolov5-watermark-detection/resolve/main/best.pt",
    ),
)


def try_download_default_yolo_weights(
    workspace_models: Path,
    *,
    timeout_s: float = 120.0,
) -> Path | None:
    """Download a public watermark YOLO weight if none is present.

    Best-effort only: network/license failures return None (residual scan still works).
    Users never configure conf/iou — only need ultralytics + a weight file once.
    """
    existing = resolve_yolo_weights(workspace_models)
    if existing is not None:
        return existing
    yolo_dir = ensure_yolo_dir(workspace_models)
    dest = yolo_dir / "watermark.pt"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest

    import urllib.error
    import urllib.request

    for name, url in _DEFAULT_YOLO_URLS:
        target = yolo_dir / name
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RemovePicWatermark/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            if len(data) < 500_000:
                # Too small — likely an HTML error page
                continue
            tmp.write_bytes(data)
            tmp.replace(target)
            if target.is_file():
                return target
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            continue
    return None


@dataclass(frozen=True)
class YoloProbe:
    """Readiness snapshot for UI / report / packaging checks."""

    status: YoloStatusKind
    weights: Path | None = None
    message: str = ""
    ultralytics: bool = False

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "weights": str(self.weights) if self.weights else None,
            "message": self.message,
            "ultralytics": self.ultralytics,
        }


def probe_yolo(
    workspace_models: Path | None = None,
    explicit: Path | str | None = None,
    *,
    try_load: bool = False,
    device: str = "cpu",
) -> YoloProbe:
    """Probe YOLO readiness without always loading the model.

    When ``try_load=False`` (default), only checks package presence + weights path
    — no torch/ultralytics import (safe for GUI startup / status chips).
    """
    has_ultra = (
        ultralytics_available() if try_load else ultralytics_package_present()
    )
    weights = resolve_yolo_weights(workspace_models, explicit)
    if not has_ultra:
        return YoloProbe(
            status="missing_ultralytics",
            weights=weights,
            ultralytics=False,
            message="检测组件未就绪",
        )
    if weights is None:
        return YoloProbe(
            status="missing_weights",
            weights=None,
            ultralytics=True,
            message="尚无检测模型，请先在「训练检测」页训练",
        )
    if not try_load:
        return YoloProbe(
            status="ready",
            weights=weights,
            ultralytics=True,
            message=f"检测模型就绪：{weights.name}",
        )
    try:
        det = YoloWatermarkDetector(weights=weights, device=normalize_yolo_device(device))
        det._ensure_model()
        return YoloProbe(
            status="ready",
            weights=weights,
            ultralytics=True,
            message=f"检测模型已加载：{weights.name}",
        )
    except Exception as error:  # noqa: BLE001
        return YoloProbe(
            status="load_error",
            weights=weights,
            ultralytics=True,
            message=f"检测模型加载失败：{error}",
        )


@dataclass
class YoloWatermarkDetector:
    """Wrap ultralytics YOLO: boxes (and optional seg masks) → Detection list."""

    weights: Path
    label: str = "ai_yolo"
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    dilate: int = 4
    max_instances: int = 64
    imgsz: int = 640
    # Empty = all classes; set e.g. [0] if single-class watermark model
    class_ids: list[int] = field(default_factory=list)
    # Off by default: solid box is better for LaMa than carved residual masks
    refine_with_residual: bool = False
    residual_keep_ratio: float = 0.12
    _model: Any = None

    @classmethod
    def try_create(
        cls,
        *,
        workspace_models: Path | None = None,
        weights: Path | str | None = None,
        device: str = "cpu",
        conf: float = 0.25,
        label: str = "ai_yolo",
        imgsz: int = 640,
        class_ids: list[int] | None = None,
        refine_with_residual: bool = False,
    ) -> "YoloWatermarkDetector | None":
        if not ultralytics_available():
            return None
        path = resolve_yolo_weights(workspace_models, weights)
        if path is None and workspace_models is not None:
            # One-shot optional download of a public watermark detector (no UI knobs)
            path = try_download_default_yolo_weights(Path(workspace_models))
        if path is None:
            return None
        try:
            det = cls(
                weights=path,
                device=normalize_yolo_device(device),
                conf=conf,
                label=label,
                imgsz=imgsz,
                class_ids=list(class_ids or []),
                refine_with_residual=refine_with_residual,
            )
            det._ensure_model()
            return det
        except Exception:  # noqa: BLE001
            return None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        YOLO = import_yolo_class()
        self._model = YOLO(str(self.weights))
        return self._model

    def propose_boxes(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return raw box proposals for template confirmation (no mask emission).

        Each item: ``{"bbox": BBox, "confidence": float, "class_id": int,
        "obb_poly": optional Nx2 float array, "angle_deg": float}``.

        Supports both detect (xyxy) and OBB (xywhr / polygon) weight files.
        """
        height, width = image.shape[:2]
        model = self._ensure_model()
        predict_kw: dict[str, Any] = {
            "source": image,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
            "max_det": self.max_instances,
            "imgsz": self.imgsz,
        }
        if self.class_ids:
            predict_kw["classes"] = list(self.class_ids)

        results = model.predict(**predict_kw)
        if not results:
            self.last_raw_proposals = 0
            return []
        result = results[0]

        # Prefer OBB output when present (rotated watermarks)
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            proposals = self._proposals_from_obb(obb, width, height)
            self.last_raw_proposals = len(proposals)
            return proposals

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self.last_raw_proposals = 0
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))

        proposals: list[dict[str, Any]] = []
        for box, conf, cls_id in zip(xyxy, confs, clss, strict=False):
            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
            bbox = BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).clamp(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            proposals.append(
                {
                    "bbox": bbox,
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "angle_deg": 0.0,
                    "obb_poly": None,
                }
            )
        self.last_raw_proposals = len(proposals)
        return proposals

    @staticmethod
    def _proposals_from_obb(obb: Any, width: int, height: int) -> list[dict[str, Any]]:
        """Parse ultralytics OBB result into AABB + optional polygon."""
        import math

        proposals: list[dict[str, Any]] = []
        try:
            confs = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None else None
            clss = obb.cls.cpu().numpy() if getattr(obb, "cls", None) is not None else None
            # xyxyxyxy: N×4×2 corners
            polys = None
            if getattr(obb, "xyxyxyxy", None) is not None:
                polys = obb.xyxyxyxy.cpu().numpy()
            xywhr = None
            if getattr(obb, "xywhr", None) is not None:
                xywhr = obb.xywhr.cpu().numpy()
            n = len(obb)
        except Exception:  # noqa: BLE001
            return []

        for i in range(n):
            conf = float(confs[i]) if confs is not None else 1.0
            cls_id = int(clss[i]) if clss is not None else 0
            angle_deg = 0.0
            poly = None
            # Prefer angle from xywhr whenever available (poly path used to leave 0°)
            if xywhr is not None and i < len(xywhr):
                try:
                    angle_deg = math.degrees(float(xywhr[i][4]))
                except Exception:  # noqa: BLE001
                    angle_deg = 0.0
            if polys is not None and i < len(polys):
                poly = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2)
                xs, ys = poly[:, 0], poly[:, 1]
                x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
                x2, y2 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
                # If xywhr angle missing, recover from corners
                if abs(angle_deg) < 1e-3 and len(poly) >= 4:
                    try:
                        (_c, _sz, ang) = cv2.minAreaRect(poly.astype(np.float32))
                        angle_deg = float(ang)
                    except Exception:  # noqa: BLE001
                        pass
            elif xywhr is not None and i < len(xywhr):
                cx, cy, bw, bh, rad = (float(v) for v in xywhr[i][:5])
                angle_deg = math.degrees(rad)
                # AABB of rotated rect
                cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
                aw = bw * cos_a + bh * sin_a
                ah = bw * sin_a + bh * cos_a
                x1 = int(round(cx - aw / 2))
                y1 = int(round(cy - ah / 2))
                x2 = int(round(cx + aw / 2))
                y2 = int(round(cy + ah / 2))
                # Build 4 corners for mask fill
                ca, sa = math.cos(rad), math.sin(rad)
                hw, hh = bw / 2, bh / 2
                corners = [
                    (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
                    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
                ]
                poly = np.asarray(corners, dtype=np.float32)
            else:
                continue
            # Normalize angle to (-90, 90] for readability
            while angle_deg <= -90.0:
                angle_deg += 180.0
            while angle_deg > 90.0:
                angle_deg -= 180.0
            bbox = BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).clamp(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            proposals.append(
                {
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "angle_deg": float(angle_deg),
                    "obb_poly": poly,
                }
            )
        return proposals

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        detections: list[Detection] = []
        props = self.propose_boxes(image)
        raw_n = int(getattr(self, "last_raw_proposals", len(props)) or 0)
        for index, prop in enumerate(props):
            bbox: BBox = prop["bbox"]
            mask = np.zeros((height, width), dtype=np.uint8)
            poly = prop.get("obb_poly")
            mask_mode = "bbox_fill"
            if poly is not None:
                # Solid rotated rectangle — tighter than AABB for diagonal text
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 255)
                mask_mode = "obb_fill"
            else:
                mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
                mask_mode = "bbox_fill"
            if self.refine_with_residual and mask_mode == "bbox_fill":
                mask, mask_mode = self._refine_bbox_mask(image, bbox, mask)
                if int(np.count_nonzero(mask)) == 0:
                    mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = 255
                    mask_mode = "bbox_fill_empty_fallback"

            mask = dilate_mask(mask, self.dilate)
            meta: dict[str, Any] = {
                "detector": "yolo_watermark",
                "weights": str(self.weights),
                "instance_index": index,
                "class_id": int(prop.get("class_id", 0)),
                "mask_mode": mask_mode,
                "angle_deg": float(prop.get("angle_deg") or 0.0),
                "device": self.device,
                "yolo_raw_proposals": raw_n,
            }
            # Keep polygon for debug overlay (rotated outline).
            # Must be plain lists — report.json uses json.dumps (ndarray not allowed).
            if poly is not None:
                try:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    meta["obb_poly"] = [
                        [float(x), float(y)] for x, y in pts.tolist()
                    ]
                except Exception:  # noqa: BLE001
                    pass
            detections.append(
                Detection(
                    label=self.label,
                    bbox=bbox,
                    confidence=float(prop["confidence"]),
                    mask=mask,
                    metadata=meta,
                )
            )
        # Remember emit count for orchestrator / report even when list is empty
        self.last_emitted = len(detections)
        return detections

    @staticmethod
    def _extract_seg_masks(result: Any, height: int, width: int) -> list[np.ndarray] | None:
        """Return per-instance binary masks if YOLO-seg output is present."""
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None:
            return None
        data = getattr(masks_obj, "data", None)
        if data is None:
            return None
        try:
            arr = data.cpu().numpy()
        except Exception:  # noqa: BLE001
            return None
        if arr.ndim != 3 or arr.shape[0] == 0:
            return None
        out: list[np.ndarray] = []
        for i in range(arr.shape[0]):
            m = arr[i]
            if m.shape[0] != height or m.shape[1] != width:
                m = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            out.append((m > 0.5).astype(np.uint8) * 255)
        return out

    def _refine_bbox_mask(
        self,
        image: np.ndarray,
        bbox: BBox,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Intersect solid bbox with in-box residual; fall back if too empty."""
        crop = image[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if crop.size == 0:
            return mask, "bbox_fill"
        soft = FeatureParams(mode="fused", kernel=31, threshold=8, adaptive=True)
        residual = compute_feature_edges(crop, soft)
        _, residual_bin = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        residual_bin = cv2.dilate(residual_bin, kernel, iterations=1)

        region = mask[bbox.y : bbox.bottom, bbox.x : bbox.right]
        if residual_bin.shape[:2] != region.shape[:2]:
            residual_bin = cv2.resize(
                residual_bin,
                (region.shape[1], region.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            _, residual_bin = cv2.threshold(residual_bin, 1, 255, cv2.THRESH_BINARY)

        refined_region = cv2.bitwise_and(region, residual_bin)
        box_px = max(1, bbox.width * bbox.height)
        ref_px = int(np.count_nonzero(refined_region))
        if ref_px >= max(32, int(box_px * self.residual_keep_ratio)):
            out = mask.copy()
            out[bbox.y : bbox.bottom, bbox.x : bbox.right] = refined_region
            return out, "bbox_x_residual"
        return mask, "bbox_fill"
