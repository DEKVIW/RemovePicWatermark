"""EdgeSAM point-prompt segmentation for template capture (build-time only).

PS-style interaction: positive points on watermark, negative on background.
Does not run on every batch image — only when editing a style template.

Requires optional packages: torch, edge_sam (git+https://github.com/chongzhou96/EdgeSAM.git), yacs.
Weights default: workspace/models/sam/edge_sam.pth (auto-download from HuggingFace).

Note: upstream edge_sam imports mmdet only for optional RPN heads we never use;
we install lightweight stubs so point-prompt inference works without mmcv/mmdet.
"""

from __future__ import annotations

import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Official EdgeSAM checkpoint (≈ lightweight SAM for edge / desktop)
EDGE_SAM_WEIGHT_URL = (
    "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth"
)
EDGE_SAM_WEIGHT_NAME = "edge_sam.pth"


def _install_mmdet_stubs() -> None:
    """Allow importing edge_sam without full mmdet/mmengine (RPN path unused)."""
    if "mmdet" in sys.modules and "mmdet.models.dense_heads" in sys.modules:
        return

    def _pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = m
        return m

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            return _Dummy()

    # mmdet tree
    for name in (
        "mmdet",
        "mmdet.models",
        "mmdet.models.dense_heads",
        "mmdet.models.necks",
        "mmengine",
        "projects",
        "projects.EfficientDet",
        "projects.EfficientDet.efficientdet",
    ):
        if name not in sys.modules:
            _pkg(name)

    dens = sys.modules["mmdet.models.dense_heads"]
    dens.RPNHead = _Dummy  # type: ignore[attr-defined]
    dens.CenterNetUpdateHead = _Dummy  # type: ignore[attr-defined]
    necks = sys.modules["mmdet.models.necks"]
    necks.FPN = _Dummy  # type: ignore[attr-defined]
    mmengine = sys.modules["mmengine"]
    mmengine.ConfigDict = dict  # type: ignore[attr-defined]
    eff = sys.modules["projects.EfficientDet.efficientdet"]
    eff.BiFPN = _Dummy  # type: ignore[attr-defined]
    eff.EfficientDetSepBNHead = _Dummy  # type: ignore[attr-defined]


def _import_edge_sam():
    _install_mmdet_stubs()
    from edge_sam import SamPredictor, sam_model_registry

    return sam_model_registry, SamPredictor


@dataclass
class EdgeSamStatus:
    available: bool
    message: str
    weights: Path | None = None


def edgesam_status(weights_dir: Path | None = None) -> EdgeSamStatus:
    """Probe whether EdgeSAM can run (import + optional weight path)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return EdgeSamStatus(False, "未安装 torch（EdgeSAM 需要 PyTorch）")
    try:
        _import_edge_sam()
    except ImportError as exc:
        missing = str(exc)
        return EdgeSamStatus(
            False,
            "EdgeSAM 依赖不完整（"
            + missing
            + "）。可执行：\n"
            "pip install yacs loralib\n"
            "pip install git+https://github.com/chongzhou96/EdgeSAM.git",
        )
    except Exception as exc:  # noqa: BLE001
        return EdgeSamStatus(False, f"EdgeSAM 导入失败：{exc}")
    wdir = Path(weights_dir) if weights_dir else None
    weight = None
    if wdir is not None:
        cand = wdir / EDGE_SAM_WEIGHT_NAME
        if cand.is_file():
            weight = cand
    if weight is None:
        return EdgeSamStatus(
            True,
            "EdgeSAM 可用；首次使用将下载权重 edge_sam.pth",
            weights=None,
        )
    return EdgeSamStatus(True, f"EdgeSAM 就绪 · {weight.name}", weights=weight)


def resolve_edge_sam_weights(
    models_sam_dir: Path,
    *,
    download: bool = True,
    progress: Any | None = None,
) -> Path:
    """Return path to edge_sam.pth; download if missing and download=True."""
    models_sam_dir = Path(models_sam_dir)
    models_sam_dir.mkdir(parents=True, exist_ok=True)
    dest = models_sam_dir / EDGE_SAM_WEIGHT_NAME
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    if not download:
        raise FileNotFoundError(f"缺少 EdgeSAM 权重: {dest}")

    tmp = dest.with_suffix(".pth.partial")
    if progress:
        progress(f"下载 EdgeSAM 权重…\n{EDGE_SAM_WEIGHT_URL}")

    def _hook(block: int, block_size: int, total: int) -> None:
        if progress and total > 0 and block % 50 == 0:
            mb = (block * block_size) / (1024 * 1024)
            tot = total / (1024 * 1024)
            progress(f"下载 EdgeSAM… {mb:.1f}/{tot:.1f} MB")

    try:
        urllib.request.urlretrieve(EDGE_SAM_WEIGHT_URL, str(tmp), reporthook=_hook)
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    if not dest.is_file() or dest.stat().st_size < 1_000_000:
        raise RuntimeError("EdgeSAM 权重下载失败或不完整")
    return dest


class EdgeSamSegmenter:
    """Lazy-loaded EdgeSAM predictor with point prompts."""

    def __init__(
        self,
        weights: Path,
        *,
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights)
        self._device = device
        self._predictor = None
        self._embedded_key: tuple | None = None

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _ensure(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        sam_model_registry, SamPredictor = _import_edge_sam()

        device = self._resolve_device()
        # edge_sam registry: "edge_sam" / "default"
        model = sam_model_registry["edge_sam"](checkpoint=str(self.weights))
        model.to(device=device)
        model.eval()
        self._predictor = SamPredictor(model)
        self._device = device
        return self._predictor

    def reset_image(self) -> None:
        self._embedded_key = None
        if self._predictor is not None:
            try:
                self._predictor.reset_image()
            except Exception:  # noqa: BLE001
                pass

    def set_image_bgr(self, image_bgr: np.ndarray) -> None:
        """Embed image (BGR OpenCV). Reuses embedding if same shape+id."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty image")
        pred = self._ensure()
        key = (image_bgr.shape[0], image_bgr.shape[1], int(image_bgr.ctypes.data))
        if self._embedded_key == key:
            return
        # EdgeSAM / SAM expect RGB uint8 HWC
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb, image_format="RGB")
        self._embedded_key = key

    def predict_from_points(
        self,
        image_bgr: np.ndarray,
        points_xy: list[tuple[float, float]] | np.ndarray,
        labels: list[int] | np.ndarray,
        *,
        multimask: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run point-prompt segmentation.

        labels: 1 = foreground (watermark), 0 = background.
        Returns binary uint8 mask HxW (0/255) and stats.
        """
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        lbs = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(pts) == 0:
            raise ValueError("请至少点一个前景点（水印上）")
        if len(pts) != len(lbs):
            raise ValueError("points / labels 数量不一致")
        if not np.any(lbs == 1):
            raise ValueError("请至少点一个前景点（水印上，绿色）")

        self.set_image_bgr(image_bgr)
        pred = self._ensure()

        # Cap prompts — too many mixed FG/BG points confuse SAM into a blob
        if len(pts) > 24:
            pts = pts[-24:]
            lbs = lbs[-24:]

        # EdgeSAM predict API (SAM-compatible)
        masks, scores, _logits = pred.predict(
            point_coords=pts,
            point_labels=lbs,
            num_multimask_outputs=3 if multimask else 1,
            return_logits=False,
        )
        # masks: C x H x W bool/float
        if masks is None or len(masks) == 0:
            raise RuntimeError("EdgeSAM 未返回 mask")
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        best_i, binary, pick_score = self._pick_best_mask(masks, scores_arr, pts, lbs)
        # Light cleanup for template matching (keep small — do not fatten into a sausage)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        fill = float(np.count_nonzero(binary) / max(1, binary.size))
        stats = {
            "method": "edgesam",
            "score": float(pick_score),
            "model_score": float(scores_arr[best_i]) if len(scores_arr) else 0.0,
            "n_points": int(len(pts)),
            "n_fg": int(np.sum(lbs == 1)),
            "n_bg": int(np.sum(lbs == 0)),
            "fill_ratio": round(fill, 4),
            "device": str(self._device),
            "weights": str(self.weights),
        }
        if fill < 0.001:
            raise RuntimeError("AI 抠图结果几乎为空，请少点几个绿点（只点在笔画上）后再试")
        if fill > 0.85:
            # Common failure: whole crop selected — retry single-mask with fewer points
            fg_only = pts[lbs == 1]
            fg_labs = lbs[lbs == 1]
            if len(fg_only) >= 1:
                masks2, scores2, _ = pred.predict(
                    point_coords=fg_only[:8],
                    point_labels=fg_labs[:8],
                    num_multimask_outputs=3,
                    return_logits=False,
                )
                if masks2 is not None and len(masks2) > 0:
                    s2 = np.asarray(scores2, dtype=np.float64).reshape(-1)
                    bi2, bin2, sc2 = self._pick_best_mask(masks2, s2, fg_only[:8], fg_labs[:8])
                    f2 = float(np.count_nonzero(bin2) / max(1, bin2.size))
                    if 0.02 < f2 < fill:
                        binary = bin2
                        fill = f2
                        stats["score"] = float(sc2)
                        stats["fill_ratio"] = round(fill, 4)
                        stats["method"] = "edgesam_fg_retry"
        return binary, stats

    @staticmethod
    def _to_binary_mask(raw) -> np.ndarray:
        m = np.asarray(raw)
        if m.dtype == np.uint8:
            return (m > 127).astype(np.uint8) * 255
        return (m > 0.5).astype(np.uint8) * 255

    def _pick_best_mask(
        self,
        masks,
        scores_arr: np.ndarray,
        pts: np.ndarray,
        lbs: np.ndarray,
    ) -> tuple[int, np.ndarray, float]:
        """Prefer mask that hits FG points, avoids BG points, not a full-crop blob.

        Model IoU score alone often picks the largest blob (fill≈0.9) on pale text.
        """
        best_i = 0
        best_val = -1e9
        best_bin = self._to_binary_mask(masks[0])
        h, w = best_bin.shape[:2]
        for i in range(len(masks)):
            binary = self._to_binary_mask(masks[i])
            fill = float(np.count_nonzero(binary) / max(1, binary.size))
            fg_hit = bg_hit = fg_n = bg_n = 0
            for (x, y), lab in zip(pts, lbs):
                xi, yi = int(round(float(x))), int(round(float(y)))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                on = binary[yi, xi] > 0
                if int(lab) == 1:
                    fg_n += 1
                    fg_hit += int(on)
                else:
                    bg_n += 1
                    bg_hit += int(on)
            fg_cov = fg_hit / max(1, fg_n)
            bg_leak = bg_hit / max(1, bg_n)
            # Text watermark templates are rarely >60% of a tight crop
            fill_pen = 0.0
            if fill > 0.65:
                fill_pen = (fill - 0.65) * 3.0
            if fill < 0.01:
                fill_pen = 2.0
            model_s = float(scores_arr[i]) if i < len(scores_arr) else 0.0
            val = fg_cov * 2.0 - bg_leak * 2.5 - fill_pen + 0.15 * model_s
            if val > best_val:
                best_val = val
                best_i = i
                best_bin = binary
        return best_i, best_bin, best_val


_segmenter_cache: dict[str, EdgeSamSegmenter] = {}


def get_edgesam_segmenter(weights: Path, *, device: str | None = None) -> EdgeSamSegmenter:
    """Process-wide cache so repeated edits do not reload weights."""
    key = str(Path(weights).resolve())
    seg = _segmenter_cache.get(key)
    if seg is None:
        seg = EdgeSamSegmenter(weights, device=device)
        _segmenter_cache[key] = seg
    return seg


def segment_template_with_edgesam(
    image_bgr: np.ndarray,
    points_xy: list[tuple[float, float]],
    labels: list[int],
    *,
    models_sam_dir: Path,
    device: str | None = None,
    progress: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """High-level: ensure weights, run EdgeSAM, return mask + stats."""
    st = edgesam_status(models_sam_dir)
    if not st.available:
        raise RuntimeError(st.message)

    weights = resolve_edge_sam_weights(models_sam_dir, download=True, progress=progress)
    seg = get_edgesam_segmenter(weights, device=device)
    return seg.predict_from_points(image_bgr, points_xy, labels)
