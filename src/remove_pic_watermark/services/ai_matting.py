"""AI matting for template capture — BiRefNet only (local project weights).

Weights live under ``workspace/models/birefnet/`` so packaging can ship the
folder next to the app. First run may download from Hugging Face into that
directory when network is available.

Pale semi-transparent watermarks often produce weak pure-matting masks.
We fuse the model soft-mask with multi-scale residual (tophat) so letter
strokes remain when the network under-fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ProgressCb = Callable[[str], None] | None

# Single supported backend (RMBG / rembg abandoned)
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_LABEL = "BiRefNet"
BACKEND = "birefnet"


@dataclass
class AiMattingStatus:
    available: bool
    message: str
    backends: list[str]
    model_dir: str = ""
    local_ready: bool = False


def birefnet_dir(workspace_root: Path | None = None) -> Path:
    """Project-local weights dir (package-friendly)."""
    if workspace_root is not None:
        return Path(workspace_root) / "models" / "birefnet"
    try:
        from ..workspace import get_workspace

        return get_workspace().birefnet_dir
    except Exception:  # noqa: BLE001
        from ..paths import app_root

        return app_root() / "workspace" / "models" / "birefnet"


def _local_model_ready(model_dir: Path) -> bool:
    """True when a HF-style snapshot exists under model_dir."""
    if not model_dir.is_dir():
        return False
    if (model_dir / "config.json").is_file():
        return True
    # nested hub layout fallback
    for p in model_dir.rglob("config.json"):
        if p.is_file():
            return True
    return False


def _resolve_load_path(model_dir: Path) -> Path | None:
    if (model_dir / "config.json").is_file():
        return model_dir
    for p in model_dir.rglob("config.json"):
        return p.parent
    return None


def ai_matting_status(workspace_root: Path | None = None) -> AiMattingStatus:
    """Lightweight status check — does **not** import torch/transformers.

    Uses importlib.find_spec so GUI startup never pays for heavy ML loads.
    Actual model load still happens on first AI matting run.
    """
    import importlib.util

    model_dir = birefnet_dir(workspace_root)
    local = _local_model_ready(model_dir)
    for mod in ("torch", "transformers", "kornia"):
        if importlib.util.find_spec(mod) is None:
            return AiMattingStatus(
                False, "AI 抠图组件未就绪", [], str(model_dir), local
            )
    msg = "AI 抠图就绪" if local else "AI 抠图可用（首次将下载模型）"
    return AiMattingStatus(True, msg, [BACKEND], str(model_dir), local)


_model_cache: dict[str, Any] = {}


def _resolve_torch_device(preference: str | None = None) -> str:
    """Honor shared menu preference (auto|cpu|gpu) via device_info."""
    try:
        from ..device_info import resolve_runtime_device

        device, _log, _fb = resolve_runtime_device(preference or "auto")
        return device  # 'cuda' | 'cpu'
    except Exception:  # noqa: BLE001
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"


def ensure_birefnet_weights(
    model_dir: Path | None = None,
    *,
    progress: ProgressCb = None,
) -> Path:
    """Ensure BiRefNet files exist under project models/birefnet; return load path."""
    root = Path(model_dir) if model_dir is not None else birefnet_dir()
    root.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_load_path(root)
    if resolved is not None:
        return resolved

    if progress:
        progress(f"下载 {BIREFNET_LABEL} → {root}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=BIREFNET_REPO,
            local_dir=str(root),
            local_dir_use_symlinks=False,
        )
    except Exception:
        # Fallback: let transformers download into root via cache_dir style
        if progress:
            progress(f"snapshot 失败，改用 from_pretrained 缓存…")
        import torch  # noqa: F401
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_REPO, trust_remote_code=True
        )
        try:
            model.save_pretrained(str(root))
        except Exception:  # noqa: BLE001
            pass
        del model

    resolved = _resolve_load_path(root)
    if resolved is None:
        # last resort: remote id (still works online)
        return root
    return resolved


def _load_model(
    *,
    model_dir: Path | None = None,
    progress: ProgressCb = None,
    device_preference: str | None = None,
):
    import torch
    from transformers import AutoModelForImageSegmentation

    dev = _resolve_torch_device(device_preference)
    cache_key = f"birefnet::{dev}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    root = Path(model_dir) if model_dir is not None else birefnet_dir()
    load_path = ensure_birefnet_weights(root, progress=progress)
    if progress:
        progress(f"加载 {BIREFNET_LABEL}…\n{load_path}")

    local_only = _local_model_ready(root)
    try:
        if local_only and _resolve_load_path(root) is not None:
            model = AutoModelForImageSegmentation.from_pretrained(
                str(load_path),
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            model = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET_REPO,
                trust_remote_code=True,
            )
            try:
                model.save_pretrained(str(root))
            except Exception:  # noqa: BLE001
                pass
    except Exception:
        # Online fallback if local incomplete
        if progress:
            progress(f"本地加载失败，尝试在线 {BIREFNET_REPO}…")
        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_REPO, trust_remote_code=True
        )
        try:
            model.save_pretrained(str(root))
        except Exception:  # noqa: BLE001
            pass

    model.to(dev)
    model.eval()
    _model_cache[cache_key] = (model, dev)
    return _model_cache[cache_key]


def _predict_soft_mask(
    crop_bgr: np.ndarray,
    *,
    model_dir: Path | None = None,
    progress: ProgressCb = None,
    device_preference: str | None = None,
) -> np.ndarray:
    """Return float32 soft mask HxW in [0,1]."""
    import torch
    from PIL import Image
    from torchvision import transforms

    model, dev = _load_model(
        model_dir=model_dir,
        progress=progress,
        device_preference=device_preference,
    )
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    h, w = crop_bgr.shape[:2]

    tfm = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    inp = tfm(pil).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            pred = out[-1]
        elif hasattr(out, "logits"):
            pred = out.logits
        else:
            pred = out
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.sigmoid().float().cpu()
        while pred.ndim > 2:
            pred = pred[0]
        soft = pred.numpy()
    soft = cv2.resize(soft.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    soft = np.clip(soft, 0.0, 1.0)
    return soft


def _fuse_with_residual(
    crop_bgr: np.ndarray,
    soft: np.ndarray,
    *,
    prefer_ai: bool = True,
) -> tuple[np.ndarray, str, float]:
    """Fuse AI soft mask with multi-scale residual for pale text logos."""
    from .template_builder import (
        _keep_primary_watermark_components,
        _letter_residual_mask,
    )

    res_bin, acc = _letter_residual_mask(crop_bgr, percentile=76.0)
    r = (res_bin > 0).astype(np.float32)
    p = soft.astype(np.float32)
    ai_mean = float(p.mean())
    res_fill = float(r.mean())

    if ai_mean < 0.03 and res_fill >= 0.05:
        mask = res_bin.copy()
        method = "residual_fallback"
    elif ai_mean > 0.15:
        score = 0.55 * p + 0.45 * r if prefer_ai else 0.4 * p + 0.6 * r
        thr = max(0.28, float(np.percentile(score, 68)))
        mask = (score >= thr).astype(np.uint8) * 255
        method = "ai_residual_fuse"
    else:
        score = np.maximum(r, p * 0.85)
        thr = max(0.35, float(np.percentile(score, 72)))
        mask = (score >= thr).astype(np.uint8) * 255
        method = "residual_ai_boost"

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    )
    mask = _keep_primary_watermark_components(mask)
    fill = float(np.count_nonzero(mask) / max(1, mask.size))

    if fill < 0.04 and res_fill >= 0.04:
        mask = res_bin
        method = "residual_guard"
        fill = res_fill
    if fill > 0.72:
        tight = ((p > 0.45) & (r > 0)).astype(np.uint8) * 255
        if tight.mean() / 255.0 >= 0.04:
            mask = _keep_primary_watermark_components(tight)
            method = "ai_residual_tight"
            fill = float(np.count_nonzero(mask) / max(1, mask.size))

    return mask, method, fill


def extract_mask_ai(
    crop_bgr: np.ndarray,
    *,
    backend: str = BACKEND,
    dilate: int = 1,
    progress: ProgressCb = None,
    model_dir: Path | None = None,
    device_preference: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run BiRefNet matting + residual fusion → binary template mask."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("empty crop")

    # Ignore legacy backend names; only BiRefNet
    _ = backend
    st = ai_matting_status()
    if not st.available:
        raise RuntimeError(st.message)

    root = Path(model_dir) if model_dir is not None else birefnet_dir()
    dev = _resolve_torch_device(device_preference)
    if progress:
        progress(f"推理 {BIREFNET_LABEL}（{dev}）…")
    soft = _predict_soft_mask(
        crop_bgr,
        model_dir=root,
        progress=progress,
        device_preference=device_preference,
    )

    mask, method, fill = _fuse_with_residual(crop_bgr, soft)
    if dilate > 0:
        d = min(int(dilate), 2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
        from .template_builder import _keep_primary_watermark_components

        mask = _keep_primary_watermark_components(mask)
        fill = float(np.count_nonzero(mask) / max(1, mask.size))

    stats = {
        "fill_ratio": round(fill, 4),
        "method": f"{method}+birefnet",
        "backend": BACKEND,
        "model_id": BIREFNET_REPO,
        "model_dir": str(root),
        "ai_soft_mean": round(float(soft.mean()), 4),
        "template_size": [int(mask.shape[1]), int(mask.shape[0])],
        "device": dev,
    }
    if fill < 0.02:
        raise RuntimeError(
            "AI 抠图结果过空。可改用自动提取或手涂，"
            f"（soft_mean={stats['ai_soft_mean']}）"
        )
    return mask, stats
