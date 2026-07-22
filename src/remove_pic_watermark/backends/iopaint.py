"""LaMa inpainting — in-process, project-local, **no iopaint/diffusers import**.

Why not IOPaint package at runtime (especially frozen GUI):
  - ``iopaint.model.utils`` imports all ``diffusers`` schedulers
  - Under PyInstaller (runw.exe), lazy imports crash with
    ``name 'name' is not defined`` / ``stderr.flush`` NoneType
  - big-lama.pt is already a TorchScript JIT model; we only need torch + cv2

Approach:
  - Load ``big-lama.pt`` via ``Path.read_bytes()`` + ``torch.jit.load(BytesIO)``
    (Chinese paths safe)
  - Replicate iopaint LaMa preprocess / pad-mod-8 / HD crop / forward / postprocess
  - Write results with unicode-safe image_io
"""

from __future__ import annotations

import io
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..image_io import read_image, write_image


def executable_available(executable: str | None = None) -> bool:
    """CLI helper; in-process path only needs torch."""
    if executable is not None:
        return Path(executable).exists() or shutil.which(executable) is not None
    return find_executable() is not None or _torch_available()


def find_executable() -> str | None:
    from_path = shutil.which("iopaint")
    if from_path is not None:
        return from_path
    local_executable = Path(sys.prefix) / (
        "Scripts/iopaint.exe" if sys.platform == "win32" else "bin/iopaint"
    )
    if local_executable.exists():
        return str(local_executable)
    return None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def path_has_non_ascii(path: Path | str) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def project_root_from_path(path: Path) -> Path:
    current = path.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "remove_pic_watermark").is_dir() or (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    if current.name == "workspace":
        return current.parent
    if current.parent.name == "workspace":
        return current.parent.parent
    return current if current.is_dir() else current.parent


def resolve_model_dir(preferred: Path | None = None, project_root: Path | None = None) -> Path:
    """Always prefer project workspace/models so weights stay in-repo."""
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(preferred)
    if project_root is not None:
        candidates.append(project_root / "workspace" / "models")
        candidates.append(project_root / "data" / "iopaint-models")
    # Frozen onedir: models next to exe
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "workspace" / "models")
    for path in candidates:
        ckpt = path / "torch" / "hub" / "checkpoints" / "big-lama.pt"
        if ckpt.exists() and ckpt.stat().st_size > 1_000_000:
            return path
    if project_root is not None:
        target = project_root / "workspace" / "models"
        target.mkdir(parents=True, exist_ok=True)
        return target
    if preferred is not None:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    if getattr(sys, "frozen", False):
        target = Path(sys.executable).resolve().parent / "workspace" / "models"
        target.mkdir(parents=True, exist_ok=True)
        return target
    fallback = Path.home() / ".cache" / "rpw_iopaint_models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def ensure_lama_checkpoint(model_dir: Path, search_roots: list[Path] | None = None) -> Path | None:
    target = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    if target.exists() and target.stat().st_size > 1_000_000:
        return target

    search: list[Path] = list(search_roots or [])
    if getattr(sys, "frozen", False):
        search.append(
            Path(sys.executable).resolve().parent
            / "workspace"
            / "models"
            / "torch"
            / "hub"
            / "checkpoints"
            / "big-lama.pt"
        )
    search.extend(
        [
            Path("data/iopaint-models/torch/hub/checkpoints/big-lama.pt"),
            Path("workspace/models/torch/hub/checkpoints/big-lama.pt"),
            Path(r"C:\tmp\rpw_iopaint_models\torch\hub\checkpoints\big-lama.pt"),
        ]
    )
    for candidate in search:
        try:
            if candidate.exists() and candidate.stat().st_size > 1_000_000:
                target.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != target.resolve():
                    shutil.copy2(candidate, target)
                return target
        except OSError:
            continue
    return target if target.exists() else None


def build_command(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    model: str = "lama",
    device: str = "cuda",
    model_dir: Path | None = None,
    executable: str | None = None,
) -> list[str]:
    """Legacy CLI command builder (kept for `iopaint-command` subcommand)."""
    command = [
        executable or find_executable() or "iopaint",
        "run",
        f"--model={model}",
        f"--device={device}",
        f"--image={image_dir}",
        f"--mask={mask_dir}",
        f"--output={output_dir}",
    ]
    if model_dir is not None:
        command.append(f"--model-dir={model_dir}")
    return command


# ---------------------------------------------------------------------------
# Pure TorchScript LaMa (no iopaint / diffusers)
# ---------------------------------------------------------------------------


def _ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _norm_img(np_img: np.ndarray) -> np.ndarray:
    if np_img.ndim == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    return np_img.astype("float32") / 255.0


def _pad_img_to_modulo(img: np.ndarray, mod: int = 8) -> np.ndarray:
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_h = _ceil_modulo(height, mod)
    out_w = _ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, out_h - height), (0, out_w - width), (0, 0)),
        mode="symmetric",
    )


def _boxes_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """mask: HxW or HxWx1, 0~255."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask.astype(np.uint8), 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[np.ndarray] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h], dtype=int)
        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)
    return boxes


def _crop_box(
    image: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    box_h = int(box[3] - box[1])
    box_w = int(box[2] - box[0])
    cx = int((box[0] + box[2]) // 2)
    cy = int((box[1] + box[3]) // 2)
    img_h, img_w = image.shape[:2]

    w = box_w + margin * 2
    h = box_h + margin * 2
    _l = cx - w // 2
    _r = cx + w // 2
    _t = cy - h // 2
    _b = cy + h // 2

    l = max(_l, 0)
    r = min(_r, img_w)
    t = max(_t, 0)
    b = min(_b, img_h)

    if _l < 0:
        r += abs(_l)
    if _r > img_w:
        l -= _r - img_w
    if _t < 0:
        b += abs(_t)
    if _b > img_h:
        t -= _b - img_h

    l = max(l, 0)
    r = min(r, img_w)
    t = max(t, 0)
    b = min(b, img_h)

    crop_img = image[t:b, l:r, :]
    crop_mask = mask[t:b, l:r]
    return crop_img, crop_mask, [l, t, r, b]


def _load_jit_from_path(model_path: Path, device: str):
    """Load torchscript via bytes — works with Chinese filesystem paths."""
    import torch

    data = model_path.read_bytes()
    model = torch.jit.load(io.BytesIO(data), map_location="cpu")
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to("cuda")
        device_str = "cuda"
    else:
        model = model.to("cpu")
        device_str = "cpu"
    model.eval()
    return model, device_str


class _InProcessLaMa:
    """Standalone big-lama TorchScript runner (iopaint-compatible I/O)."""

    pad_mod = 8
    crop_trigger_size = 800
    crop_margin = 128

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        try:
            from ..stdio_fix import ensure_stdio

            ensure_stdio()
        except Exception:
            pass
        import torch

        self.model, self.device_str = _load_jit_from_path(model_path, device)
        self.device = torch.device(self.device_str)

    def _forward_core(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image RGB HxWx3 uint8, mask HxW uint8 → BGR HxWx3 uint8.

        TorchScript LaMa is picky: mask must be 1×1×H×W float32 in {0,1},
        image 1×3×H×W float32 in [0,1], same spatial size, contiguous.
        """
        import torch

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"LaMa expects RGB HxWx3, got {getattr(image_rgb, 'shape', None)}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.ascontiguousarray(mask)
        origin_h, origin_w = image_rgb.shape[:2]
        if mask.shape[0] != origin_h or mask.shape[1] != origin_w:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (origin_w, origin_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Empty mask → no-op (avoids JIT mul crash on all-zero / degenerate)
        if int(np.count_nonzero(mask)) < 4:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Binary single-channel mask only
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        pad_image = _pad_img_to_modulo(
            np.ascontiguousarray(image_rgb), self.pad_mod
        )
        pad_mask = _pad_img_to_modulo(mask_u8[:, :, np.newaxis], self.pad_mod)

        # Force shapes: image CHW float32 [0,1]; mask 1HW float32 {0,1}
        image_n = _norm_img(pad_image).astype(np.float32, copy=False)
        mask_n = (pad_mask[:, :, 0] > 0).astype(np.float32)[np.newaxis, ...]  # 1,H,W

        if image_n.shape[0] != 3 or mask_n.shape[0] != 1:
            raise ValueError(
                f"LaMa tensor channel mismatch: image {image_n.shape} mask {mask_n.shape}"
            )
        if image_n.shape[1:] != mask_n.shape[1:]:
            raise ValueError(
                f"LaMa spatial mismatch: image {image_n.shape} mask {mask_n.shape}"
            )

        image_t = torch.from_numpy(np.ascontiguousarray(image_n)).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_n)).unsqueeze(0)
        image_t = image_t.to(self.device, dtype=torch.float32)
        mask_t = mask_t.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            try:
                inpainted = self.model(image_t, mask_t)
            except Exception as exc:  # noqa: BLE001
                # Surface a short product message; keep original for logs
                raise RuntimeError(
                    f"高质量修补失败（输入尺寸 {origin_w}×{origin_h}）：{exc}"
                ) from exc

        cur = inpainted[0].permute(1, 2, 0).detach().float().cpu().numpy()
        cur = np.clip(cur * 255.0, 0, 255).astype("uint8")
        cur = cur[0:origin_h, 0:origin_w, :]
        cur_bgr = cv2.cvtColor(cur, cv2.COLOR_RGB2BGR)

        # Keep unmasked original pixels (BGR)
        mask_u = mask_u8[:origin_h, :origin_w]
        m = (mask_u > 0).astype(np.float32)[:, :, np.newaxis]
        image_bgr = cv2.cvtColor(image_rgb[:origin_h, :origin_w], cv2.COLOR_RGB2BGR)
        out = cur_bgr.astype(np.float32) * m + image_bgr.astype(np.float32) * (1.0 - m)
        return np.clip(out, 0, 255).astype("uint8")

    def inpaint_bgr(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        if int(np.count_nonzero(mask_u8)) < 4:
            return image_bgr.copy()

        # HD crop strategy for large images (same idea as iopaint LaMa)
        if max(image_rgb.shape[:2]) > self.crop_trigger_size:
            boxes = _boxes_from_mask(mask_u8)
            if boxes:
                result = image_bgr.copy()
                for box in boxes:
                    # Skip tiny / degenerate boxes
                    if int(box[2] - box[0]) < 4 or int(box[3] - box[1]) < 4:
                        continue
                    crop_rgb, crop_mask, (l, t, r, b) = _crop_box(
                        image_rgb, mask_u8, box, self.crop_margin
                    )
                    if crop_rgb.size == 0 or crop_rgb.shape[0] < 8 or crop_rgb.shape[1] < 8:
                        continue
                    if int(np.count_nonzero(crop_mask)) < 4:
                        continue
                    crop_bgr = self._forward_core(crop_rgb, crop_mask)
                    result[t:b, l:r, :] = crop_bgr
                return result

        return self._forward_core(image_rgb, mask_u8)


_MODEL_CACHE: dict[str, _InProcessLaMa] = {}


def get_lama_engine(model_path: Path, device: str = "cpu") -> _InProcessLaMa:
    try:
        from ..stdio_fix import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    key = f"{model_path.resolve()}::{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _InProcessLaMa(model_path, device=device)
    return _MODEL_CACHE[key]


def run_iopaint(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    model: str = "lama",
    device: str = "cpu",
    model_dir: Path | None = None,
    executable: str | None = None,
    search_model_roots: list[Path] | None = None,
    project_root: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """In-process LaMa over a folder of images/masks. All I/O stays in project."""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    if model != "lama":
        _log(f"当前内置引擎仅稳定支持 lama，收到 model={model}，仍按 lama 处理")

    root = project_root or project_root_from_path(image_dir)
    resolved_model_dir = resolve_model_dir(model_dir or (root / "workspace" / "models"), root)
    ckpt = ensure_lama_checkpoint(resolved_model_dir, search_model_roots)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError("未找到修补模型，请检查安装是否完整。")

    _log(f"加载修补模型（约 {ckpt.stat().st_size // (1024 * 1024)} MB）")
    if progress:
        progress(0, 1, "加载修补模型…")

    engine = get_lama_engine(ckpt, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    )
    work_list: list[tuple[Path, Path]] = []
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            candidates = list(mask_dir.glob(f"{image_path.stem}.*"))
            mask_path = candidates[0] if candidates else mask_path
        if not mask_path.exists():
            continue
        mask = read_image(mask_path)
        mask_gray = mask[:, :, 0] if mask.ndim == 3 else mask
        if np.any(mask_gray > 0):
            work_list.append((image_path, mask_path))

    repair_total = max(1, len(work_list))
    processed = 0
    skipped = 0

    if not work_list:
        if progress:
            progress(1, 1, "无需 LaMa 修补")
        _log("没有非空 mask，跳过 LaMa")
        return {"processed": 0, "skipped": len(images), "output_dir": str(output_dir)}

    for image_path, mask_path in work_list:
        if progress:
            progress(
                processed,
                repair_total,
                f"LaMa 修补中 ({processed + 1}/{repair_total}) {image_path.name}",
            )
        image = read_image(image_path)
        mask = read_image(mask_path)
        mask_gray = mask[:, :, 0] if mask.ndim == 3 else mask
        _log(f"LaMa 修补: {image_path.name} mask_px={int(np.count_nonzero(mask_gray))}")
        result = engine.inpaint_bgr(image, mask_gray)
        out_path = output_dir / f"{image_path.stem}.png"
        write_image(out_path, result)
        processed += 1
        if progress:
            progress(
                processed,
                repair_total,
                f"LaMa 已完成 ({processed}/{repair_total}) {image_path.name}",
            )

    for image_path in images:
        if any(image_path == w[0] for w in work_list):
            continue
        out = output_dir / f"{image_path.stem}.png"
        if not out.exists():
            write_image(out, read_image(image_path))
            skipped += 1

    _log(f"完成: 修补 {processed} 张，跳过 {skipped} 张 → {output_dir}")
    return {"processed": processed, "skipped": skipped, "output_dir": str(output_dir)}
