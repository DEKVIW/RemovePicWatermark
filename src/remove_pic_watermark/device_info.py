"""Detect whether LaMa can use GPU (PyTorch CUDA) and build UI copy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class CudaStatus(str, Enum):
    OK = "ok"
    NO_TORCH_CUDA = "no_torch_cuda"  # CPU wheel or cuda not linked
    NO_NVIDIA = "no_nvidia"
    DRIVER_ISSUE = "driver_issue"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeviceProbe:
    status: CudaStatus
    cuda_available: bool
    torch_version: str
    torch_cuda_built: str | None
    gpu_name: str | None = None
    vram_gb: float | None = None
    detail: str = ""

    @property
    def short_gpu_label(self) -> str:
        if self.gpu_name and self.vram_gb is not None:
            return f"{self.gpu_name}（{self.vram_gb:g}GB）"
        if self.gpu_name:
            return self.gpu_name
        return "GPU"


@lru_cache(maxsize=1)
def probe_cuda() -> DeviceProbe:
    """Probe once per process (call clear_probe_cache to re-detect)."""
    return _probe_cuda_uncached()


def clear_probe_cache() -> None:
    probe_cuda.cache_clear()


def _probe_cuda_uncached() -> DeviceProbe:
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return DeviceProbe(
            status=CudaStatus.UNKNOWN,
            cuda_available=False,
            torch_version="?",
            torch_cuda_built=None,
            detail=f"无法导入 torch: {error}",
        )

    version = getattr(torch, "__version__", "?")
    built = getattr(torch.version, "cuda", None)
    is_cpu_wheel = "+cpu" in str(version).lower() or built in (None, "")

    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001
        return DeviceProbe(
            status=CudaStatus.DRIVER_ISSUE,
            cuda_available=False,
            torch_version=str(version),
            torch_cuda_built=str(built) if built else None,
            detail=str(error),
        )

    if available:
        name = None
        vram = None
        try:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / (1024**3), 1)
        except Exception as error:  # noqa: BLE001
            return DeviceProbe(
                status=CudaStatus.DRIVER_ISSUE,
                cuda_available=False,
                torch_version=str(version),
                torch_cuda_built=str(built) if built else None,
                detail=str(error),
            )
        return DeviceProbe(
            status=CudaStatus.OK,
            cuda_available=True,
            torch_version=str(version),
            torch_cuda_built=str(built) if built else None,
            gpu_name=name,
            vram_gb=vram,
            detail="CUDA ready",
        )

    # Not available — classify
    if is_cpu_wheel:
        status = CudaStatus.NO_TORCH_CUDA
        detail = "当前为 CPU 版 PyTorch，或未链接 CUDA"
    else:
        # CUDA build but is_available False → often driver / no device
        device_count = 0
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
        if device_count <= 0:
            status = CudaStatus.NO_NVIDIA
            detail = "CUDA 构建可用但未枚举到 GPU 设备"
        else:
            status = CudaStatus.DRIVER_ISSUE
            detail = "已检测到设备计数异常或驱动/CUDA 不匹配"

    return DeviceProbe(
        status=status,
        cuda_available=False,
        torch_version=str(version),
        torch_cuda_built=str(built) if built else None,
        detail=detail,
    )


def resolve_runtime_device(preference: str, probe: DeviceProbe | None = None) -> tuple[str, str, bool]:
    """Map UI preference to runtime device.

    Returns:
        (device, log_line, fell_back)
        device is 'cuda' or 'cpu'
    """
    pref = (preference or "auto").strip().lower()
    if pref in {"cuda", "gpu"}:
        pref = "gpu"
    if pref not in {"auto", "cpu", "gpu"}:
        pref = "auto"

    info = probe or probe_cuda()

    if pref == "cpu":
        note = "用户指定"
        if info.cuda_available:
            note = "用户指定（本机 GPU 可用）"
        return "cpu", f"设备：CPU（{note}）", False

    if pref == "gpu":
        if info.cuda_available:
            return "cuda", f"设备：GPU（{info.short_gpu_label}）", False
        reason = _reason_short(info.status)
        return "cpu", f"设备：GPU 不可用，已回退 CPU（{reason}）", True

    # auto
    if info.cuda_available:
        return "cuda", f"设备：自动 → GPU（{info.short_gpu_label}）", False
    reason = _reason_short(info.status)
    return "cpu", f"设备：自动 → CPU（{reason}）", False


def status_line(preference: str, probe: DeviceProbe | None = None) -> str:
    """One-line status (legacy prefix「当前：」for option bars)."""
    return "当前：" + header_device_caption(preference, probe)


def header_device_caption(
    preference: str,
    probe: DeviceProbe | None = None,
    *,
    extra: str = "",
) -> str:
    """Compact device line for page title captions (shared across modules).

    Examples:
      GPU · NVIDIA …（4GB）
      自动 → CPU · 未启用 CUDA（软件环境）
      CPU（本机 GPU 可用）
    Append ``extra`` for module-specific notes (e.g. ultralytics / weights).
    """
    pref = (preference or "auto").strip().lower()
    if pref in {"cuda", "gpu"}:
        pref = "gpu"
    if pref not in {"auto", "cpu", "gpu"}:
        pref = "auto"
    info = probe or probe_cuda()

    if pref == "cpu":
        base = "CPU（本机 GPU 可用）" if info.cuda_available else "CPU"
    elif pref == "gpu":
        if info.cuda_available:
            base = f"GPU · {info.short_gpu_label}"
        else:
            base = f"CPU · {_reason_status_suffix(info.status)}"
    else:
        # auto
        if info.cuda_available:
            base = f"自动 → GPU · {info.short_gpu_label}"
        else:
            base = f"自动 → CPU · {_reason_status_suffix(info.status)}"

    extra = (extra or "").strip()
    if extra:
        return f"{base} · {extra}"
    return base


def resolve_ultralytics_device(
    preference: str, probe: DeviceProbe | None = None
) -> str:
    """Map shared preference to ultralytics device: 'cpu' or '0'."""
    device, _log, _fb = resolve_runtime_device(preference, probe)
    return "0" if device == "cuda" else "cpu"


def device_tooltip(preference: str, probe: DeviceProbe | None = None) -> str:
    pref = (preference or "auto").strip().lower()
    info = probe or probe_cuda()

    if pref in {"cuda", "gpu"}:
        pref = "gpu"

    if pref == "auto":
        if info.cuda_available:
            head = f"自动 · 当前 GPU（{info.short_gpu_label}）"
        else:
            head = f"自动 · 当前 CPU（{_reason_short(info.status)}）"
    elif pref == "cpu":
        head = "使用 CPU"
    else:
        head = (
            f"使用 GPU（{info.short_gpu_label}）"
            if info.cuda_available
            else _reason_tooltip(info.status)
        )

    return head


def _reason_short(status: CudaStatus) -> str:
    return {
        CudaStatus.NO_TORCH_CUDA: "当前环境不支持 GPU",
        CudaStatus.NO_NVIDIA: "未检测到 NVIDIA 显卡",
        CudaStatus.DRIVER_ISSUE: "显卡驱动异常",
        CudaStatus.UNKNOWN: "无法使用 GPU",
        CudaStatus.OK: "就绪",
    }.get(status, "无法使用 GPU")


def _reason_status_suffix(status: CudaStatus) -> str:
    return _reason_short(status)


def _reason_tooltip(status: CudaStatus) -> str:
    return {
        CudaStatus.NO_TORCH_CUDA: "当前无法使用 GPU，已使用 CPU。",
        CudaStatus.NO_NVIDIA: "未检测到可用的 NVIDIA 显卡，请使用 CPU。",
        CudaStatus.DRIVER_ISSUE: "显卡驱动异常，已使用 CPU。请更新驱动后重试。",
        CudaStatus.UNKNOWN: "无法确认 GPU 状态，已使用 CPU。",
        CudaStatus.OK: "GPU 可用。",
    }.get(status, "无法使用 GPU，已使用 CPU。")


def fallback_dialog_text(probe: DeviceProbe) -> str:
    if probe.status == CudaStatus.NO_NVIDIA:
        return (
            "未检测到可用的 NVIDIA 显卡，已使用 CPU。\n\n"
            "GPU 加速需要 NVIDIA 显卡。"
        )
    if probe.status == CudaStatus.NO_TORCH_CUDA:
        return "当前环境无法使用 GPU，已改用 CPU。"
    if probe.status == CudaStatus.DRIVER_ISSUE:
        return (
            "GPU 初始化失败，已改用 CPU。\n\n"
            "请更新 NVIDIA 显卡驱动后重试。"
        )
    return "无法使用 GPU，已改用 CPU。"
