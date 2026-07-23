from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remove_pic_watermark.device_info import (
    CudaStatus,
    DeviceProbe,
    clear_probe_cache,
    resolve_runtime_device,
    status_line,
)


class DeviceInfoTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_probe_cache()

    def tearDown(self) -> None:
        clear_probe_cache()

    def test_auto_uses_cuda_when_available(self) -> None:
        probe = DeviceProbe(
            status=CudaStatus.OK,
            cuda_available=True,
            torch_version="2.0+cu124",
            torch_cuda_built="12.4",
            gpu_name="NVIDIA GeForce GTX 1650",
            vram_gb=4.0,
        )
        device, log_line, fell = resolve_runtime_device("auto", probe)
        self.assertEqual(device, "cuda")
        self.assertFalse(fell)
        self.assertIn("GPU", log_line)
        self.assertIn("1650", status_line("auto", probe))

    def test_auto_falls_back_cpu(self) -> None:
        probe = DeviceProbe(
            status=CudaStatus.NO_TORCH_CUDA,
            cuda_available=False,
            torch_version="2.12.0+cpu",
            torch_cuda_built=None,
        )
        device, log_line, fell = resolve_runtime_device("auto", probe)
        self.assertEqual(device, "cpu")
        self.assertFalse(fell)
        self.assertIn("CPU", log_line)
        self.assertIn("未启用 CUDA", status_line("auto", probe))

    def test_forced_gpu_falls_back(self) -> None:
        probe = DeviceProbe(
            status=CudaStatus.NO_NVIDIA,
            cuda_available=False,
            torch_version="2.0+cu124",
            torch_cuda_built="12.4",
        )
        device, log_line, fell = resolve_runtime_device("gpu", probe)
        self.assertEqual(device, "cpu")
        self.assertTrue(fell)
        self.assertIn("回退", log_line)

    def test_forced_cpu(self) -> None:
        probe = DeviceProbe(
            status=CudaStatus.OK,
            cuda_available=True,
            torch_version="2.0+cu124",
            torch_cuda_built="12.4",
            gpu_name="NVIDIA GeForce GTX 1650",
            vram_gb=4.0,
        )
        device, log_line, fell = resolve_runtime_device("cpu", probe)
        self.assertEqual(device, "cpu")
        self.assertFalse(fell)
        self.assertIn("用户指定", log_line)


if __name__ == "__main__":
    unittest.main()
