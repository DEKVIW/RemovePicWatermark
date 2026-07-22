# GPU / CUDA self-check for RemovePicWatermark (ASCII-only comments for PS 5.1).
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\check_gpu.ps1

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== 1) nvidia-smi ===" -ForegroundColor Cyan
nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "nvidia-smi failed: no NVIDIA driver or not an NVIDIA GPU." -ForegroundColor Yellow
}

Write-Host "`n=== 2) Win32_VideoController ===" -ForegroundColor Cyan
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, Status | Format-List

Write-Host "`n=== 3) PyTorch in .venv ===" -ForegroundColor Cyan
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "No .venv python, fallback to 'python' on PATH" -ForegroundColor Yellow
    $Py = "python"
}
Write-Host "Python: $Py"

& $Py -c @"
import sys
try:
    import torch
except Exception as e:
    print('import torch failed:', e)
    sys.exit(1)
print('torch', torch.__version__)
print('cuda_built', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('vram_GB', round(props.total_memory / 1024**3, 2))
    print('capability', torch.cuda.get_device_capability(0))
    print('RESULT: GPU OK for this Python env')
else:
    print('RESULT: GPU NOT available to this torch build (CPU only or driver mismatch)')
"@

Write-Host "`nDone. See docs/GPU加速与设备自检.md" -ForegroundColor Green
