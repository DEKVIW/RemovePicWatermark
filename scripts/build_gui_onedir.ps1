# Build onedir GUI package for RemovePicWatermark.
# IMPORTANT: avoid Chinese string literals in this .ps1 (encoding issues on PS 5.1).
#
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\build_gui_onedir.ps1
# Optional:
#   -SkipInstall
#   -WithModels     copy LaMa / YOLO weights when present (default: on if files exist)
#   -SkipZip
#   -NoModels       skip all weight copies

param(
    [switch]$SkipInstall,
    [switch]$WithModels,
    [switch]$NoModels,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
    Write-Host "Warning: .venv not found, using system python"
}

Write-Host "Project root: $Root"
Write-Host "Python: $Py"

$env:PYTHONPATH = (Join-Path $Root "src")
$ver = & $Py -c "from remove_pic_watermark import __version__; print(__version__)"
if (-not $ver) { $ver = "0.0.0" }
$ver = $ver.Trim()
Write-Host "Version: $ver"

if (-not $SkipInstall) {
    Write-Host "==> Installing GUI + packaging deps"
    & $Py -m pip install -q "PySide6>=6.6" "PySide6-Fluent-Widgets>=1.6" "pyinstaller>=6.0"
    # ensure runtime deps used by GUI
    & $Py -m pip install -q "numpy>=1.24" "opencv-python>=4.8" "Pillow>=10.0"
    # YOLO detect + train (torch/torchvision should already be in the venv)
    Write-Host "==> Installing ultralytics (YOLO) + matplotlib (required by ultralytics)"
    & $Py -m pip install -q "ultralytics>=8.0" "PyYAML>=6.0" "lap>=0.5.12" "matplotlib>=3.7"
    # BiRefNet AI matting on styles page (optional feature, OOB when weights present)
    Write-Host "==> Installing matting deps (BiRefNet)"
    & $Py -m pip install -q "transformers>=4.40" "kornia>=0.7" "einops>=0.7" "timm>=0.9" "huggingface_hub>=0.20"
}

$DistRoot = Join-Path $Root "dist"
$Build = Join-Path $Root "build"
$Spec = Join-Path $Root "packaging\remove_pic_watermark_gui.spec"
$AppName = "RemovePicWatermark"
$OutDir = Join-Path $DistRoot $AppName

Write-Host "==> Cleaning previous dist/$AppName"
if (Test-Path $OutDir) {
    Remove-Item -Recurse -Force $OutDir
}

Write-Host "==> PyInstaller onedir (may take a long time)"
& $Py -m PyInstaller --noconfirm --clean --distpath $DistRoot --workpath $Build $Spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$ExePath = Join-Path $OutDir "$AppName.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build finished but $AppName.exe not found under $OutDir"
}

Write-Host "==> Writing meta + start.bat + icons"
& $Py (Join-Path $Root "packaging\write_dist_meta.py") $OutDir $ver
# Also place icon beside exe (some Windows shells prefer it)
foreach ($name in @("app.ico", "app.png")) {
    $src = Join-Path $Root "packaging\$name"
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $OutDir $name) -Force
    }
}

# PyInstaller freezes module-level for-loop names in torch._numpy._ufuncs
# (NameError: name 'name' is not defined even with globals()[name]).
Write-Host "==> Patching torch._numpy._ufuncs (function attach for freeze)"
$ufuncsPy = Join-Path $OutDir "_internal\torch\_numpy\_ufuncs.py"
$patchPy = Join-Path $Root "packaging\patch_torch_ufuncs.py"
if ((Test-Path $ufuncsPy) -and (Test-Path $patchPy)) {
    & $Py $patchPy $ufuncsPy
}
else {
    Write-Host "  NOTE: skip ufuncs patch (missing file)"
}

# Empty workspace skeleton next to exe — NO test profiles/jobs/yolo_train data
$ws = Join-Path $OutDir "workspace"
New-Item -ItemType Directory -Force -Path (Join-Path $ws "profiles") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ws "jobs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ws "models\torch\hub\checkpoints") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ws "models\yolo") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ws "models\birefnet") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ws "yolo_train") | Out-Null
# Marker: never auto-seed built-in styles into the packaged empty library
$seedMarker = Join-Path $ws ".builtins_seeded"
Set-Content -Path $seedMarker -Value "packaged-empty`n" -Encoding UTF8
Write-Host "==> workspace/profiles starts empty (no test styles)"

# Copy weights when present (default). -NoModels skips; -WithModels is kept for compatibility.
$doModels = -not $NoModels
if ($doModels) {
    Write-Host "==> Copying model weights if present"

    # LaMa (big-lama.pt) — already used by previous releases
    $lamaCandidates = @(
        (Join-Path $Root "workspace\models\torch\hub\checkpoints\big-lama.pt"),
        (Join-Path $Root "data\iopaint-models\torch\hub\checkpoints\big-lama.pt"),
        "C:\tmp\rpw_iopaint_models\torch\hub\checkpoints\big-lama.pt"
    )
    $lamaDst = Join-Path $ws "models\torch\hub\checkpoints\big-lama.pt"
    $lamaOk = $false
    foreach ($src in $lamaCandidates) {
        if (Test-Path $src) {
            $len = (Get-Item $src).Length
            if ($len -gt 1000000) {
                Write-Host "  LaMa: $src ($([math]::Round($len/1MB)) MB)"
                Copy-Item $src $lamaDst -Force
                $lamaOk = $true
                break
            }
        }
    }
    if (-not $lamaOk) {
        Write-Host "  WARNING: big-lama.pt not found"
    }

    # YOLO base weights for training (ultralytics looks in cwd = app root)
    $yoloBasePairs = @(
        @("yolov8n.pt", @((Join-Path $Root "yolov8n.pt"), (Join-Path $Root "workspace\models\yolo\yolov8n.pt"))),
        @("yolov8n-obb.pt", @((Join-Path $Root "yolov8n-obb.pt"), (Join-Path $Root "workspace\models\yolo\yolov8n-obb.pt")))
    )
    foreach ($pair in $yoloBasePairs) {
        $name = $pair[0]
        $found = $false
        foreach ($src in $pair[1]) {
            if (Test-Path $src) {
                $len = (Get-Item $src).Length
                if ($len -gt 1000000) {
                    Write-Host "  YOLO base: $src ($([math]::Round($len/1MB)) MB)"
                    Copy-Item $src (Join-Path $OutDir $name) -Force
                    Copy-Item $src (Join-Path $ws "models\yolo\$name") -Force
                    $found = $true
                    break
                }
            }
        }
        if (-not $found) {
            Write-Host "  WARNING: $name not found (train may try to download)"
        }
    }

    # Trained watermark detector (optional but recommended for batch auto-scan)
    $wmCandidates = @(
        (Join-Path $Root "workspace\models\yolo\watermark.pt"),
        (Join-Path $Root "workspace\yolo_train\runs\train\weights\best.pt")
    )
    $wmOk = $false
    foreach ($src in $wmCandidates) {
        if (Test-Path $src) {
            $len = (Get-Item $src).Length
            if ($len -gt 100000) {
                Write-Host "  YOLO watermark: $src ($([math]::Round($len/1MB)) MB)"
                Copy-Item $src (Join-Path $ws "models\yolo\watermark.pt") -Force
                $wmOk = $true
                break
            }
        }
    }
    if (-not $wmOk) {
        Write-Host "  NOTE: watermark.pt not found; user can train on Train page"
    }

    # BiRefNet (styles AI matting) — copy local HF snapshot when complete
    $birefSrc = Join-Path $Root "workspace\models\birefnet"
    $birefDst = Join-Path $ws "models\birefnet"
    $birefCfg = Join-Path $birefSrc "config.json"
    $birefW = @(
        (Join-Path $birefSrc "model.safetensors"),
        (Join-Path $birefSrc "pytorch_model.bin")
    )
    $hasW = $false
    foreach ($w in $birefW) { if (Test-Path $w) { $hasW = $true; break } }
    if ((Test-Path $birefCfg) -and $hasW) {
        Write-Host "  BiRefNet: $birefSrc"
        if (Test-Path $birefDst) { Remove-Item -Recurse -Force $birefDst }
        # Copy snapshot files; skip huge .cache if present
        New-Item -ItemType Directory -Force -Path $birefDst | Out-Null
        Get-ChildItem $birefSrc -Force | Where-Object {
            $_.Name -notin @(".cache", ".git")
        } | ForEach-Object {
            Copy-Item $_.FullName $birefDst -Recurse -Force
        }
    }
    else {
        Write-Host "  WARNING: BiRefNet weights incomplete under workspace\models\birefnet"
    }

    # README for yolo folder
    $yoloReadme = Join-Path $ws "models\yolo\README.txt"
    @(
        "YOLO weights folder",
        "  watermark.pt   - trained detector used by batch auto-scan",
        "  yolov8n.pt     - base for Detect training (also next to exe)",
        "  yolov8n-obb.pt - base for OBB (tilted boxes) training",
        "",
        "After training, best.pt is auto-copied to watermark.pt"
    ) | Set-Content -Path $yoloReadme -Encoding UTF8

    $birefReadme = Join-Path $ws "models\birefnet\README.txt"
    if (-not (Test-Path $birefReadme)) {
        @(
            "BiRefNet weights (styles page AI matting)",
            "Place HF snapshot here: config.json + model.safetensors",
            "Repo: ZhengPeng7/BiRefNet"
        ) | Set-Content -Path $birefReadme -Encoding UTF8
    }
}

# Versioned release zip
$RelRoot = Join-Path $DistRoot "releases"
New-Item -ItemType Directory -Force -Path $RelRoot | Out-Null
$VerName = "$AppName-$ver"
$VerDir = Join-Path $RelRoot $VerName

if (Test-Path $VerDir) {
    Remove-Item -Recurse -Force $VerDir
}
Write-Host "==> Copy release folder $VerName"
Copy-Item -Recurse $OutDir $VerDir

if (-not $SkipZip) {
    $ZipPath = Join-Path $RelRoot "$VerName.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Write-Host "==> Zip $ZipPath"
    Compress-Archive -Path $VerDir -DestinationPath $ZipPath -Force
}

Write-Host ""
Write-Host "Build OK"
Write-Host "  Latest:  $OutDir"
Write-Host "  Release: $VerDir"
if (-not $SkipZip) {
    Write-Host "  Zip:     $(Join-Path $RelRoot "$VerName.zip")"
}
Write-Host "  Version: $ver"
