# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir for RemovePicWatermark GUI.
# Build: scripts/build_gui_onedir.ps1

import sys
from pathlib import Path

block_cipher = None

SPECDIR = Path(SPEC).resolve().parent if "SPEC" in dir() else Path(".").resolve()
ROOT = SPECDIR.parent if SPECDIR.name == "packaging" else SPECDIR
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

datas = [
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "packaging" / "user_README.txt"), "."),
    (str(ROOT / "packaging" / "app.ico"), "."),
    (str(ROOT / "packaging" / "app.png"), "."),
    (str(ROOT / "packaging" / "patch_torch_ufuncs.py"), "."),
    (
        str(ROOT / "src" / "remove_pic_watermark" / "gui" / "resources"),
        "remove_pic_watermark/gui/resources",
    ),
]

_icon = str(ROOT / "packaging" / "app.ico")
if not Path(_icon).is_file():
    _icon = None
binaries = []
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "qfluentwidgets",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "requests",
    "psutil",
    "matplotlib",
    "scipy",
    "remove_pic_watermark",
    "remove_pic_watermark.stdio_fix",
    "remove_pic_watermark.gui",
    "remove_pic_watermark.gui.app",
    "remove_pic_watermark.gui.main_window",
    "remove_pic_watermark.gui.workers",
    "remove_pic_watermark.gui.pages.batch_page",
    "remove_pic_watermark.gui.pages.profiles_page",
    "remove_pic_watermark.gui.pages.refine_page",
    "remove_pic_watermark.gui.pages.train_page",
    "remove_pic_watermark.gui.pages.results_page",
    "remove_pic_watermark.backends.iopaint",
    "remove_pic_watermark.backends.opencv",
    "remove_pic_watermark.services.job_service",
    "remove_pic_watermark.services.profile_service",
    "remove_pic_watermark.services.yolo_dataset",
    "remove_pic_watermark.detectors.yolo_watermark",
    "remove_pic_watermark.detectors.residual_ai",
    "remove_pic_watermark.detectors.orchestrator",
    "torch",
    "torch.jit",
    "torchvision",
    "ultralytics",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.utils",
    "ultralytics.engine",
    "ultralytics.engine.trainer",
    "ultralytics.engine.model",
    "ultralytics.cfg",
    "ultralytics.data",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_agg",
    "six",
    "six.moves",
    "dateutil",
    "python_dateutil",
]

# Must feed collect_all into Analysis(), never mutate a.datas after Analysis.
# Do NOT collect_all("iopaint"): it pulls diffusers and breaks under runw.exe.
# LaMa runs via pure torch.jit on big-lama.pt (see backends/iopaint.py).
# ultralytics needs matplotlib (plots/utils); do NOT exclude it.
try:
    from PyInstaller.utils.hooks import collect_all

    for pkg in (
        "torch",
        "torchvision",
        "cv2",
        "qfluentwidgets",
        "ultralytics",
        "matplotlib",
        # BiRefNet matting (styles AI)
        "transformers",
        "kornia",
        "timm",
        "einops",
        "huggingface_hub",
    ):
        try:
            pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
            datas += pkg_datas
            binaries += pkg_binaries
            hiddenimports += pkg_hidden
            print("collect_all ok", pkg, "datas", len(pkg_datas), "bins", len(pkg_binaries))
        except Exception as exc:
            print("collect_all skip", pkg, exc)
except Exception as exc:
    print("collect_all unavailable", exc)

a = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(SRC), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyi_rth_app_root.py")],
    excludes=[
        "tkinter",
        "notebook",
        "pytest",
        "IPython",
        "tensorboard",
        # Heavy / fragile under freeze; not required for big-lama.pt JIT path
        "iopaint",
        "diffusers",
        "accelerate",
        "peft",
        # transformers / timm collected above for BiRefNet matting
        "skimage",
        "sklearn",
        "pandas",
        "altair",
        "fastapi",
        "uvicorn",
        "gradio",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RemovePicWatermark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RemovePicWatermark",
)
