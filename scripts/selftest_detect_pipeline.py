"""Self-test stages 0–3 detection pipeline (no GUI required).

Run: python scripts/selftest_detect_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remove_pic_watermark.detectors.residual_ai import ResidualAiDetector
from remove_pic_watermark.detectors.template_stamp import TemplateStampDetector
from remove_pic_watermark.detectors.yolo_watermark import (
    ensure_yolo_dir,
    probe_yolo,
    resolve_yolo_weights,
    ultralytics_available,
)
from remove_pic_watermark.image_io import read_image, write_image
from remove_pic_watermark.masking import combine_masks
from remove_pic_watermark.profiles.models import MatchStrategy
from remove_pic_watermark.services.job_service import JobService, JobSpec
from remove_pic_watermark.services.profile_service import ProfileService, RoiNorm
from remove_pic_watermark.workspace import get_workspace


def main() -> int:
    lines: list[str] = []
    out = ROOT / "workspace" / "jobs" / "_selftest_stage23"
    out.mkdir(parents=True, exist_ok=True)

    left = ROOT / "workspace" / "jobs" / "_diag_monica" / "left_watermarked.png"
    if not left.exists():
        print("SKIP: monica left image missing", left)
        return 2

    img = read_image(left)
    h, w = img.shape[:2]
    lines.append(f"image {w}x{h}")

    t0 = time.time()
    ai = ResidualAiDetector.from_config({"max_instances": 96})
    ai_hits = ai.detect(img)
    ai_mask = combine_masks(ai_hits, img.shape[:2])
    lines.append(
        f"residual_ai hits={len(ai_hits)} mask_px={int(np.count_nonzero(ai_mask))} "
        f"t={time.time() - t0:.2f}s"
    )
    write_image(out / "ai_mask.png", ai_mask)

    ws = get_workspace()
    svc = ProfileService(ws)
    roi = RoiNorm(left=20 / w, top=30 / h, right=130 / w, bottom=70 / h)
    profile, _build, directory = svc.create_from_roi(
        name="Monica_selftest",
        image_path=left,
        roi=roi,
        match_strategy=MatchStrategy.SEARCH,
        profile_id="Monica_selftest",
    )
    cfg = {
        "label": profile.id,
        "template_path": str(directory / profile.template_file),
        **profile.detector,
        "multi_instance": True,
    }
    t0 = time.time()
    style = TemplateStampDetector.from_config(cfg, directory / "profile.json")
    st_hits = style.detect(img)
    st_mask = combine_masks(st_hits, img.shape[:2])
    lines.append(
        f"style_multi hits={len(st_hits)} mask_px={int(np.count_nonzero(st_mask))} "
        f"t={time.time() - t0:.2f}s"
    )
    write_image(out / "style_mask.png", st_mask)

    union = cv2.bitwise_or(ai_mask, st_mask)
    lines.append(f"union mask_px={int(np.count_nonzero(union))}")
    write_image(out / "union_mask.png", union)

    job = JobService(ws)
    t0 = time.time()
    res_ai = job.run(
        JobSpec(
            input_path=left,
            profile_ids=[],
            backend="opencv",
            detect_mode="ai",
            job_id="selftest_ai",
            match_strategy="search",
        )
    )
    lines.append(
        f"job_ai detected={res_ai.summary.get('detected')} "
        f"actions={res_ai.summary.get('actions')} "
        f"mode={res_ai.summary.get('detect_mode')} "
        f"t={time.time() - t0:.2f}s"
    )

    t0 = time.time()
    res_both = job.run(
        JobSpec(
            input_path=left,
            profile_ids=["Monica_selftest"],
            backend="opencv",
            detect_mode="both",
            job_id="selftest_both",
            match_strategy="search",
        )
    )
    n_det = len(res_both.images[0].get("detections") or [])
    m = read_image(Path(res_both.images[0]["mask"]))
    mg = m[:, :, 0] if m.ndim == 3 else m
    lines.append(
        f"job_both detected={res_both.summary.get('detected')} "
        f"n_det={n_det} mask_px={int(np.count_nonzero(mg))} "
        f"t={time.time() - t0:.2f}s"
    )

    ensure_yolo_dir(ws.models_dir)
    yolo = probe_yolo(ws.models_dir)
    lines.append(
        f"ultralytics={ultralytics_available()} "
        f"yolo_weights={resolve_yolo_weights(ws.models_dir)} "
        f"yolo_status={yolo.status} ready={yolo.ready}"
    )

    # GUI smoke
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from remove_pic_watermark.gui.main_window import MainWindow
    from remove_pic_watermark.gui.theme_style import apply_app_typography

    app = QApplication.instance() or QApplication([])
    apply_app_typography(app)
    win = MainWindow(ws)
    win.batch_page.set_detect_mode("both")
    assert win.batch_page.current_detect_mode() == "both"
    win.close()
    lines.append("gui_smoke OK")

    # assertions
    assert len(ai_hits) >= 1, "residual AI should find something"
    assert len(st_hits) >= 1, "style multi should find something"
    assert res_ai.summary.get("detected", 0) >= 1
    assert res_both.summary.get("detected", 0) >= 1
    assert int(np.count_nonzero(union)) >= int(np.count_nonzero(st_mask))

    text = "\n".join(lines)
    (out / "SELFTEST_REPORT.txt").write_text(text + "\nALL PASSED\n", encoding="utf-8")
    print(text)
    print("--- ALL SELFTEST PASSED ---")
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
