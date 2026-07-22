"""Background workers that call services (never shell out to CLI strings)."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ..services.job_service import JobResult, JobService, JobSpec
from ..workspace import Workspace


@dataclass
class RunJobRequest:
    input_path: Path
    profile_ids: list[str]
    backend: str = "opencv"
    opencv_radius: int = 7
    iopaint_device: str = "cpu"
    iopaint_model: str = "lama"
    iopaint_model_dir: Path | None = None
    match_strategy: str | None = None
    detect_mode: str = "styles"
    enable_yolo: bool = True
    yolo_weights: Path | None = None


class JobWorker(QObject):
    log_line = Signal(str)
    stage = Signal(str)
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, workspace: Workspace, request: RunJobRequest) -> None:
        super().__init__()
        self.workspace = workspace
        self.request = request
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            try:
                from ..stdio_fix import ensure_stdio

                ensure_stdio()
            except Exception:
                pass
            self.stage.emit("开始处理")
            backend_name = {"iopaint": "高质量", "opencv": "快速"}.get(
                self.request.backend, self.request.backend
            )
            n_styles = len(self.request.profile_ids)
            self.log_line.emit(
                f"处理启动 · 修补={backend_name} · 样式={n_styles} 个"
            )
            service = JobService(self.workspace)

            def on_progress(current: int, total: int, message: str) -> None:
                self.progress.emit(current, total)
                self.stage.emit(message)
                self.log_line.emit(message)

            result: JobResult = service.run(
                JobSpec(
                    input_path=self.request.input_path,
                    profile_ids=list(self.request.profile_ids),
                    backend=self.request.backend,
                    opencv_radius=self.request.opencv_radius,
                    iopaint_device=self.request.iopaint_device,
                    iopaint_model=self.request.iopaint_model,
                    iopaint_model_dir=self.request.iopaint_model_dir or self.workspace.models_dir,
                    match_strategy=self.request.match_strategy,
                    detect_mode=self.request.detect_mode,
                    enable_yolo=bool(self.request.enable_yolo),
                    yolo_weights=self.request.yolo_weights,
                ),
                progress=on_progress,
                should_cancel=lambda: self._cancel,
            )
            if result.summary.get("cancelled"):
                self.log_line.emit("已按请求停止")
            else:
                n_hit = (result.summary or {}).get("detected")
                n_img = (result.summary or {}).get("image_count")
                if n_img is not None and n_hit is not None:
                    self.log_line.emit(f"处理完成 · {n_hit}/{n_img} 张有识别结果")
                else:
                    self.log_line.emit("处理完成")
            self.finished_ok.emit(result)
        except Exception as error:  # noqa: BLE001
            detail = traceback.format_exc()
            self.log_line.emit(detail)
            self.failed.emit(f"{error}\n\n{detail[-800:]}")


def start_job_worker(owner: QObject, worker: JobWorker) -> QThread:
    """Start job worker; returns the QThread (keep a reference until finished)."""
    thread = QThread(owner)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    # quit thread when job ends
    worker.finished_ok.connect(thread.quit)
    worker.failed.connect(thread.quit)
    # delete worker after thread stops; thread is owned by owner
    thread.finished.connect(worker.deleteLater)

    thread.start()
    return thread


@dataclass
class YoloTrainRequest:
    data_yaml: Path
    project_dir: Path  # runs root
    name: str = "train"
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 4
    device: str = "cpu"  # cpu | 0
    task: str = "detect"  # detect | obb


class YoloTrainWorker(QObject):
    """Run ultralytics YOLO train in background; deploy best.pt on success."""

    log_line = Signal(str)
    stage = Signal(str)
    # epoch progress for status bar: (current_epoch 1-based, total_epochs, percent 0-100)
    progress = Signal(int, int, int)
    finished_ok = Signal(str)  # path to deployed watermark.pt
    failed = Signal(str)

    def __init__(self, workspace: Workspace, request: YoloTrainRequest) -> None:
        super().__init__()
        self.workspace = workspace
        self.request = request
        self._cancel = False
        self._last_progress_pct = -1
        self._last_progress_emit_at = 0.0

    def request_cancel(self) -> None:
        self._cancel = True

    def _emit_train_progress(
        self,
        epoch_1based: int,
        total_epochs: int,
        *,
        batch_i: int | None = None,
        batch_n: int | None = None,
        force: bool = False,
    ) -> None:
        """Update stage + progress; throttle batch-level spam."""
        import time

        total = max(1, int(total_epochs))
        ep = max(0, min(int(epoch_1based), total))
        if batch_i is not None and batch_n is not None and batch_n > 0 and ep > 0:
            # In-epoch fraction so the bar moves during long epochs
            frac = (max(0, ep - 1) + min(1.0, (int(batch_i) + 1) / float(batch_n))) / float(
                total
            )
            pct = int(min(99, max(0, frac * 100)))
            show_ep = max(1, ep)
        else:
            pct = int(min(100, max(0, ep * 100 // total))) if ep > 0 else 0
            if ep >= total and ep > 0:
                pct = min(99, pct)  # reserve 100% for deploy/finish
            show_ep = max(0, ep)
        msg = f"训练中 {show_ep}/{total}（{pct}%）"

        now = time.monotonic()
        # Throttle: only every 0.4s or when percent changes (or force)
        if (
            not force
            and pct == self._last_progress_pct
            and (now - self._last_progress_emit_at) < 0.4
        ):
            return
        if (
            not force
            and pct == self._last_progress_pct
            and batch_i is not None
            and (now - self._last_progress_emit_at) < 0.8
        ):
            return
        self._last_progress_pct = pct
        self._last_progress_emit_at = now
        self.progress.emit(ep, total, pct)
        self.stage.emit(msg)

    def _attach_train_callbacks(self, model: Any, epochs: int) -> None:
        """Hook ultralytics train loop so the UI can show percent progress."""

        def _stop_if_cancelled(trainer: Any) -> None:
            if not self._cancel:
                return
            for attr in ("stop", "stop_training"):
                try:
                    setattr(trainer, attr, True)
                except Exception:
                    pass

        def on_train_epoch_end(trainer: Any) -> None:
            try:
                _stop_if_cancelled(trainer)
                ep = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", epochs) or epochs)
                self._emit_train_progress(ep, total, force=True)
            except Exception:
                pass

        def on_train_batch_end(trainer: Any) -> None:
            try:
                _stop_if_cancelled(trainer)
                ep = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", epochs) or epochs)
                # Ultralytics: nb = batches/epoch, ni = global iter (1-based often)
                batch_n = getattr(trainer, "nb", None)
                if not batch_n:
                    loader = getattr(trainer, "train_loader", None)
                    if loader is not None:
                        try:
                            batch_n = len(loader)
                        except Exception:
                            batch_n = None
                batch_i = None
                if batch_n:
                    batch_n = max(1, int(batch_n))
                    ni = getattr(trainer, "ni", None)
                    if ni is not None:
                        # ni is cumulative; map to index within current epoch
                        batch_i = (int(ni) - 1) % batch_n
                    else:
                        batch_i = getattr(trainer, "batch_i", None)
                        if batch_i is None:
                            batch_i = getattr(trainer, "i", None)
                if batch_i is not None and batch_n:
                    self._emit_train_progress(
                        ep, total, batch_i=int(batch_i), batch_n=int(batch_n)
                    )
            except Exception:
                pass

        def on_train_start(trainer: Any) -> None:
            try:
                total = int(getattr(trainer, "epochs", epochs) or epochs)
                self._emit_train_progress(0, total, force=True)
                self.stage.emit(f"训练中 0/{total}（0%）")
            except Exception:
                self.stage.emit("训练中…")

        try:
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_batch_end", on_train_batch_end)
        except Exception:
            # Older ultralytics: still try epoch-only
            try:
                model.add_callback("on_fit_epoch_end", on_train_epoch_end)
            except Exception:
                pass

    def _probe_train_quality(self, weights: Path, data_yaml: Path) -> str:
        """Predict on train images; warn if max conf is too low for batch defaults."""
        try:
            import cv2
            from ultralytics import YOLO

            text = Path(data_yaml).read_text(encoding="utf-8")
            train_line = ""
            for line in text.splitlines():
                if line.strip().startswith("train:"):
                    train_line = line.split(":", 1)[1].strip()
                    break
            if not train_line:
                return ""
            train_path = Path(train_line)
            image_paths: list[Path] = []
            if train_path.is_file() and train_path.suffix.lower() == ".txt":
                for row in train_path.read_text(encoding="utf-8").splitlines():
                    p = Path(row.strip())
                    if p.is_file():
                        image_paths.append(p)
            elif train_path.is_dir():
                image_paths = sorted(train_path.glob("*.*"))[:8]
            if not image_paths:
                return ""

            model = YOLO(str(weights))
            max_conf = 0.0
            n_at_15 = 0
            n_at_05 = 0
            for img_path in image_paths[:6]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                r = model.predict(
                    img,
                    conf=0.01,
                    iou=0.45,
                    device="cpu",
                    verbose=False,
                    imgsz=640,
                    max_det=100,
                )[0]
                confs: list[float] = []
                # OBB weights expose r.obb; detect uses r.boxes
                obb = getattr(r, "obb", None)
                if obb is not None and len(obb) > 0 and getattr(obb, "conf", None) is not None:
                    confs = [float(x) for x in obb.conf.cpu().numpy()]
                elif r.boxes is not None and len(r.boxes) > 0:
                    confs = [float(x) for x in r.boxes.conf.cpu().numpy()]
                if confs:
                    max_conf = max(max_conf, max(confs))
                    n_at_15 += sum(1 for c in confs if c >= 0.15)
                    n_at_05 += sum(1 for c in confs if c >= 0.05)

            if max_conf < 0.08:
                return (
                    "效果偏弱：建议在同一张图上框全各处水印，"
                    "再补充不同背景样例后重训。"
                )
            if max_conf < 0.20:
                return "效果一般：批量可用，建议继续补充标注。"
            return "训练完成，可用于批量「样式+模型」或「水印模型」。"
        except Exception:  # noqa: BLE001
            return ""

    def run(self) -> None:
        try:
            try:
                from ..stdio_fix import ensure_stdio

                ensure_stdio()
            except Exception:
                pass

            req = self.request
            self.stage.emit("准备训练…")
            from ..detectors.yolo_watermark import import_yolo_class

            YOLO = import_yolo_class()

            if not Path(req.data_yaml).is_file():
                raise FileNotFoundError(f"找不到训练数据配置: {req.data_yaml}")

            task = (req.task or "detect").lower().strip()
            if task not in {"detect", "obb"}:
                task = "detect"
            self.log_line.emit(f"轮数 {req.epochs} · 设备 {req.device}")
            self.stage.emit("加载模型…")
            # OBB must use an OBB checkpoint; detect weights reject polygon labels.
            model = YOLO(req.model)
            model_task = str(getattr(model, "task", "") or "").lower()
            if task == "obb" and model_task == "detect":
                raise RuntimeError(
                    "斜框训练需要对应的基础模型，当前模型不匹配。"
                )

            # Invalidate stale ultralytics label cache next to data.yaml
            try:
                data_dir = Path(req.data_yaml).resolve().parent
                for name in ("labels.cache", "train.cache", "val.cache"):
                    for cand in (data_dir / name, data_dir / "labels" / name):
                        if cand.is_file():
                            cand.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

            total_epochs = max(1, int(req.epochs))
            self._emit_train_progress(0, total_epochs, force=True)
            if task == "obb":
                self.log_line.emit("斜框训练：请尽量框全图中每一处水印")
            else:
                self.log_line.emit("矩形框训练：请尽量框全图中每一处水印")
            self._attach_train_callbacks(model, total_epochs)
            # Few-shot friendly knobs. OBB: skip copy_paste (detect-oriented).
            train_kw: dict = {
                "data": str(req.data_yaml),
                "epochs": total_epochs,
                "imgsz": int(req.imgsz),
                "batch": int(req.batch),
                "device": req.device,
                "project": str(req.project_dir),
                "name": str(req.name),
                "exist_ok": True,
                "verbose": True,
                "workers": 0,  # Windows-friendly
                "single_cls": True,
                "cos_lr": True,
                "close_mosaic": max(10, total_epochs // 5),
                "patience": max(20, total_epochs // 2),
                "degrees": 20.0 if task == "detect" else 15.0,
                "translate": 0.12,
                "scale": 0.55,
                "shear": 2.0 if task == "detect" else 0.0,
                "perspective": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "mosaic": 1.0,
                "mixup": 0.15 if task == "detect" else 0.05,
                "hsv_h": 0.02,
                "hsv_s": 0.55,
                "hsv_v": 0.35,
            }
            if task == "obb":
                # Prefer model.train(task=...) only when supported; OBB model already sets it
                train_kw["task"] = "obb"
            else:
                train_kw["copy_paste"] = 0.45
            results = model.train(**train_kw)
            if self._cancel:
                self.failed.emit("已取消")
                return

            # Locate best.pt
            best = None
            save_dir = getattr(results, "save_dir", None)
            if save_dir is not None:
                cand = Path(save_dir) / "weights" / "best.pt"
                if cand.is_file():
                    best = cand
            if best is None:
                # fallback search
                runs = Path(req.project_dir) / req.name / "weights" / "best.pt"
                if runs.is_file():
                    best = runs
            if best is None or not best.is_file():
                raise FileNotFoundError("训练结束但未找到 best.pt")

            # Self-check: few-shot models often peak at conf≈0.02 and look "trained"
            # in the UI while batch (conf≥0.15) emits zero masks. Probe train images.
            self.stage.emit("检查效果…")
            quality_note = self._probe_train_quality(best, req.data_yaml)
            if quality_note:
                self.log_line.emit(quality_note)

            self.stage.emit("保存模型…")
            from ..services.yolo_dataset import YoloDatasetService

            dest = YoloDatasetService(self.workspace).deploy_weights(best)
            self.log_line.emit(f"已部署: {dest}")
            if quality_note:
                self.log_line.emit(quality_note)
            self.stage.emit("训练完成")
            # Path + optional quality warning (UI parses after first line)
            payload = str(dest)
            if quality_note:
                payload = f"{dest}\n{quality_note}"
            self.finished_ok.emit(payload)
        except Exception as error:  # noqa: BLE001
            detail = traceback.format_exc()
            self.log_line.emit(detail)
            self.failed.emit(f"{error}\n\n{detail[-600:]}")


def start_yolo_train_worker(owner: QObject, worker: YoloTrainWorker) -> QThread:
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished_ok.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    return thread


@dataclass
class RefineRequest:
    """Single-image inpaint request.

    Prefer ``image_bgr`` + ``mask`` (in-memory) so the UI thread does not block on
    PNG encode / temp-file I/O before the worker starts. Path fields remain for
    tests or callers that already have files on disk.
    """

    backend: str = "iopaint"  # iopaint | opencv
    device: str = "cpu"
    model_dir: Path | None = None
    opencv_radius: int = 7
    output_path: Path | None = None
    image_path: Path | None = None
    mask_path: Path | None = None
    # HxWx3 BGR uint8 / HxW uint8 — owned by worker after start (UI must not mutate)
    image_bgr: object | None = None
    mask: object | None = None


class RefineWorker(QObject):
    """Single-image paint → inpaint (does not freeze UI)."""

    stage = Signal(str)
    finished_ok = Signal(str)  # output path
    failed = Signal(str)

    def __init__(self, workspace: Workspace, request: RefineRequest) -> None:
        super().__init__()
        self.workspace = workspace
        self.request = request

    def run(self) -> None:
        try:
            # Worker thread in frozen GUI may also see None stdout/stderr
            try:
                from ..stdio_fix import ensure_stdio

                ensure_stdio()
            except Exception:
                pass

            import numpy as np

            from ..backends.iopaint import (
                ensure_lama_checkpoint,
                get_lama_engine,
                project_root_from_path,
                resolve_model_dir,
            )
            from ..backends.opencv import inpaint as cv_inpaint
            from ..image_io import read_image, write_image

            req = self.request
            self.stage.emit("准备输入…")
            if req.image_bgr is not None:
                image = np.asarray(req.image_bgr)
            elif req.image_path is not None:
                self.stage.emit("读取图片…")
                image = read_image(req.image_path)
            else:
                raise ValueError("RefineRequest 缺少 image_bgr / image_path")

            if req.mask is not None:
                mask = np.asarray(req.mask)
            elif req.mask_path is not None:
                mask = read_image(req.mask_path)
            else:
                raise ValueError("RefineRequest 缺少 mask / mask_path")

            if mask.ndim == 3:
                mask = mask[:, :, 0]
            # binary white = inpaint
            mask_bin = (mask > 127).astype("uint8") * 255

            if req.backend == "iopaint":
                self.stage.emit("高质量修补中…")
                root = project_root_from_path(self.workspace.root)
                model_dir = resolve_model_dir(
                    req.model_dir or self.workspace.models_dir, root
                )
                ckpt = ensure_lama_checkpoint(
                    model_dir,
                    [self.workspace.models_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"],
                )
                if ckpt is None:
                    raise FileNotFoundError("未找到修补模型，请检查安装是否完整")
                engine = get_lama_engine(ckpt, device=req.device)
                result = engine.inpaint_bgr(image, mask_bin)
            else:
                self.stage.emit("快速修补中…")
                result = cv_inpaint(image, mask_bin, radius=req.opencv_radius)

            out = req.output_path
            if out is None:
                # Ephemeral temp only — caller loads into memory then deletes.
                # Avoid filling workspace/jobs/_refine on every refine pass.
                import os
                import tempfile

                fd, name = tempfile.mkstemp(suffix="_refine_out.png")
                os.close(fd)
                out = Path(name)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)

            self.stage.emit("写入结果…")
            write_image(out, result)
            self.finished_ok.emit(str(out))
        except Exception as error:  # noqa: BLE001
            detail = traceback.format_exc()
            self.failed.emit(f"{error}\n\n{detail[-600:]}")


def start_refine_worker(owner: QObject, worker: RefineWorker) -> QThread:
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # Quit after result; keep worker alive until thread fully stops so slots can run
    worker.finished_ok.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    return thread
