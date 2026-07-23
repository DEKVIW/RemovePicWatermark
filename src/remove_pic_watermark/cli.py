from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from .backends.iopaint import build_command, executable_available
from .pipeline import run_detection_batch, run_opencv_preview
from .profiles.store import bootstrap_builtin_profiles
from .services.job_service import JobService, JobSpec
from .services.profile_service import ProfileService, RoiNorm
from .workspace import get_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch watermark mask generation and inpainting helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Generate masks and debug overlays.")
    add_common_detection_args(detect)

    run = subparsers.add_parser("run", help="Generate masks and optionally create an OpenCV preview.")
    add_common_detection_args(run)
    run.add_argument("--backend", choices=["none", "opencv"], default="opencv")
    run.add_argument("--output-dir", type=Path, default=Path("data/output/opencv"))
    run.add_argument("--opencv-radius", type=int, default=7)

    preview = subparsers.add_parser("preview", help="Apply OpenCV inpaint using existing masks.")
    preview.add_argument("--input", type=Path, required=True)
    preview.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    preview.add_argument("--output-dir", type=Path, default=Path("data/output/opencv"))
    preview.add_argument("--opencv-radius", type=int, default=7)

    iopaint = subparsers.add_parser("iopaint-command", help="Print or run the matching IOPaint command.")
    iopaint.add_argument("--image", type=Path, required=True)
    iopaint.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    iopaint.add_argument("--output-dir", type=Path, default=Path("data/output/iopaint"))
    iopaint.add_argument("--model", default="lama")
    iopaint.add_argument("--device", default="cpu")
    iopaint.add_argument("--model-dir", type=Path, default=None, help="Optional IOPaint model/cache directory.")
    iopaint.add_argument("--executable", default=None, help="Optional path to the iopaint executable.")
    iopaint.add_argument("--execute", action="store_true", help="Run IOPaint instead of only printing the command.")

    profile = subparsers.add_parser("profile", help="Manage watermark profiles.")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)

    profile_list = profile_sub.add_parser("list", help="List profiles.")
    profile_list.add_argument("--workspace", type=Path, default=None)

    profile_bootstrap = profile_sub.add_parser("bootstrap", help="Import built-in watermarks into workspace.")
    profile_bootstrap.add_argument("--workspace", type=Path, default=None)
    profile_bootstrap.add_argument("--overwrite", action="store_true")

    profile_create = profile_sub.add_parser("create", help="Create a profile from ROI or crop image.")
    profile_create.add_argument("--name", required=True)
    profile_create.add_argument("--image", type=Path, required=True, help="Sample image or crop.")
    profile_create.add_argument(
        "--roi",
        default=None,
        help="Normalized ROI left,top,right,bottom in 0-1 (e.g. 0.7,0.85,0.98,0.98).",
    )
    profile_create.add_argument("--crop", action="store_true", help="Treat --image as a cropped watermark.")
    profile_create.add_argument("--description", default="")
    profile_create.add_argument("--workspace", type=Path, default=None)

    profile_enable = profile_sub.add_parser("enable", help="Enable a profile.")
    profile_enable.add_argument("profile_id")
    profile_enable.add_argument("--workspace", type=Path, default=None)

    profile_disable = profile_sub.add_parser("disable", help="Disable a profile.")
    profile_disable.add_argument("profile_id")
    profile_disable.add_argument("--workspace", type=Path, default=None)

    job = subparsers.add_parser("job", help="Run a structured workspace job.")
    job.add_argument("--input", type=Path, required=True)
    job.add_argument("--profiles", default=None, help="Comma-separated profile ids (default: all enabled).")
    job.add_argument("--backend", choices=["none", "opencv", "iopaint"], default="opencv")
    job.add_argument("--opencv-radius", type=int, default=7)
    job.add_argument("--device", default="cpu")
    job.add_argument("--model", default="lama")
    job.add_argument("--model-dir", type=Path, default=None)
    job.add_argument("--workspace", type=Path, default=None)

    gui = subparsers.add_parser("gui", help="Launch the desktop GUI.")
    gui.add_argument("--workspace", type=Path, default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "detect":
            report = run_detection_batch(
                args.input,
                args.mask_dir,
                args.debug_dir,
                args.report,
                args.config,
                profile_ids=_split_ids(getattr(args, "profiles", None)),
                use_profiles=args.config is None,
            )
            print(f"Processed {len(report)} image(s).")
            return 0

        if args.command == "run":
            report = run_detection_batch(
                args.input,
                args.mask_dir,
                args.debug_dir,
                args.report,
                args.config,
                profile_ids=_split_ids(getattr(args, "profiles", None)),
                use_profiles=args.config is None,
            )
            print(f"Processed {len(report)} image(s).")
            if args.backend == "opencv":
                outputs = run_opencv_preview(args.input, args.mask_dir, args.output_dir, args.opencv_radius)
                print(format_output_summary(outputs, "OpenCV output"))
            return 0

        if args.command == "preview":
            outputs = run_opencv_preview(args.input, args.mask_dir, args.output_dir, args.opencv_radius)
            print(format_output_summary(outputs, "OpenCV output"))
            return 0

        if args.command == "iopaint-command":
            command = build_command(
                args.image,
                args.mask_dir,
                args.output_dir,
                args.model,
                args.device,
                args.model_dir,
                args.executable,
            )
            print(format_shell_command(command))
            if args.execute:
                if not executable_available(args.executable):
                    raise SystemExit("iopaint executable was not found. Install IOPaint first, then retry.")
                subprocess.run(command, check=True)
            return 0

        if args.command == "profile":
            return handle_profile(args)

        if args.command == "job":
            workspace = get_workspace(args.workspace)
            service = JobService(workspace)
            profile_ids = _split_ids(args.profiles)
            result = service.run(
                JobSpec(
                    input_path=args.input,
                    profile_ids=profile_ids,
                    backend=args.backend,
                    opencv_radius=args.opencv_radius,
                    iopaint_model=args.model,
                    iopaint_device=args.device,
                    iopaint_model_dir=args.model_dir or workspace.models_dir,
                )
            )
            print(f"Job {result.job_id} finished in {result.job_dir}")
            print(json_summary(result.summary))
            return 0

        if args.command == "gui":
            from .gui.app import main as gui_main

            gui_main(workspace=args.workspace)
            return 0

    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    return 1


def handle_profile(args: argparse.Namespace) -> int:
    workspace = get_workspace(getattr(args, "workspace", None))
    service = ProfileService(workspace)

    if args.profile_command == "list":
        service.ensure_builtins()
        profiles = service.list_profiles()
        if not profiles:
            print("No profiles found.")
            return 0
        for profile in profiles:
            flag = "on " if profile.enabled else "off"
            print(f"[{flag}] {profile.id:24}  {profile.kind.value:10}  {profile.name}")
        return 0

    if args.profile_command == "bootstrap":
        created = bootstrap_builtin_profiles(workspace, overwrite=args.overwrite)
        print(f"Bootstrapped {len(created)} profile(s): {', '.join(created) if created else '(none new)'}")
        return 0

    if args.profile_command == "create":
        if args.crop:
            profile, build, directory = service.create_from_crop_image(
                name=args.name,
                crop_path=args.image,
                description=args.description,
            )
        else:
            if not args.roi:
                raise ValueError("Provide --roi left,top,right,bottom or pass --crop for a cropped watermark image.")
            parts = [float(x.strip()) for x in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError("--roi must be left,top,right,bottom")
            roi = RoiNorm(*parts)
            profile, build, directory = service.create_from_roi(
                name=args.name,
                image_path=args.image,
                roi=roi,
                description=args.description,
            )
        print(f"Created profile {profile.id} at {directory}")
        print(f"Template method={build.stats.get('method')} fill_ratio={build.stats.get('fill_ratio')}")
        return 0

    if args.profile_command == "enable":
        service.set_enabled(args.profile_id, True)
        print(f"Enabled {args.profile_id}")
        return 0

    if args.profile_command == "disable":
        service.set_enabled(args.profile_id, False)
        print(f"Disabled {args.profile_id}")
        return 0

    return 1


def json_summary(summary: dict) -> str:
    import json

    return json.dumps(summary, ensure_ascii=False)


def format_shell_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def format_output_summary(outputs: list[dict[str, str]], label: str) -> str:
    inpainted = sum(1 for item in outputs if item.get("action") == "inpainted")
    copied = sum(1 for item in outputs if item.get("action") == "copied")
    return f"Created {len(outputs)} {label} image(s): {inpainted} inpainted, {copied} copied unchanged."


def add_common_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--config", type=Path, default=None, help="Legacy detector config JSON.")
    parser.add_argument("--profiles", default=None, help="Comma-separated profile ids (workspace profiles).")
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--debug-dir", type=Path, default=Path("data/debug"))
    parser.add_argument("--report", type=Path, default=Path("data/report.json"))


def _split_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


if __name__ == "__main__":
    raise SystemExit(main())
