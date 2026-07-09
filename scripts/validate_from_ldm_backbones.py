#!/usr/bin/env python3
"""Validate LDM backbones with 2K and 4K from-LDM inference.

The runner launches the repository entrypoint one backbone at a time:

    PYTHONPATH=. torchrun --nproc_per_node=N -m pid._src.inference.from_ldm ...

For diffusers and SigLIP backbones it uses prompt_creative.txt by default. DINOv2
is ImageNet class-conditional, so it uses a small default class-id list instead.
Diffusers `--resolution` is the final PiD output resolution, so this script passes
2048 for the baseline sweep. DINOv2 and SigLIP use their native 512/256 LDM
resolutions with scale 4/8, which also produce 2048px PiD outputs.

In addition, every registered 4K diffusers backbone is run again at
`--resolution 4096` by default to validate 4K PiD output quality.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pid._src.inference.checkpoint_registry import PID_CHECKPOINT_REGISTRY, get_pid_checkpoint
from pid._src.inference.cli_utils import LDM_BACKBONES

DEFAULT_PROMPT_FILE = REPO_ROOT / "pid/_src/inference/prompts/prompt_creative.txt"
DIFFUSERS_BACKBONES = [b for b in LDM_BACKBONES if b not in {"dinov2", "siglip"}]
FOUR_K_CKPT_TYPES = ("2kto4k", "2kto4k_v1pt5")

# Optional intermediate x_t capture points from README. By default the script
# saves only final x0 to keep validation output compact; pass --capture_recommended_xt
# to include these.
RECOMMENDED_XT_STEPS = {
    "flux": [24],
    "sd3": [24],
    "sdxl": [26],
    "flux2": [46],
    "flux2-klein-4b": [3],
    "flux2-klein-9b": [3],
    "qwenimage": [44],
    "qwenimage-2512": [44],
    "zimage": [46],
    "zimage-turbo": [7],
    "dinov2": [46],
    "siglip": [46],
}


@dataclass
class LDMValidationCase:
    name: str
    backbone: str
    ckpt_type: str
    experiment: str
    checkpoint_path: str
    pid_scale: int
    prompt_source: str
    validation_target: str
    resolution_arg: int
    pid_output_resolution: int


def _safe_name(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _select_ckpt_type(backbone: str) -> str:
    if (backbone, "2k") in PID_CHECKPOINT_REGISTRY:
        return "2k"
    for ckpt_type in FOUR_K_CKPT_TYPES:
        if (backbone, ckpt_type) in PID_CHECKPOINT_REGISTRY:
            return ckpt_type
    valid = ", ".join(sorted(f"{b}+{t}" for b, t in PID_CHECKPOINT_REGISTRY))
    raise KeyError(f"No checkpoint registered for backbone={backbone!r}. Valid registry keys: {valid}")


def _make_case(
    *,
    index: int,
    backbone: str,
    ckpt_type: str,
    validation_target: str,
    resolution_arg: int,
    pid_output_resolution: int,
) -> LDMValidationCase:
    ckpt = get_pid_checkpoint(backbone, ckpt_type)
    checkpoint_dir = Path(ckpt.checkpoint_path).parent.name or Path(ckpt.checkpoint_path).stem
    prompt_source = "imagenet_class_ids" if backbone == "dinov2" else "prompt_file"
    return LDMValidationCase(
        name=_safe_name(f"{index:02d}_{validation_target}_{backbone}_{ckpt_type}_{checkpoint_dir}"),
        backbone=backbone,
        ckpt_type=ckpt_type,
        experiment=ckpt.experiment,
        checkpoint_path=ckpt.checkpoint_path,
        pid_scale=ckpt.pid_scale,
        prompt_source=prompt_source,
        validation_target=validation_target,
        resolution_arg=resolution_arg,
        pid_output_resolution=pid_output_resolution,
    )


def _build_cases(
    backbones: list[str],
    *,
    resolution_2k: int,
    include_4k_2kto4k: bool,
    resolution_4k: int,
) -> list[LDMValidationCase]:
    cases: list[LDMValidationCase] = []
    index = 1

    for backbone in backbones:
        ckpt_type = _select_ckpt_type(backbone)
        if backbone == "dinov2":
            resolution_arg = 512
            pid_output_resolution = 512 * get_pid_checkpoint(backbone, ckpt_type).pid_scale
        elif backbone == "siglip":
            resolution_arg = 256
            pid_output_resolution = 256 * get_pid_checkpoint(backbone, ckpt_type).pid_scale
        else:
            resolution_arg = resolution_2k
            pid_output_resolution = resolution_2k
        cases.append(
            _make_case(
                index=index,
                backbone=backbone,
                ckpt_type=ckpt_type,
                validation_target="2k",
                resolution_arg=resolution_arg,
                pid_output_resolution=pid_output_resolution,
            )
        )
        index += 1

    if include_4k_2kto4k:
        for backbone in backbones:
            if backbone not in DIFFUSERS_BACKBONES:
                continue
            for ckpt_type in FOUR_K_CKPT_TYPES:
                if (backbone, ckpt_type) not in PID_CHECKPOINT_REGISTRY:
                    continue
                cases.append(
                    _make_case(
                        index=index,
                        backbone=backbone,
                        ckpt_type=ckpt_type,
                        validation_target=f"4k_{ckpt_type}",
                        resolution_arg=resolution_4k,
                        pid_output_resolution=resolution_4k,
                    )
                )
                index += 1
    return cases


def _resolve_nproc(value: str) -> int:
    if value != "auto":
        nproc = int(value)
        if nproc < 1:
            raise argparse.ArgumentTypeError("--nproc_per_node must be >= 1 or 'auto'")
        return nproc
    try:
        import torch

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _launcher(args: argparse.Namespace, nproc_per_node: int) -> list[str]:
    if nproc_per_node <= 1:
        return [args.python, "-m", "pid._src.inference.from_ldm"]
    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "pid._src.inference.from_ldm",
    ]


def _build_command(
    args: argparse.Namespace,
    case: LDMValidationCase,
    output_dir: Path,
    nproc_per_node: int,
) -> list[str]:
    command = [
        *_launcher(args, nproc_per_node),
        "--backbone",
        case.backbone,
        "--pid_ckpt_type",
        case.ckpt_type,
        "--experiment",
        case.experiment,
        "--checkpoint_path",
        case.checkpoint_path,
        "--output_dir",
        str(output_dir),
        "--cfg_scale",
        str(args.cfg_scale),
        "--pid_inference_steps",
        str(args.pid_inference_steps),
        "--scale",
        str(case.pid_scale),
        "--seed",
        str(args.seed),
        "--save_format",
        args.save_format,
        "--dtype",
        args.dtype,
    ]

    if args.load_ema_to_reg:
        command.append("--load_ema_to_reg")
    if args.compile:
        command.append("--compile")

    save_xt_steps = RECOMMENDED_XT_STEPS.get(case.backbone, []) if args.capture_recommended_xt else []
    if save_xt_steps:
        command.extend(["--save_xt_steps", *[str(step) for step in save_xt_steps]])

    if case.backbone == "dinov2":
        command.extend(
            [
                "--resolution",
                str(case.resolution_arg),
                "--num_inference_steps",
                str(args.rae_num_inference_steps),
                "--rae_class_ids",
                *[str(class_id) for class_id in args.rae_class_ids],
            ]
        )
    elif case.backbone == "siglip":
        command.extend(["--resolution", str(case.resolution_arg), "--prompt_file", str(args.prompt_file)])
    else:
        command.extend(["--resolution", str(case.resolution_arg), "--prompt_file", str(args.prompt_file)])
        if args.cpu_offload:
            command.append("--cpu_offload")

    if args.extra_from_ldm_args:
        command.extend(args.extra_from_ldm_args)

    return command


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive_output(output_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_root, arcname=output_root.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2K from-LDM validation, plus 4K validation for every registered 4K checkpoint."
    )
    parser.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--nproc_per_node", type=str, default="auto", help="'auto' or an integer GPU count.")
    parser.add_argument("--backbones", nargs="+", choices=LDM_BACKBONES, default=list(LDM_BACKBONES))
    parser.add_argument(
        "--resolution",
        type=int,
        default=2048,
        help="Final output resolution for the baseline 2K diffusers validation cases.",
    )
    parser.add_argument(
        "--resolution_4k",
        type=int,
        default=4096,
        help="Final output resolution for additional 4K checkpoint validation cases.",
    )
    parser.add_argument(
        "--skip_4k_2kto4k",
        action="store_true",
        help="Do not add the default 4K validation cases for registered 4K checkpoints.",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--pid_inference_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_format", choices=["png", "jpg"], default="jpg")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--load_ema_to_reg", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true", help="Forward --cpu_offload to diffusers backbones.")
    parser.add_argument(
        "--capture_recommended_xt",
        action="store_true",
        help="Also save README-recommended intermediate xt steps. Default saves only final x0.",
    )
    parser.add_argument(
        "--rae_class_ids",
        nargs="+",
        type=int,
        default=[207, 281, 387],
        help="ImageNet class IDs for the dinov2 class-conditional backbone.",
    )
    parser.add_argument("--rae_num_inference_steps", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true", help="Write planned commands without running inference.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed inference command.")
    parser.add_argument("--no_archive", action="store_true", help="Do not create a .tar.gz archive of the output root.")
    parser.add_argument(
        "--extra_from_ldm_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended to every from_ldm command. Put this flag last.",
    )
    args = parser.parse_args()

    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = REPO_ROOT / "results" / f"from_ldm_backbone_validation_{stamp}"

    args.prompt_file = args.prompt_file.resolve()
    args.output_root = args.output_root.resolve()
    args.nproc_per_node_resolved = _resolve_nproc(args.nproc_per_node)
    return args


def main() -> int:
    args = parse_args()
    cases = _build_cases(
        args.backbones,
        resolution_2k=args.resolution,
        include_4k_2kto4k=not args.skip_4k_2kto4k,
        resolution_4k=args.resolution_4k,
    )

    logs_dir = args.output_root / "logs"
    cases_dir = args.output_root / "cases"
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "repo_root": str(REPO_ROOT),
        "prompt_file": str(args.prompt_file),
        "output_root": str(args.output_root),
        "nproc_per_node": args.nproc_per_node,
        "nproc_per_node_resolved": args.nproc_per_node_resolved,
        "resolution": args.resolution,
        "resolution_4k": args.resolution_4k,
        "include_4k_2kto4k": not args.skip_4k_2kto4k,
        "cfg_scale": args.cfg_scale,
        "pid_inference_steps": args.pid_inference_steps,
        "seed": args.seed,
        "save_format": args.save_format,
        "dtype": args.dtype,
        "dry_run": args.dry_run,
        "notes": [
            f"Diffusers baseline cases use --resolution {args.resolution} as final PiD output resolution.",
            f"Every registered diffusers 4K checkpoint is also run with --resolution {args.resolution_4k} by default.",
            "dinov2 is ImageNet class-conditional and uses --rae_class_ids instead of prompt_file.",
            "siglip runs at native --resolution 256 with --scale 8, producing 2048px PiD output.",
        ],
        "cases": [],
        "archive_path": None,
    }

    command_lines: list[str] = []
    failures = 0
    missing = 0

    for index, case in enumerate(cases, start=1):
        case_output_dir = cases_dir / case.name
        log_path = logs_dir / f"{case.name}.log"
        checkpoint_abs = (REPO_ROOT / case.checkpoint_path).resolve()
        command = _build_command(args, case, case_output_dir, args.nproc_per_node_resolved)
        command_line = _format_command(command)
        command_lines.append(command_line)

        case_record = {
            **asdict(case),
            "index": index,
            "output_dir": str(case_output_dir),
            "log_path": str(log_path),
            "command": command_line,
            "checkpoint_exists": checkpoint_abs.is_file(),
            "status": "pending",
            "returncode": None,
            "elapsed_sec": None,
        }

        if not checkpoint_abs.is_file():
            missing += 1
            case_record["status"] = "missing_checkpoint"
            _write_text(log_path, f"Missing checkpoint: {checkpoint_abs}\n\nPlanned command:\n{command_line}\n")
            summary["cases"].append(case_record)
            continue

        if args.dry_run:
            case_record["status"] = "dry_run"
            _write_text(log_path, f"Dry run. Planned command:\n{command_line}\n")
            summary["cases"].append(case_record)
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}:{existing_pythonpath}"

        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Command:\n{command_line}\n\n")
            log_file.flush()
            proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - start

        case_record["returncode"] = proc.returncode
        case_record["elapsed_sec"] = round(elapsed, 3)
        if proc.returncode == 0:
            case_record["status"] = "passed"
        else:
            failures += 1
            case_record["status"] = "failed"
        summary["cases"].append(case_record)

        _write_json(args.output_root / "summary.json", summary)
        if proc.returncode != 0 and args.fail_fast:
            break

    _write_text(args.output_root / "commands.txt", "\n".join(command_lines) + "\n")
    readme = [
        "PiD from-LDM backbone validation",
        "",
        f"Prompt file: {args.prompt_file}",
        f"Resolved nproc_per_node: {args.nproc_per_node_resolved}",
        f"Baseline diffusers resolution: {args.resolution}",
        f"4K diffusers resolution: {args.resolution_4k}",
        f"Cases directory: {cases_dir}",
        f"Logs directory: {logs_dir}",
        "",
        "Each case output is written under cases/<case-name>/ using the from_ldm.py layout:",
        "  vae_decode/ or dinov2_decode/ or siglip_decode/",
        "  <run-tag>/step_x0/",
        "",
        "dinov2 is class-conditional and uses --rae_class_ids; all other backbones use the prompt file.",
        "Cases marked validation_target=4k_<ckpt_type> run registered 4K checkpoints at 4K PiD output.",
        "Commands are recorded in commands.txt.",
        "",
    ]
    _write_text(args.output_root / "README.txt", "\n".join(readme))
    _write_json(args.output_root / "summary.json", summary)

    if not args.no_archive:
        archive_path = args.output_root.with_suffix(".tar.gz")
        summary["archive_path"] = str(archive_path)
        _write_json(args.output_root / "summary.json", summary)
        _archive_output(args.output_root, archive_path)

    passed = sum(1 for item in summary["cases"] if item["status"] == "passed")
    dry_runs = sum(1 for item in summary["cases"] if item["status"] == "dry_run")
    print(f"Output root: {args.output_root}")
    if summary["archive_path"]:
        print(f"Archive: {summary['archive_path']}")
    print(
        f"Cases: {len(summary['cases'])}, passed: {passed}, dry_run: {dry_runs}, failed: {failures}, missing: {missing}"
    )

    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
