#!/usr/bin/env python3
"""Validate every unique registry checkpoint with from-clean inference.

The runner follows the README's from-clean recipe:

    PYTHONPATH=. python -m pid._src.inference.from_clean --backbone ...

By default it uses assets/clean_image_manifest.jsonl, whose images are 512x512.
For sr4x checkpoints this produces 2048x2048 outputs. The SigLIP checkpoint is
sr8x and its encoder uses a fixed 256px native interface, also producing 2048px
outputs.

For 4K checkpoints, the runner also validates assets/clean_image_manifest_1024.jsonl
by default.
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

from pid._src.inference.checkpoint_registry import PID_CHECKPOINT_REGISTRY
from pid._src.inference.cli_utils import CLEAN_BACKBONES

FOUR_K_CKPT_TYPES = ("2kto4k", "2kto4k_v1pt5")


@dataclass
class ValidationCase:
    name: str
    backbone: str
    ckpt_type: str
    experiment: str
    checkpoint_path: str
    pid_scale: int
    covered_registry_keys: list[str]
    covered_alias_keys: list[str]


def _registry_key_name(key: tuple[str, str]) -> str:
    backbone, ckpt_type = key
    return f"{backbone}+{ckpt_type}"


def _safe_name(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _build_cases() -> tuple[list[ValidationCase], list[dict]]:
    """Return one runnable case per unique checkpoint, plus unsupported groups."""
    clean_backbones = set(CLEAN_BACKBONES)
    clean_order = {name: idx for idx, name in enumerate(CLEAN_BACKBONES)}

    groups: dict[tuple[str, str, int], dict] = {}
    for key, ckpt in sorted(PID_CHECKPOINT_REGISTRY.items()):
        fingerprint = (ckpt.experiment, ckpt.checkpoint_path, ckpt.pid_scale)
        groups.setdefault(fingerprint, {"ckpt": ckpt, "keys": []})["keys"].append(key)

    cases: list[ValidationCase] = []
    unsupported_groups: list[dict] = []

    for idx, (fingerprint, group) in enumerate(sorted(groups.items(), key=lambda item: item[0][1]), start=1):
        ckpt = group["ckpt"]
        keys = sorted(group["keys"], key=lambda key: (key[1], clean_order.get(key[0], 999), key[0]))
        clean_keys = [key for key in keys if key[0] in clean_backbones]
        alias_keys = [key for key in keys if key not in clean_keys]

        if not clean_keys:
            unsupported_groups.append(
                {
                    "experiment": ckpt.experiment,
                    "checkpoint_path": ckpt.checkpoint_path,
                    "pid_scale": ckpt.pid_scale,
                    "registry_keys": [_registry_key_name(key) for key in keys],
                    "reason": "No registry key for this checkpoint is accepted by from_clean.",
                }
            )
            continue

        backbone, ckpt_type = clean_keys[0]
        checkpoint_dir = Path(ckpt.checkpoint_path).parent.name or Path(ckpt.checkpoint_path).stem
        name = _safe_name(f"{idx:02d}_{backbone}_{ckpt_type}_{checkpoint_dir}")
        cases.append(
            ValidationCase(
                name=name,
                backbone=backbone,
                ckpt_type=ckpt_type,
                experiment=ckpt.experiment,
                checkpoint_path=ckpt.checkpoint_path,
                pid_scale=ckpt.pid_scale,
                covered_registry_keys=[_registry_key_name(key) for key in keys],
                covered_alias_keys=[_registry_key_name(key) for key in alias_keys],
            )
        )

    return cases, unsupported_groups


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_command(args: argparse.Namespace, case: ValidationCase, manifest: Path, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "pid._src.inference.from_clean",
        "--backbone",
        case.backbone,
        "--pid_ckpt_type",
        case.ckpt_type,
        "--experiment",
        case.experiment,
        "--checkpoint_path",
        case.checkpoint_path,
        "--manifest",
        str(manifest),
        "--degrade_sigmas",
        *[str(sigma) for sigma in args.degrade_sigmas],
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
    ]
    if args.load_ema_to_reg:
        command.append("--load_ema_to_reg")
    if args.compile:
        command.append("--compile")
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
        description="Run from-clean validation for every unique checkpoint in checkpoint_registry.py."
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "assets/clean_image_manifest.jsonl")
    parser.add_argument("--manifest_1024", type=Path, default=REPO_ROOT / "assets/clean_image_manifest_1024.jsonl")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--degrade_sigmas", type=float, nargs="+", default=[0.0, 0.3])
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--pid_inference_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--save_format", choices=["png", "jpg"], default="jpg")
    parser.add_argument("--load_ema_to_reg", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write the planned commands without running inference.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed inference command.")
    parser.add_argument("--no_archive", action="store_true", help="Do not create a .tar.gz archive of the output root.")
    args = parser.parse_args()

    if args.output_root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = REPO_ROOT / "results" / f"from_clean_checkpoint_validation_{stamp}"

    args.manifest = args.manifest.resolve()
    args.manifest_1024 = args.manifest_1024.resolve()
    args.output_root = args.output_root.resolve()
    return args


def _iter_case_runs(args: argparse.Namespace, case: ValidationCase) -> list[tuple[str, Path, str]]:
    runs = [("base", args.manifest, "")]
    if case.ckpt_type in FOUR_K_CKPT_TYPES:
        runs.append(("1024", args.manifest_1024, "__1024"))
    return runs


def main() -> int:
    args = parse_args()
    cases, unsupported_groups = _build_cases()

    logs_dir = args.output_root / "logs"
    cases_dir = args.output_root / "cases"
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "repo_root": str(REPO_ROOT),
        "manifest": str(args.manifest),
        "manifest_1024": str(args.manifest_1024),
        "extra_1024_manifest_ckpt_types": list(FOUR_K_CKPT_TYPES),
        "output_root": str(args.output_root),
        "degrade_sigmas": args.degrade_sigmas,
        "cfg_scale": args.cfg_scale,
        "pid_inference_steps": args.pid_inference_steps,
        "seed": args.seed,
        "save_format": args.save_format,
        "dry_run": args.dry_run,
        "cases": [],
        "unsupported_groups": unsupported_groups,
        "archive_path": None,
    }

    command_lines: list[str] = []
    failures = 0
    missing = 0
    run_index = 0

    for index, case in enumerate(cases, start=1):
        checkpoint_abs = (REPO_ROOT / case.checkpoint_path).resolve()
        stop_requested = False

        for run_label, manifest, suffix in _iter_case_runs(args, case):
            run_index += 1
            run_name = f"{case.name}{suffix}"
            case_output_dir = cases_dir / run_name
            log_path = logs_dir / f"{run_name}.log"
            command = _build_command(args, case, manifest, case_output_dir)
            command_line = _format_command(command)
            command_lines.append(command_line)

            case_record = {
                **asdict(case),
                "index": run_index,
                "case_index": index,
                "run_label": run_label,
                "run_name": run_name,
                "manifest": str(manifest),
                "output_dir": str(case_output_dir),
                "log_path": str(log_path),
                "command": command_line,
                "checkpoint_exists": checkpoint_abs.is_file(),
                "manifest_exists": manifest.is_file(),
                "status": "pending",
                "returncode": None,
                "elapsed_sec": None,
            }

            if not checkpoint_abs.is_file():
                missing += 1
                case_record["status"] = "missing_checkpoint"
                _write_text(log_path, f"Missing checkpoint: {checkpoint_abs}\n\nPlanned command:\n{command_line}\n")
                summary["cases"].append(case_record)
                _write_json(args.output_root / "summary.json", summary)
                if args.fail_fast:
                    stop_requested = True
                    break
                continue

            if not manifest.is_file():
                missing += 1
                case_record["status"] = "missing_manifest"
                _write_text(log_path, f"Missing manifest: {manifest}\n\nPlanned command:\n{command_line}\n")
                summary["cases"].append(case_record)
                _write_json(args.output_root / "summary.json", summary)
                if args.fail_fast:
                    stop_requested = True
                    break
                continue

            if args.dry_run:
                case_record["status"] = "dry_run"
                _write_text(log_path, f"Dry run. Planned command:\n{command_line}\n")
                summary["cases"].append(case_record)
                _write_json(args.output_root / "summary.json", summary)
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
                stop_requested = True
                break

        if stop_requested:
            break

    _write_text(args.output_root / "commands.txt", "\n".join(command_lines) + "\n")

    readme = [
        "PiD from-clean checkpoint validation",
        "",
        f"Manifest: {args.manifest}",
        f"4K extra manifest: {args.manifest_1024}",
        f"Sigmas: {args.degrade_sigmas}",
        f"Cases directory: {cases_dir}",
        f"Logs directory: {logs_dir}",
        "",
        "Each run output is written under cases/<run-name>/ using the from_clean.py layout:",
        "  input/",
        "  vae_decode/sigma_*/",
        "  <run-tag>/sigma_*/",
        "",
        "For 4K checkpoints, the extra 1024 manifest run uses a __1024 suffix.",
        "",
        "Registry aliases that do not have a from_clean entry are listed in summary.json as covered aliases.",
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
    if unsupported_groups:
        print(f"Unsupported checkpoint groups: {len(unsupported_groups)}; see summary.json")

    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
