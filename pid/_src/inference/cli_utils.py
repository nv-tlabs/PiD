# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI construction + argument parsing for the from_ldm / from_clean dispatchers.

Both demos select the backbone with a single `--backbone` flag whose choices are the
canonical registry keys. The parser shape for from_ldm is backbone-flavored (diffusers
backbones expose --ldm_inference_steps / --guidance_scale / --cpu_offload /
--backbone_model_id; dinov2 / siglip expose their own arg groups), so we pre-parse
--backbone first, then build the full parser.
"""

import argparse

import torch

from pid._src.inference.checkpoint_registry import VALID_CKPT_TYPES, get_pid_checkpoint
from pid._src.inference.pipeline_registry import PIPELINE_REGISTRY

# Canonical backbone keys (registry keys), per demo.
LDM_BACKBONES = [
    "flux",
    "flux2",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "sd3",
    "sdxl",
    "qwenimage",
    "qwenimage-2512",
    "zimage",
    "zimage-turbo",
    "dinov2",
    "siglip",
]
CLEAN_BACKBONES = ["flux", "flux2", "sd3", "sdxl", "qwenimage", "dinov2", "siglip"]


def parse_resolution(s: str) -> tuple[int, int]:
    """Parse "512" -> (512, 512) or "W,H" -> (H, W).

    The comma form is width,height (more intuitive, e.g. 2304,1728 for a 4:3 landscape);
    the returned tuple is (H, W) in pixels for downstream consumers.
    """
    parts = s.strip().split(",")
    if len(parts) == 1:
        n = int(parts[0])
        return n, n
    if len(parts) == 2:
        w, h = int(parts[0]), int(parts[1])
        return h, w
    raise argparse.ArgumentTypeError(f"--resolution must be 'N' or 'W,H', got {s!r}")


def maybe_init_distributed(world_size: int, rank: int):
    if world_size > 1:
        import torch.distributed as dist

        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=device)


def _add_common_decoder_args(parser: argparse.ArgumentParser, default_output_subdir: str, default_seed: int = 0):
    """Add the PiD-decoder + output/S3 flags common to both demos."""
    # --experiment / --checkpoint_path default to whatever checkpoint_registry.py registers
    # for this backbone — pass them explicitly only to override.
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Our pixel decoder experiment config name (default: checkpoint_registry[backbone].experiment)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="pid/_src/configs/pid/config.py",
        help="Hydra config file for our decoder",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to our pixel decoder checkpoint (default: checkpoint_registry[backbone].checkpoint_path)",
    )
    parser.add_argument(
        "--pid_ckpt_type",
        type=str,
        choices=list(VALID_CKPT_TYPES),
        default="2k",
        help="Which PiD checkpoint variant to load from the registry when "
        "--experiment / --checkpoint_path are omitted. Default: '2k' (the "
        "original 2048px-trained decoders). '2kto4k' picks the v1 multi-res "
        "decoders; '2kto4k_v1pt5' picks the v1.5 multi-res decoders "
        "(1024 LDM → 4K output).",
    )
    parser.add_argument("--load_ema_to_reg", action="store_true", help="Load EMA weights into the regular model")

    # Our decoder inference params (common)
    parser.add_argument(
        "--seed", type=int, default=default_seed, help="Base random seed (incremented per prompt/class/sample)"
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Our pixel decoder CFG scale")
    parser.add_argument(
        "--pid_inference_steps",
        type=int,
        default=4,
        help="Pixel-diffusion decoder denoising steps (default from model config)",
    )
    parser.add_argument("--shift", type=float, default=None, help="Our pixel decoder flow shift")
    parser.add_argument("--scale", type=int, default=4, help="Our decoder upscale factor (output = baseline * scale)")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the pixel decoder net for faster inference. Compiles once per "
        "output resolution (first call is slow). Standard SR path only (no context-parallel / "
        "encoder-decoder).",
    )

    # Output / S3 (common)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: ./results/{default_output_subdir}/<backbone>",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["png", "jpg"],
        default="jpg",
        help="Image format for saved outputs (jpg uses quality=95)",
    )
    parser.add_argument("--upload", action="store_true", help="Upload results to S3 (async)")
    parser.add_argument("--note", type=str, default="", help="Note appended to tag")


def _fill_registry_defaults(args):
    """Fill --experiment / --checkpoint_path from the official registry when omitted."""
    if args.experiment is None or args.checkpoint_path is None:
        default_ckpt = get_pid_checkpoint(args.backbone, args.pid_ckpt_type)
        if args.experiment is None:
            args.experiment = default_ckpt.experiment
        if args.checkpoint_path is None:
            args.checkpoint_path = default_ckpt.checkpoint_path


# =============================================================================
# from_ldm parser
# =============================================================================


def build_ldm_parser(backbone: str) -> argparse.ArgumentParser:
    # Backbone-flavored parser. Diffusers-only args (--ldm_inference_steps, --guidance_scale,
    # --cpu_offload, --backbone_model_id) are skipped for "dinov2" and "siglip" because those
    # backbones bypass the diffusers pipeline. The dinov2 backbone is class-conditional, so
    # --prompt / --prompt_file are also omitted for it; siglip remains text-conditional.
    is_dinov2 = backbone == "dinov2"
    is_siglip = backbone == "siglip"
    is_diffusers = not (is_dinov2 or is_siglip)
    cfg = PIPELINE_REGISTRY[backbone] if is_diffusers else None

    parser = argparse.ArgumentParser(
        description=f"Official demo: {backbone} latent diffusion vs ours pixel-diffusion decoder"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=LDM_BACKBONES,
        help="Which LDM backbone to run.",
    )

    _add_common_decoder_args(parser, default_output_subdir="official_demo")
    parser.add_argument("--group_name", type=str, default="official_demo", help="S3 group name")

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="Backbone dtype",
    )

    # Step capture (intermediate xt) — common to all backbones; range validated per-flow.
    parser.add_argument(
        "--save_xt_steps",
        type=int,
        nargs="+",
        default=None,
        help="Capture noisy latent AFTER K forward passes for each K (1-indexed). "
        "Final clean latent (x0) is always saved. K must be in [1, num_inference_steps].",
    )

    # ===== Backbone-specific =====
    if is_diffusers:
        parser.add_argument(
            "--backbone_model_id",
            type=str,
            default=None,
            help=f"Override HuggingFace model ID (default: {cfg.default_model_id})",
        )
        parser.add_argument(
            "--resolution",
            type=parse_resolution,
            default=(2048, 2048),
            help="Final (post-SR) output resolution in pixels. Either 'N' (square N x N) or 'W,H'. "
            "The LDM runs at this divided by --scale (must be evenly divisible). "
            "Default: 2048 (→ 512 LDM at the default scale 4).",
        )
        parser.add_argument(
            "--ldm_inference_steps",
            type=int,
            default=None,
            help=f"Latent diffusion backbone denoising steps. Default: {cfg.default_num_inference_steps}",
        )
        parser.add_argument(
            "--guidance_scale",
            type=float,
            default=None,
            help=f"Backbone CFG scale. Default: {cfg.default_guidance_scale}",
        )
        parser.add_argument(
            "--cpu_offload",
            action="store_true",
            help="Use enable_model_cpu_offload (needed for large models like Flux2 / QwenImage on small GPUs)",
        )
        prompt_group = parser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument("--prompt", type=str, default=None, help="Single inline prompt string")
        prompt_group.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")
    elif is_dinov2:
        # Class-conditional ImageNet-512. RAE-specific flags come from rae_generation.add_rae_args.
        from pid._src.inference.rae_generation import add_rae_args

        add_rae_args(parser)
        parser.add_argument(
            "--num_inference_steps",
            type=int,
            default=50,
            help="RAE ODE step count. Drives Sampler.sample_ode(num_steps=num_inference_steps+1).",
        )
        parser.add_argument("--resolution", type=int, default=512, help="dinov2 backbone only supports 512.")
    else:  # siglip
        from pid._src.inference.scale_rae_generation import add_scale_rae_args

        add_scale_rae_args(parser)
        parser.add_argument(
            "--resolution",
            type=int,
            default=256,
            help="siglip backbone only supports 256 (decoder is 14-multiple 224, bicubic to 256).",
        )
        prompt_group = parser.add_mutually_exclusive_group(required=True)
        prompt_group.add_argument("--prompt", type=str, default=None, help="Single inline prompt string")
        prompt_group.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")

    return parser


def parse_ldm_args() -> argparse.Namespace:
    # Pre-parse --backbone to decide the parser's backbone-specific shape.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backbone", type=str, required=True, choices=LDM_BACKBONES)
    known, _ = pre.parse_known_args()
    backbone = known.backbone

    parser = build_ldm_parser(backbone)
    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown
    args.backbone = backbone
    _fill_registry_defaults(args)
    return args


# =============================================================================
# from_clean parser
# =============================================================================


def build_clean_parser(backbone: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"From-clean demo: image -> {backbone} VAE encode -> optional noise -> ours pixel decoder"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=CLEAN_BACKBONES,
        help="Which VAE the loaded checkpoint uses.",
    )

    # from_clean historically defaulted --seed to 5 — preserve for reproducibility.
    _add_common_decoder_args(parser, default_output_subdir="official_demo_from_clean", default_seed=5)
    parser.add_argument("--group_name", type=str, default="official_demo_from_clean", help="S3 group name")

    # Input source (mutually exclusive, required): single --input_path OR a JSONL --manifest.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_path", type=str, default=None, help="Path to a single input image (PNG/JPG/...).")
    input_group.add_argument(
        "--manifest",
        type=str,
        default=None,
        help='JSONL file with one {"image": <path>, "prompt": <str>} object per line. '
        'The "prompt" key is optional and falls back to --prompt → model.config.fixed_positive_prompt. '
        "Image paths are interpreted as absolute or CWD-relative. Samples are round-robin sharded across ranks.",
    )

    # Text prompt — used as a global default (or only prompt under --input_path mode).
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt describing the input image — fed into the pixel decoder's data batch as the "
        "caption condition. Under --input_path, this is THE prompt. Under --manifest, this is the "
        "fallback used for entries without a per-line 'prompt' key.",
    )

    # The input image is always fed at its native resolution (only center-cropped so each
    # side is a multiple of 16 to keep the VAE latent grid integer); dinov2 / siglip resize
    # internally to their own fixed native interface.

    # Noise sweep
    parser.add_argument(
        "--degrade_sigmas",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of sigma values (each in [0, 1]) to inject into the clean latent before decoding. "
        "0.0 = clean round-trip. One decode + save per sigma.",
    )
    return parser


def parse_clean_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backbone", type=str, required=True, choices=CLEAN_BACKBONES)
    known, _ = pre.parse_known_args()
    backbone = known.backbone

    parser = build_clean_parser(backbone)
    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown
    args.backbone = backbone
    _fill_registry_defaults(args)

    for s in args.degrade_sigmas:
        if not (0.0 <= s <= 1.0):
            parser.error(f"--degrade_sigmas value {s} out of range [0.0, 1.0]")
    return args
