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
RAE (Representation Autoencoder) generation helpers for the from_ldm --backbone dinov2 demo.

Loads the DINOv2-B encoder + ViT-XL decoder (stage-1 RAE) and the DiT^DH-XL
class-conditional ImageNet-512 diffusion model (stage-2) + DiT^DH-S guidance
model, then provides a `sample_fn` that returns the full ODE trajectory in
one call. Capturing `--save_xt_steps K` intermediates is a tensor index into
that trajectory — no callback plumbing needed.

Requires the upstream RAE GitHub repo (https://github.com/bytetriper/RAE)
cloned somewhere on disk. The location is resolved from the
``RAE_REPO_PATH`` environment variable (default: ``../RAE``, i.e. cloned
as a sibling of the pid working tree); the ``--rae_repo_path`` CLI flag
falls back to that default.
"""

import argparse
import logging
import math
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default location: ``$RAE_REPO_PATH`` if set, else ``../RAE`` relative to CWD
# (the README convention is to clone RAE as a sibling of the pid working tree).
DEFAULT_RAE_REPO_PATH = os.environ.get("RAE_REPO_PATH", "../RAE")


def _ensure_rae_on_path(repo_path: str) -> None:
    """Insert <repo_path>/src at the front of sys.path so `from stage1 import RAE` works."""
    src_dir = os.path.join(repo_path, "src")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"RAE repo src dir not found: {src_dir}. Set RAE_REPO_PATH or pass "
            f"--rae_repo_path. See README for installation instructions."
        )
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _patch_rae_decoder_config(repo_path: str) -> None:
    """Rewrite RAE's ViTXL `decoder/config.json` if its `patch_size` is the
    legacy ``"SHOULD BE RELOADED"`` placeholder.

    The placeholder was historically a forcing function: RAE.__init__ overwrites
    `patch_size` two lines after `AutoConfig.from_pretrained`. Newer `transformers`
    (5.x) routes config-from-dict through `huggingface_hub`'s strict dataclass
    validator, which rejects the string before the override gets a chance to
    fire. We patch it to an int (RAE still overwrites it) so the existing
    upstream RAE code path keeps working.
    """
    import json

    cfg_path = os.path.join(repo_path, "configs", "decoder", "ViTXL", "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if cfg.get("patch_size") == "SHOULD BE RELOADED":
        cfg["patch_size"] = 16  # value is irrelevant — RAE.__init__ overwrites it
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


def load_class_names(txt_path: str) -> list[str]:
    """Read the 1000-line human-readable ImageNet-1k class-name file."""
    with open(txt_path) as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != 1000:
        raise ValueError(f"Expected 1000 ImageNet class names in {txt_path}, got {len(names)}")
    return names


def compute_t_schedule(
    num_timepoints: int,
    time_dist_shift: float,
    t0: float = 0.0,
    t1: float = 1.0 - 1.0 / 1000,
) -> torch.Tensor:
    """Replicate the RAE ODE time schedule (integrators.py:99-101).

    Returns a length-`num_timepoints` tensor, monotone decreasing from ≈1 (noise)
    to ≈0 (clean). Used to tag each xt snapshot with its noise level.
    """
    t = 1.0 - torch.linspace(t0, t1, num_timepoints)
    t = time_dist_shift * t / (1 + (time_dist_shift - 1) * t)
    return t


def _load_dit(
    hidden: list,
    depth: list,
    num_heads: list,
    ckpt_path: str,
    device: str,
):
    """Load a DiTwDDTHead with the ImageNet-512 topology and pull 'ema' state when present.

    DiT params stay in fp32 — bf16 is applied via `torch.autocast` in
    `sample_rae_trajectory` (mirrors upstream `sample_ddp.py:248`). Casting the
    DiT to bf16 directly breaks because torchdiffeq's solver builds `t` as a
    fp32 scalar (see integrators.py:110), which collides with bf16 Linear
    weights inside `t_embedder`.
    """
    from stage2.models.DDT import DiTwDDTHead  # noqa: local import after sys.path hook

    m = DiTwDDTHead(
        input_size=32,
        patch_size=1,
        in_channels=768,
        hidden_size=hidden,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "ema" in sd:
        sd = sd["ema"]
    m.load_state_dict(sd, strict=True)
    return m.to(device=device).eval().requires_grad_(False)


def load_rae_stack(
    repo_path: str,
    decoder_ckpt: str,
    stats_path: str,
    dit_main_ckpt: str,
    dit_guid_ckpt: str,
    num_inference_steps: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Build (RAE, DiT main, DiT guidance, sample_fn, t_schedule) for ImageNet-512 autoguidance.

    Hyperparameters are pinned to the config in
    `configs/stage2/sampling/ImageNet512/DiTDH-XL_DINOv2-B_decXL_AG.yaml`.

    Returns:
        rae:         RAE module (DINOv2 encoder + ViT-XL decoder + stats).
        dit_main:    DiT^DH-XL (main score network).
        dit_guid:    DiT^DH-S (autoguidance network).
        sample_fn:   Callable (z, model_fwd, **kwargs) -> trajectory tensor of
                     shape (num_inference_steps+1, B, 768, 32, 32).
        t_schedule:  (num_inference_steps+1,) tensor of time values; [0]≈1 (noise),
                     [-1]≈0 (clean). Use t_schedule[K] as the noise label of
                     trajectory[K].
    """
    _ensure_rae_on_path(repo_path)
    _patch_rae_decoder_config(repo_path)

    from stage1 import RAE  # noqa: local import after sys.path hook
    from stage2.transport import Sampler, create_transport

    rae = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="facebook/dinov2-with-registers-base",
        encoder_input_size=448,
        encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
        # Pass an absolute path so rae.py's AutoConfig.from_pretrained() does
        # not rely on CWD == RAE repo root.
        decoder_config_path=os.path.join(repo_path, "configs/decoder/ViTXL"),
        pretrained_decoder_path=decoder_ckpt,
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=stats_path,
    )
    rae = rae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    # `rae`'s normalization stats (latent_mean, latent_var, encoder_mean, encoder_std)
    # are plain tensor attributes — not `register_buffer`d — so `.to(dtype=...)`
    # leaves them in fp32. That breaks downstream when `rae.decode` multiplies
    # bf16 latents by fp32 stats and feeds the resulting fp32 tensor into a
    # bf16 decoder. Cast them explicitly to keep the chain in one dtype.
    for _name in ("latent_mean", "latent_var", "encoder_mean", "encoder_std"):
        _val = getattr(rae, _name, None)
        if isinstance(_val, torch.Tensor):
            setattr(rae, _name, _val.to(device=device, dtype=dtype))

    dit_main = _load_dit(
        hidden=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        ckpt_path=dit_main_ckpt,
        device=device,
    )
    dit_guid = _load_dit(
        hidden=[384, 2048],
        depth=[12, 2],
        num_heads=[6, 16],
        ckpt_path=dit_guid_ckpt,
        device=device,
    )

    # time_dist_shift = sqrt(C*H*W / 4096) — see misc.time_dist_shift_dim/base
    # in the RAE YAML. For (768, 32, 32): sqrt(786432 / 4096) ≈ 13.856.
    shift = math.sqrt(32 * 32 * 768 / 4096)
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        time_dist_type="uniform",
        time_dist_shift=shift,
    )
    num_timepoints = num_inference_steps + 1
    sample_fn = Sampler(transport).sample_ode(
        sampling_method="euler",
        num_steps=num_timepoints,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    )
    t_schedule = compute_t_schedule(num_timepoints, shift)
    return rae, dit_main, dit_guid, sample_fn, t_schedule


@torch.no_grad()
def sample_rae_trajectory(
    class_id: int,
    dit_main,
    dit_guid,
    sample_fn,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    cfg_scale: float = 1.5,
    cfg_interval=(0.0, 1.0),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample the full ODE trajectory for a single class ID.

    Returns a tensor of shape (num_timepoints, 768, 32, 32) — conditional branch
    only (the autoguidance unconditional half is discarded).
    """
    # DiTs are kept in fp32 (see `_load_dit` docstring) — bf16/fp16 acceleration
    # is delivered via autocast, mirroring upstream `sample_ddp.py:248`.
    z = torch.randn(1, 768, 32, 32, generator=generator, device=device, dtype=torch.float32)
    # Autoguidance requires duplicating the batch: conditional + null class.
    z = torch.cat([z, z], dim=0)
    y = torch.tensor([class_id, 1000], device=device)  # 1000 = null class
    kwargs = dict(
        y=y,
        cfg_scale=cfg_scale,
        cfg_interval=tuple(cfg_interval),
        additional_model_forward=dit_guid.forward,
    )
    use_autocast = dtype != torch.float32
    if use_autocast:
        with torch.autocast(device_type="cuda", dtype=dtype):
            traj = sample_fn(z, dit_main.forward_with_autoguidance, **kwargs)
    else:
        traj = sample_fn(z, dit_main.forward_with_autoguidance, **kwargs)
    # traj: (num_timepoints, 2, 768, 32, 32). Split along batch, keep conditional.
    cond, _uncond = traj.chunk(2, dim=1)
    return cond.squeeze(1)


@torch.no_grad()
def decode_rae_latent(rae, latent: torch.Tensor) -> torch.Tensor:
    """Decode (B, 768, 32, 32) normalized RAE latent to (B, 3, 512, 512) in [0, 1].

    RAE.decode internally applies the inverse latent normalization (using the
    ImageNet-1k stats) and returns pixels in the [0, 1] ImageNet-denormalized space.

    The latent's dtype may diverge from `rae`'s param dtype: DiT sampling keeps
    state in fp32 (see `sample_rae_trajectory`) while `rae` itself is loaded in
    the user-selected dtype (default bf16). Cast on entry to keep `decoder_embed`
    happy.
    """
    # rae has mixed dtypes (encoder vs decoder vs buffers), so target the decoder
    # specifically — that's what `rae.decode` calls.
    decoder_dtype = next(rae.decoder.parameters()).dtype
    out = rae.decode(latent.to(decoder_dtype))
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# CLI helpers — imported by cli_utils.py's parser when backbone=="rae".
# ---------------------------------------------------------------------------


def add_rae_args(p: argparse.ArgumentParser) -> None:
    """Register RAE-specific CLI flags."""
    p.add_argument(
        "--rae_class_ids",
        nargs="+",
        type=int,
        default=None,
        help="ImageNet-1k class IDs to generate (0..999). Mutually exclusive with --rae_class_range.",
    )
    p.add_argument(
        "--rae_class_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Class ID range [START, END) — e.g. 0 1000 generates all classes.",
    )
    p.add_argument(
        "--rae_repo_path",
        type=str,
        default=DEFAULT_RAE_REPO_PATH,
        help=(
            "Path to the RAE repo (expects <path>/src/stage1, <path>/src/stage2, <path>/configs). "
            f"Defaults to $RAE_REPO_PATH or '../RAE' (current: {DEFAULT_RAE_REPO_PATH!r})."
        ),
    )
    p.add_argument(
        "--rae_decoder_ckpt",
        type=str,
        default="checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt",
        help="ViT-XL decoder weights for the DINOv2-B 512 RAE.",
    )
    p.add_argument(
        "--rae_stats_path",
        type=str,
        default="checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt",
        help="Per-(C,H,W) latent normalization stats for the DINOv2-B 512 RAE.",
    )
    p.add_argument(
        "--rae_dit_main_ckpt",
        type=str,
        default=None,
        help=(
            "DiT^DH-XL main score checkpoint. Defaults to "
            "<rae_repo_path>/models/DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-XL_ep400/stage2_model.pt"
        ),
    )
    p.add_argument(
        "--rae_dit_guid_ckpt",
        type=str,
        default=None,
        help=(
            "DiT^DH-S autoguidance checkpoint. Defaults to "
            "<rae_repo_path>/models/DiTs/Dinov2/wReg_base/ImageNet512/DiTDH-S_ep20/stage2_model.pt"
        ),
    )
    p.add_argument("--rae_cfg_scale", type=float, default=1.5, help="Autoguidance scale.")
    p.add_argument(
        "--rae_cfg_interval",
        nargs=2,
        type=float,
        metavar=("T_MIN", "T_MAX"),
        default=[0.0, 1.0],
        help="Time interval in which autoguidance is applied; defaults to entire schedule.",
    )


def resolve_rae_class_ids(args) -> list[int]:
    """Validate + expand --rae_class_ids / --rae_class_range into a list."""
    has_ids = args.rae_class_ids is not None
    has_range = args.rae_class_range is not None
    if has_ids == has_range:
        raise ValueError("Must provide exactly one of --rae_class_ids or --rae_class_range for backbone=rae")
    if has_ids:
        ids = list(args.rae_class_ids)
    else:
        start, end = args.rae_class_range
        if start < 0 or end > 1000 or start >= end:
            raise ValueError(f"--rae_class_range must satisfy 0 <= START < END <= 1000, got [{start}, {end})")
        ids = list(range(start, end))
    for cid in ids:
        if cid < 0 or cid >= 1000:
            raise ValueError(f"Class ID {cid} out of [0, 1000)")
    return ids


def resolve_rae_dit_ckpts(args) -> tuple[str, str]:
    """Fall back to ``<rae_repo_path>/models/DiTs/...`` when CLI omits the explicit ckpt paths."""
    base = os.path.join(args.rae_repo_path, "models", "DiTs", "Dinov2", "wReg_base", "ImageNet512")
    main_ckpt = args.rae_dit_main_ckpt or os.path.join(base, "DiTDH-XL_ep400", "stage2_model.pt")
    guid_ckpt = args.rae_dit_guid_ckpt or os.path.join(base, "DiTDH-S_ep20", "stage2_model.pt")
    return main_ckpt, guid_ckpt


def run_rae_demo(args):
    """from_ldm demo flow for the RAE (DINOv2 + ViT-XL) class-conditional backbone.

    Bypasses the diffusers pipeline: samples the full RAE ODE trajectory, slices the
    requested `--save_xt_steps` snapshots plus the final clean x0, decodes each with the
    RAE decoder (baseline), and runs the shared PiD decode/save step. Invoked from
    from_ldm.py when --backbone is "dinov2".
    """
    from pid._src.inference.cli_utils import maybe_init_distributed
    from pid._src.inference.decoder import load_our_decoder, run_ours_and_save_step
    from pid._src.inference.inference_utils import AsyncUploader, build_tag, get_rank_and_world_size

    rank, world_size = get_rank_and_world_size()
    maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    if args.resolution != 512:
        raise ValueError(f"dinov2 backbone only supports --resolution 512, got {args.resolution}")
    num_inference_steps = args.num_inference_steps
    save_xt_set = sorted(set(args.save_xt_steps)) if args.save_xt_steps else []
    for k in save_xt_set:
        if k < 1 or k > num_inference_steps:
            raise ValueError(f"--save_xt_steps value {k} out of range [1, {num_inference_steps}]")

    class_ids = resolve_rae_class_ids(args)
    rae_dit_main_ckpt, rae_dit_guid_ckpt = resolve_rae_dit_ckpts(args)
    class_names_path = os.path.join(os.path.dirname(__file__), "prompts", "imagenet_classes.txt")
    class_names = load_class_names(class_names_path)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tag = build_tag(args, "dinov2")
    if is_rank0:
        logger.info(
            f"Backbone: dinov2  resolution: 512  num_inference_steps: {num_inference_steps}  "
            f"rae_cfg_scale: {args.rae_cfg_scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Classes: {len(class_ids)}  save_xt_steps: {save_xt_set}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load RAE stack (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        rae = dit_main = dit_guid = sample_fn = t_schedule = None
        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading RAE stack ({msg}) ...")
                rae, dit_main, dit_guid, sample_fn, t_schedule = load_rae_stack(
                    repo_path=args.rae_repo_path,
                    decoder_ckpt=args.rae_decoder_ckpt,
                    stats_path=args.rae_stats_path,
                    dit_main_ckpt=rae_dit_main_ckpt,
                    dit_guid_ckpt=rae_dit_guid_ckpt,
                    num_inference_steps=num_inference_steps,
                    device="cuda",
                    dtype=dtype,
                )
            dist.barrier()
    else:
        logger.info("Loading RAE stack ...")
        rae, dit_main, dit_guid, sample_fn, t_schedule = load_rae_stack(
            repo_path=args.rae_repo_path,
            decoder_ckpt=args.rae_decoder_ckpt,
            stats_path=args.rae_stats_path,
            dit_main_ckpt=rae_dit_main_ckpt,
            dit_guid_ckpt=rae_dit_guid_ckpt,
            num_inference_steps=num_inference_steps,
            device="cuda",
            dtype=dtype,
        )

    if is_rank0:
        logger.info(f"t_schedule range: {t_schedule[0].item():.4f} (noise) -> {t_schedule[-1].item():.4f} (clean)")

    model = load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or "./results/official_demo/dinov2"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    indexed_classes = list(enumerate(class_ids))
    if world_size > 1:
        indexed_classes = indexed_classes[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_classes)} classes")

    for prompt_idx, cid in indexed_classes:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        gen = torch.Generator(device="cuda").manual_seed(seed)
        caption = class_names[cid]

        logger.info(f"[{prompt_idx}] Sampling RAE trajectory (seed={seed}, class={cid}: {caption!r})")
        traj = sample_rae_trajectory(
            class_id=cid,
            dit_main=dit_main,
            dit_guid=dit_guid,
            sample_fn=sample_fn,
            device="cuda",
            dtype=dtype,
            cfg_scale=args.rae_cfg_scale,
            cfg_interval=args.rae_cfg_interval,
            generator=gen,
        )  # (num_inference_steps+1, 768, 32, 32)

        # Yield (label, latent_[1,768,32,32], sigma) for each xt step + final x0.
        steps: list[tuple[str, torch.Tensor, float]] = []
        for K in save_xt_set:
            steps.append((f"{K:02d}xt", traj[K : K + 1], float(t_schedule[K].item())))
        steps.append(("x0", traj[-1:], 0.0))

        for step_label, latent, sigma in steps:
            with torch.no_grad():
                baseline_01 = decode_rae_latent(rae, latent)  # (1, 3, 512, 512) in [0, 1]

            run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=baseline_01,
                sigma=sigma,
                caption=caption,
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="dinov2_decode",
                baseline_upload_tag_prefix="dinov2_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


__all__ = [
    "DEFAULT_RAE_REPO_PATH",
    "add_rae_args",
    "compute_t_schedule",
    "decode_rae_latent",
    "load_class_names",
    "load_rae_stack",
    "resolve_rae_class_ids",
    "resolve_rae_dit_ckpts",
    "run_rae_demo",
    "sample_rae_trajectory",
]
