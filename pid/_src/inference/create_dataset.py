#!/usr/bin/env python
"""
Generate webdataset tars from diffusion backbone outputs for FPD evaluation.

Runs a diffusers backbone (Flux / SDXL / SD3 / Flux2 / QwenImage / ZImage) on text prompts, extracts
the native VAE latent + decoded image, and writes them as sharded webdataset tars
compatible with fpd_with_GT/benchmark.py.

No FPD inference or metrics here — just dataset creation.
Generate once, evaluate many times with benchmark.py.

Supports multi-GPU via torchrun: each rank processes a disjoint subset of samples
and writes to separate shards. Rank 0 writes the final wdinfo.json.

Output structure (e.g. Flux at 1024px):
  <output_dir>/aspect_ratio_1_1/
    image_1024/part_00000000/00000000.tar
    flux_latent_1024/part_00000000/00000000.tar
    caption/part_00000000/00000000.tar
    wdinfo.json

Usage:
  # Single GPU — 100 prompts from prompts.txt, 5 images each
  PYTHONPATH=. python -m pid._src.inference.create_dataset \
      --backbone flux \
      --prompts_file pid/_src/inference/prompts.txt \
      --num_images_per_prompt 1 \
      --output_dir data/generated_latent_webdataset_val/flux/ \
      --max_samples_per_shard 10 \
      --seed 42

  # Multi-GPU (8 GPUs) — same dataset, ~8x faster
  PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=8 \
      -m pid._src.inference.create_dataset \
      --backbone flux \
      --num_images_per_prompt 5 \
      --output_dir data/generated_latent_webdataset/flux/ \
      --max_samples_per_shard 10 \
      --seed 42

  # Quick test — inline prompts, 1 image each
  PYTHONPATH=. python -m pid._src.inference.create_dataset \
      --backbone flux --prompts "a cat" "a dog" \
      --num_images_per_prompt 1 \
      --output_dir /tmp/test_fpd_gen/ --seed 42

  # Save raw noisy xt latents at specified denoising steps (all backbones).
  # Creates {backbone}-{step}step_xt/ alongside {backbone}/ with same structure.
  # Uses diffusers callback_on_step_end for generic xt capture.
  PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=8 \
      -m pid._src.inference.create_dataset \
      --backbone flux \
      --prompts_file pid/_src/inference/prompts.txt \
      --num_images_per_prompt 1 \
      --num_inference_steps 28 --save_xt_steps 4 8 12 16 20 24 \
      --output_dir data/generated_latent_xt_webdataset/flux/ --seed 42 --resolution 960

  # Save x_0 prediction at specified denoising steps (Flux only).
  # x_0_pred = x_t - sigma * velocity (flow-matching). Creates
  # {backbone}-{step}step_x0/ alongside {backbone}/. Uses a forward hook on
  # pipeline.transformer to capture velocity, since callback_on_step_end can't reach it.
  PYTHONPATH=. /usr/local/bin/torchrun --nproc_per_node=8 \
      -m pid._src.inference.create_dataset \
      --backbone flux \
      --prompts_file pid/_src/inference/prompts_half_text.txt \
      --num_images_per_prompt 1 \
      --num_inference_steps 28 --save_x0_steps 4 8 12 16 20 24 \
      --output_dir data/generated_latent_x0_webdataset/flux/ --seed 42 --resolution 512
"""

"""
New model backend
# QwenImage
PYTHONPATH=. python -m pid._src.inference.create_dataset \
    --backbone qwenimage \
    --prompts_file pid/_src/inference/prompts_w_text.txt \
    --num_images_per_prompt 1 \
    --output_dir data/generated_latent_xt_webdataset/qwenimage/ \
    --seed 42

# ZImage
PYTHONPATH=. python -m pid._src.inference.create_dataset \
    --backbone zimage \
    --prompts_file pid/_src/inference/prompts_w_text.txt \
    --num_images_per_prompt 1 \
    --output_dir data/generated_latent_xt_webdataset/zimage/ \
    --seed 42

# Flux2
PYTHONPATH=. python -m pid._src.inference.create_dataset \
    --backbone flux2 \
    --prompts_file pid/_src/inference/prompts.txt \
    --num_images_per_prompt 1 \
    --output_dir data/generated_latent_xt_webdataset/flux2/ \
    --seed 42
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from PIL import Image

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize distributed process group if launched via torchrun. Returns (rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def print_rank0(msg: str, rank: int):
    if rank == 0:
        print(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Generate webdataset from diffusion backbone outputs")

    # Backbone
    p.add_argument(
        "--backbone",
        required=True,
        choices=["flux", "sdxl", "sd3", "flux2", "qwenimage", "zimage", "zimage_turbo", "rae", "scale_rae"],
    )
    p.add_argument("--backbone_model_id", type=str, default=None, help="Override HF model ID")

    # Prompts (ignored when backbone=="rae"; use --rae_class_ids / --rae_class_range instead)
    p.add_argument("--prompts", nargs="+", type=str, default=None, help="Inline prompts")
    p.add_argument("--prompts_file", type=str, default=None, help="File with one prompt per line")
    p.add_argument("--num_images_per_prompt", type=int, default=1)

    # RAE-specific arguments are registered in rae_generation.add_rae_args.
    from pid._src.inference.rae_generation import add_rae_args

    add_rae_args(p)

    # Scale-RAE-specific arguments registered in scale_rae_generation.add_scale_rae_args.
    from pid._src.inference.scale_rae_generation import add_scale_rae_args

    add_scale_rae_args(p)

    # Generation params
    p.add_argument("--resolution", type=int, default=None, help="Generation resolution (square)")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)

    # Output
    p.add_argument("--output_dir", required=True, help="Root output dir, e.g. data/generated_latent_webdataset/flux/")
    p.add_argument("--seed", type=int, default=0, help="Base seed (incremented per image)")
    p.add_argument("--max_samples_per_shard", type=int, default=50)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Use enable_model_cpu_offload() to keep weights on CPU and only move the "
        "active component to GPU during forward. Necessary for large models (Flux2, etc.) "
        "that OOM when loading all components onto a single GPU.",
    )

    # Intermediate xt saving (all backbones, via callback_on_step_end)
    p.add_argument(
        "--save_xt_steps",
        nargs="+",
        type=int,
        default=None,
        help="K values at which to save the noisy xt latent AFTER K forward passes of the "
        "network. `--save_xt_steps 16` at N=28 means 'latent that's gone through 16 "
        "denoising steps (16 of 28 steps completed)'; its noise level is sigmas[16]. "
        "Each K gets its own output dir (e.g. flux-16step_xt/). K ∈ [1, num_inference_steps].",
    )

    # Intermediate x0-prediction saving (Flux only — needs the velocity output, which the
    # generic callback_on_step_end can't reach. We hook pipeline.transformer to grab it.)
    p.add_argument(
        "--save_x0_steps",
        nargs="+",
        type=int,
        default=None,
        help="K values at which to save the model's x_0 prediction from the K-th forward "
        "pass. Flow-matching: x_0_pred = x_t - sigma * velocity. degrade_sigma stored is "
        "sigmas[K-1] (sigma of the input that produced this prediction). Each K gets its "
        "own output dir (e.g. flux-16step_x0/). FLUX ONLY. K ∈ [1, num_inference_steps].",
    )

    return p.parse_args()


def load_prompts(args) -> list[str]:
    if args.prompts:
        return args.prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    # Default to bundled prompts.txt
    default_path = Path(__file__).parent / "prompts.txt"
    if default_path.exists():
        with open(default_path) as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("Must provide --prompts or --prompts_file")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def image_tensor_to_png(img: torch.Tensor) -> bytes:
    """Convert (3, H, W) float [0,1] tensor to PNG bytes."""
    import numpy as np

    arr = (img.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def caption_to_bytes(
    prompt: str,
    sample_id: str,
    degrade_sigma: float = 0.0,
    rae_t: float | None = None,
    scale_rae_t: float | None = None,
    scale_rae_guidance: float | None = None,
) -> bytes:
    """Serialize caption as JSON (matches existing webdataset caption format).

    degrade_sigma encodes the noise level of the associated latent. 0.0 for a clean
    final latent; scheduler.sigmas[step_index+1] for an xt captured at denoising step
    `step_index`. Downstream inference reads this per-sample to drive the sigma-aware
    LQ gate.

    rae_t (optional, RAE backbone only) is the raw flow-matching time value at the
    trajectory snapshot — t≈1 for noise, t≈0 for clean. Lets downstream code reason
    about the RAE ODE schedule directly without re-deriving it from `degrade_sigma`.

    scale_rae_t / scale_rae_guidance (optional, Scale-RAE backbone only) are the
    rectified-flow time at the trajectory snapshot and the CFG level used for the
    sample respectively.
    """
    payload = {"prompt": prompt, "file_name": f"{sample_id}.png", "degrade_sigma": float(degrade_sigma)}
    if rae_t is not None:
        payload["rae_t"] = float(rae_t)
    if scale_rae_t is not None:
        payload["scale_rae_t"] = float(scale_rae_t)
    if scale_rae_guidance is not None:
        payload["scale_rae_guidance"] = float(scale_rae_guidance)
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Sharded tar writer (simple synchronous — small datasets)
# ---------------------------------------------------------------------------


class ShardedTarWriter:
    """Write samples to sharded tar files under aspect_ratio_1_1/<key>/part_00000000/.

    Args:
        shard_offset: Starting shard ID. In multi-GPU mode each rank uses a different
            offset so shards don't collide (e.g. rank 0 starts at 0, rank 1 at 1000, ...).
    """

    def __init__(
        self,
        output_dir: str,
        keys: list[str],
        key_ext: dict[str, str],
        max_samples_per_shard: int,
        shard_offset: int = 0,
    ):
        self.base_dir = os.path.join(output_dir, "aspect_ratio_1_1")
        self.keys = keys
        self.key_ext = key_ext
        self.max_samples_per_shard = max_samples_per_shard

        self.current_shard_id = shard_offset
        self.current_shard_count = 0
        self.tar_files: dict[str, tarfile.TarFile] = {}
        self.total_written = 0
        self._open_shards()

    def _shard_path(self, key: str, shard_id: int) -> str:
        part_id = shard_id // 10000
        return os.path.join(self.base_dir, key, f"part_{part_id:08d}", f"{shard_id:08d}.tar")

    def _open_shards(self):
        for key in self.keys:
            path = self._shard_path(key, self.current_shard_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.tar_files[key] = tarfile.open(path, "w")

    def _rotate_shards(self):
        for key in self.keys:
            self.tar_files[key].close()
        self.current_shard_id += 1
        self.current_shard_count = 0
        self._open_shards()

    def add_sample(self, sample_id: str, data: dict[str, bytes]):
        """Add one sample. data maps key -> bytes."""
        for key in self.keys:
            ext = self.key_ext[key]
            raw = data[key]
            info = tarfile.TarInfo(name=f"{sample_id}{ext}")
            info.size = len(raw)
            self.tar_files[key].addfile(info, io.BytesIO(raw))

        self.current_shard_count += 1
        self.total_written += 1
        if self.current_shard_count >= self.max_samples_per_shard:
            self._rotate_shards()

    def close(self):
        for key in self.keys:
            self.tar_files[key].close()


# ---------------------------------------------------------------------------
# wdinfo.json generation
# ---------------------------------------------------------------------------


def write_wdinfo(output_dir: str, keys: list[str], total_samples: int, max_samples_per_shard: int):
    """Write wdinfo.json compatible with SSDDValDataset / benchmark.py."""
    base_dir = os.path.join(output_dir, "aspect_ratio_1_1")
    abs_root = str(Path(base_dir).absolute())

    # Discover tar files from the first key
    ref_key = keys[0]
    ref_dir = os.path.join(base_dir, ref_key)
    tar_paths = []
    for dirpath, _, filenames in os.walk(ref_dir):
        for f in sorted(filenames):
            if f.endswith(".tar"):
                tar_paths.append(os.path.relpath(os.path.join(dirpath, f), os.path.join(base_dir, ref_key)))
    tar_paths.sort()

    # Count samples per shard
    sample_counts = []
    for rel_path in tar_paths:
        tar_full = os.path.join(base_dir, ref_key, rel_path)
        try:
            with tarfile.open(tar_full, "r") as tf:
                count = sum(1 for m in tf if m.isfile())
            sample_counts.append(count)
        except Exception:
            sample_counts.append(max_samples_per_shard)

    wdinfo = {
        "data_keys": keys,
        "root": abs_root,
        "data_list": tar_paths,
        "data_list_key_count": sample_counts,
        "total_key_count": sum(sample_counts),
    }

    wdinfo_path = os.path.join(base_dir, "wdinfo.json")
    with open(wdinfo_path, "w") as f:
        json.dump(wdinfo, f, indent=2)
    print(f"Wrote {wdinfo_path} ({wdinfo['total_key_count']} samples, {len(tar_paths)} shards)")
    return wdinfo_path


# ---------------------------------------------------------------------------
# Generic xt capture via diffusers callback_on_step_end.
# Works for all backbones — no manual denoising loop needed.
# ---------------------------------------------------------------------------


class XtCaptureCallback:
    """Callback for pipeline.__call__() that captures raw noisy xt after K inference steps.

    User semantics: `K in save_ks` means "capture the latent AFTER K forward passes of
    the network". diffusers' `callback_on_step_end(step_index=i)` fires after step_index=i
    executes, so K steps have completed when step_index == K - 1 fires. At that point
    `callback_kwargs["latents"]` is already at `sigmas[K]` and that's what we store as
    the latent's `degrade_sigma`.

    Captured dict is keyed by the user-facing K (not step_index), so output dirs and
    caption JSON land at `flux-{K}step_xt/` with sigma[K] — matching cheatsheet semantics.
    """

    def __init__(self, save_ks: set[int]):
        # Map internal step_index -> user K so the caller can key by K.
        self.save_map = {k - 1: k for k in save_ks}
        self.captured: dict[int, tuple[torch.Tensor, float]] = {}  # keyed by K

    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        k = self.save_map.get(step_index)
        if k is not None:
            sigmas = pipe.scheduler.sigmas
            sigma_idx = min(step_index + 1, len(sigmas) - 1)  # == K
            sigma_val = float(sigmas[sigma_idx].item())
            self.captured[k] = (callback_kwargs["latents"].cpu(), sigma_val)
        return callback_kwargs


class X0CaptureCallback:
    """Capture x_0 prediction from the K-th transformer forward pass (Flux only).

    `callback_on_step_end` only exposes `latents` (xt AFTER the scheduler step), so to
    reach the velocity output of the transformer we register a forward post-hook on
    `pipeline.transformer` that stashes `(last_x_input, last_v_output)` on every call.
    Then in the callback (which fires after each step) we use the most recent stash to
    compute x_0_pred for the just-completed step:

        x_t  = (1 - sigma) * x_0 + sigma * noise         (flow matching)
        v    = noise - x_0                                (Flux predicts velocity)
        =>   x_0_pred = x_input - sigmas[step_index] * v

    User-facing K is 1-indexed (K=1 means "x_0 from the 1st forward pass"); the callback
    fires for step_index = K - 1. degrade_sigma stored is sigmas[K-1] — the sigma of the
    *input* that produced this prediction (matches each_step_vis.py:150-151).

    Flux-only because (a) Flux runs one transformer forward per step (guidance is
    distilled into the model), so the hook records exactly one (x, v) per step; and
    (b) latents stay packed (B, seq_len, 64) — extract_latent handles unpacking later.
    """

    def __init__(self, save_ks: set[int], transformer):
        self.save_map = {k - 1: k for k in save_ks}  # step_index -> user K
        self.captured: dict[int, tuple[torch.Tensor, float]] = {}
        self._last_x: torch.Tensor | None = None
        self._last_v: torch.Tensor | None = None
        self._handle = transformer.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        x = kwargs.get("hidden_states")
        if x is None and args:
            x = args[0]
        v = output[0] if isinstance(output, tuple) else output
        self._last_x = x.detach()
        self._last_v = v.detach()

    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        k = self.save_map.get(step_index)
        if k is not None and self._last_x is not None and self._last_v is not None:
            sigma = float(pipe.scheduler.sigmas[step_index].item())
            x_0_pred = self._last_x.float() - sigma * self._last_v.float()
            self.captured[k] = (x_0_pred.to(self._last_v.dtype).cpu(), sigma)
        return callback_kwargs

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def compose_callbacks(*callbacks):
    """Chain multiple callback_on_step_end-compatible callables into one."""

    def combined(pipe, step_index, timestep, callback_kwargs):
        for cb in callbacks:
            callback_kwargs = cb(pipe, step_index, timestep, callback_kwargs)
        return callback_kwargs

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rank, world_size = init_distributed()
    args = parse_args()

    # RAE follows a different generation path (class-conditional, non-diffusers sampler).
    # Branch early so the diffusers setup below stays untouched.
    if args.backbone == "rae":
        from pid._src.inference.rae_generation import run_rae_main

        try:
            run_rae_main(args, rank, world_size)
        finally:
            if world_size > 1:
                dist.destroy_process_group()
        return

    # Scale-RAE: SigLIP-2 encoder + Qwen LM + DiT diffusion head; non-diffusers path.
    if args.backbone == "scale_rae":
        from pid._src.inference.scale_rae_generation import run_scale_rae_main

        try:
            run_scale_rae_main(args, rank, world_size)
        finally:
            if world_size > 1:
                dist.destroy_process_group()
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompts = load_prompts(args)

    # Fail fast on backbone mismatch (pipeline load is expensive); range checked later.
    if args.save_x0_steps and args.backbone != "flux":
        raise ValueError(f"--save_x0_steps only supports backbone=flux (got {args.backbone})")

    # Validate --save_xt_steps range (checked after num_inference_steps is resolved below)

    # Build flat list of (global_sample_idx, prompt) for all samples, then shard across ranks.
    # global_sample_idx is used for deterministic seed and sample_id regardless of GPU count.
    all_samples = []
    for pi, prompt in enumerate(prompts):
        for img_i in range(args.num_images_per_prompt):
            global_idx = pi * args.num_images_per_prompt + img_i
            all_samples.append((global_idx, prompt))

    total_samples = len(all_samples)
    # Each rank gets a contiguous slice
    per_rank = (total_samples + world_size - 1) // world_size
    rank_start = rank * per_rank
    rank_end = min(rank_start + per_rank, total_samples)
    local_samples = all_samples[rank_start:rank_end]

    print_rank0(
        f"Backbone: {args.backbone}, Prompts: {len(prompts)}, "
        f"Images/prompt: {args.num_images_per_prompt}, Total: {total_samples}, "
        f"World size: {world_size}",
        rank,
    )
    print(f"[Rank {rank}] Processing samples {rank_start}..{rank_end} ({len(local_samples)} samples)")

    # --- Load diffusion pipeline ---
    # Multi-GPU: rank 0 loads from disk (Lustre) and caches to /dev/shm (tmpfs, in-RAM).
    # Other ranks load from tmpfs — zero network filesystem I/O, near-instant.
    from pid._src.inference.pipeline_registry import (
        decode_with_pipeline_vae,
        extract_latent,
        load_pipeline,
    )

    if world_size > 1:
        # Sequential load: rank 0 populates OS page cache from Lustre,
        # subsequent ranks read from warm cache — one at a time to avoid I/O contention.
        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                print(f"[Rank {rank}] Loading pipeline ({msg})...")
                pipeline, pipe_cfg = load_pipeline(
                    args.backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload
                )
            dist.barrier()
    else:
        print("Loading pipeline...")
        pipeline, pipe_cfg = load_pipeline(
            args.backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload
        )

    res = args.resolution or pipe_cfg.default_resolution[0]
    height, width = res, res
    num_inference_steps = args.num_inference_steps or pipe_cfg.default_num_inference_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else pipe_cfg.default_guidance_scale

    # Derive key names. Caption subdir is resolution-suffixed because `degrade_sigma`
    # depends on resolution (via FlowMatch scheduler's shift), so each run's sigma
    # must land in its own caption_{res} tar to avoid clobbering other-resolution runs.
    image_key = f"image_{res}"
    latent_key = f"{args.backbone}_latent_{res}"
    caption_key = f"caption_{res}"

    output_keys = [image_key, latent_key, caption_key]
    key_ext = {image_key: ".png", latent_key: ".pth", caption_key: ".json"}

    # Validate step indices after num_inference_steps is resolved.
    # --save_xt_steps K means "latent after K forward passes of the network", so K is
    # the count of completed steps. Valid K ∈ [1, num_inference_steps]. K=num_inference_steps
    # means the fully-denoised final latent (same as the clean dir — allowed for symmetry).
    if args.save_xt_steps:
        for s in args.save_xt_steps:
            if s < 1 or s > num_inference_steps:
                raise ValueError(f"--save_xt_steps value {s} out of range [1, {num_inference_steps}]")
    if args.save_x0_steps:
        if args.backbone != "flux":
            raise ValueError(f"--save_x0_steps only supports backbone=flux (got {args.backbone})")
        for s in args.save_x0_steps:
            if s < 1 or s > num_inference_steps:
                raise ValueError(f"--save_x0_steps value {s} out of range [1, {num_inference_steps}]")

    print_rank0(f"Resolution: {res}x{res}", rank)
    print_rank0(f"Output keys: {output_keys}", rank)

    # --- Phase 3: Generate and collect ---
    # Each rank writes to non-overlapping shard IDs (offset by rank * max_shards_per_rank).
    # Use a large gap (1000) so shard IDs don't collide even with many samples.
    max_shards_per_rank = (per_rank + args.max_samples_per_shard - 1) // args.max_samples_per_shard + 1
    shard_offset = rank * max_shards_per_rank
    print(f"[Rank {rank}] Phase 3: Generating {len(local_samples)} images (shards starting at {shard_offset})...")
    writer = ShardedTarWriter(
        args.output_dir, output_keys, key_ext, args.max_samples_per_shard, shard_offset=shard_offset
    )

    # Set up writers for intermediate xt steps (each step → separate output dir)
    save_xt_set = set(args.save_xt_steps) if args.save_xt_steps else set()
    xt_writers: dict[int, ShardedTarWriter] = {}
    xt_output_dirs: dict[int, str] = {}
    if args.save_xt_steps:
        for step_idx in args.save_xt_steps:
            step_dir = args.output_dir.rstrip("/") + f"-{step_idx}step_xt"
            xt_output_dirs[step_idx] = step_dir
            xt_writers[step_idx] = ShardedTarWriter(
                step_dir, output_keys, key_ext, args.max_samples_per_shard, shard_offset=shard_offset
            )
        print_rank0(f"Saving xt at steps {args.save_xt_steps}", rank)

    save_x0_set = set(args.save_x0_steps) if args.save_x0_steps else set()
    x0_writers: dict[int, ShardedTarWriter] = {}
    x0_output_dirs: dict[int, str] = {}
    if args.save_x0_steps:
        for step_idx in args.save_x0_steps:
            step_dir = args.output_dir.rstrip("/") + f"-{step_idx}step_x0"
            x0_output_dirs[step_idx] = step_dir
            x0_writers[step_idx] = ShardedTarWriter(
                step_dir, output_keys, key_ext, args.max_samples_per_shard, shard_offset=shard_offset
            )
        print_rank0(f"Saving x0_pred at steps {args.save_x0_steps}", rank)

    for li, (global_idx, prompt) in enumerate(local_samples):
        seed = args.seed + global_idx
        generator = torch.Generator(device="cuda").manual_seed(seed)
        sample_id = f"{global_idx:08d}"

        # Set up xt / x0 capture callbacks if needed
        xt_callback = XtCaptureCallback(save_xt_set) if save_xt_set else None
        x0_callback = X0CaptureCallback(save_x0_set, pipeline.transformer) if save_x0_set else None

        gen_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            output_type="latent",
            generator=generator,
        )
        gen_kwargs.update(pipe_cfg.extra_generate_kwargs)
        active_cbs = [cb for cb in (xt_callback, x0_callback) if cb is not None]
        if active_cbs:
            gen_kwargs["callback_on_step_end"] = (
                active_cbs[0] if len(active_cbs) == 1 else compose_callbacks(*active_cbs)
            )
            gen_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            raw_output = pipeline(**gen_kwargs)
        finally:
            # Always remove the transformer hook, even on error, to avoid leaks across samples.
            if x0_callback is not None:
                x0_callback.detach()
        latent = extract_latent(pipeline, raw_output, pipe_cfg, height, width)
        vae_image = decode_with_pipeline_vae(pipeline, latent, pipe_cfg)

        # Write intermediate xt samples captured by callback
        if xt_callback:
            for step_idx, (xt_raw_cpu, xt_sigma) in xt_callback.captured.items():
                xt_raw = xt_raw_cpu.to(device="cuda", dtype=dtype)
                # Reuse extract_latent to handle unpacking (Flux/Flux2/QwenImage packed formats)
                xt_latent = extract_latent(pipeline, SimpleNamespace(images=xt_raw), pipe_cfg, height, width)
                xt_image = decode_with_pipeline_vae(pipeline, xt_latent, pipe_cfg)

                step_data = {
                    image_key: image_tensor_to_png(xt_image[0]),
                    latent_key: tensor_to_bytes(xt_latent[0].to(torch.bfloat16).cpu().clone()),
                    caption_key: caption_to_bytes(prompt, sample_id, degrade_sigma=xt_sigma),
                }
                xt_writers[step_idx].add_sample(sample_id, step_data)

        # Write intermediate x0_pred samples captured by callback (Flux only)
        if x0_callback:
            for step_idx, (x0_raw_cpu, x0_sigma) in x0_callback.captured.items():
                x0_raw = x0_raw_cpu.to(device="cuda", dtype=dtype)
                # x0_pred is in packed Flux latent shape — extract_latent unpacks it.
                x0_latent = extract_latent(pipeline, SimpleNamespace(images=x0_raw), pipe_cfg, height, width)
                x0_image = decode_with_pipeline_vae(pipeline, x0_latent, pipe_cfg)

                step_data = {
                    image_key: image_tensor_to_png(x0_image[0]),
                    latent_key: tensor_to_bytes(x0_latent[0].to(torch.bfloat16).cpu().clone()),
                    caption_key: caption_to_bytes(prompt, sample_id, degrade_sigma=x0_sigma),
                }
                x0_writers[step_idx].add_sample(sample_id, step_data)

        # Write final sample (clean latent, sigma=0)
        sample_data = {
            image_key: image_tensor_to_png(vae_image[0]),
            latent_key: tensor_to_bytes(latent[0].to(torch.bfloat16).cpu().clone()),
            caption_key: caption_to_bytes(prompt, sample_id, degrade_sigma=0.0),
        }
        writer.add_sample(sample_id, sample_data)

        if (li + 1) % 10 == 0 or (li + 1) == len(local_samples):
            print(f"  [Rank {rank}] [{li + 1}/{len(local_samples)}] global_idx={global_idx}, seed={seed}")

    writer.close()
    for step_idx, w in xt_writers.items():
        w.close()
        print(f"[Rank {rank}] Wrote {w.total_written} xt samples for step {step_idx}")
    for step_idx, w in x0_writers.items():
        w.close()
        print(f"[Rank {rank}] Wrote {w.total_written} x0_pred samples for step {step_idx}")
    print(f"[Rank {rank}] Phase 3 done: wrote {writer.total_written} samples")

    # --- Phase 4: Barrier + rank 0 writes wdinfo.json ---
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        print("Phase 4: Generating wdinfo.json...")
        wdinfo_path = write_wdinfo(args.output_dir, output_keys, total_samples, args.max_samples_per_shard)

        # Write wdinfo for each intermediate xt step directory
        if args.save_xt_steps:
            for step_idx in args.save_xt_steps:
                step_dir = xt_output_dirs[step_idx]
                write_wdinfo(step_dir, output_keys, total_samples, args.max_samples_per_shard)

        # Write wdinfo for each intermediate x0_pred step directory
        if args.save_x0_steps:
            for step_idx in args.save_x0_steps:
                step_dir = x0_output_dirs[step_idx]
                write_wdinfo(step_dir, output_keys, total_samples, args.max_samples_per_shard)

        print(f"\nDataset created at: {args.output_dir}")
        print(f"  wdinfo: {wdinfo_path}")
        print(f"  Total samples: {total_samples}")
        print(f"  Keys: {output_keys}")
        if args.save_xt_steps:
            for step_idx in args.save_xt_steps:
                print(f"  xt step {step_idx}: {xt_output_dirs[step_idx]}")
        if args.save_x0_steps:
            for step_idx in args.save_x0_steps:
                print(f"  x0_pred step {step_idx}: {x0_output_dirs[step_idx]}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
