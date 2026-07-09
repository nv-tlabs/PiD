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

#!/usr/bin/env python
# Direct T2I-to-PiD-fix-batch asset generation.
#
# This intentionally bypasses the older two-stage path:
#   create_dataset.py -> webdataset tar shards -> prepare_pixeldit_sr_fix_batch.py
# and writes the final per-sample fix_batch_XXXX.pt files directly. The LQ image
# and LQ_latent come from the requested T2I backbone at --resolution. The HQ side
# is a zero tensor placeholder at 4x resolution because generated T2I samples do
# not have a real paired HQ target.

import argparse
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


def encode_tensor_as_png(tensor: torch.Tensor) -> bytes:
    """Encode [C, H, W] or [1, C, H, W] tensor in [-1, 1] to PNG bytes.

    Returns raw PNG bytes that can be stored in a .pt file.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # [1, C, H, W] -> [C, H, W]
    arr = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def init_distributed() -> tuple[int, int]:
    """Initialize torchrun if present; otherwise run as rank 0."""
    if "RANK" not in os.environ:
        return 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def print_rank0(msg: str, rank: int) -> None:
    if rank == 0:
        print(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PiD fix_batch assets directly from T2I backbone outputs")

    parser.add_argument(
        "--backbone",
        required=True,
        choices=["flux", "sdxl", "sd3", "flux2", "qwenimage", "qwenimage_2512", "zimage"],
    )
    parser.add_argument("--backbone_model_id", type=str, default=None, help="Override HF model ID")

    parser.add_argument("--prompts", nargs="+", type=str, default=None, help="Inline prompts")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with one prompt per line")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    parser.add_argument("--resolution", type=int, default=None, help="Generated LQ resolution")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--save_xt_steps",
        nargs="+",
        type=int,
        default=None,
        help="Save noisy xt fix batches after these 1-indexed denoising steps.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Asset root, e.g. /lustre/.../assets/pixel_diffusion_small_face_and_text/flux",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed; global sample index is added")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Use diffusers model CPU offload for large backbones.",
    )

    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("Must provide --prompts or --prompts_file")


class XtCaptureCallback:
    """Capture latents after the requested denoising step count.

    K in --save_xt_steps means "after K forward passes have completed". Diffusers
    invokes callback_on_step_end with zero-based step_index, so K maps to
    step_index == K - 1. The captured latent is converted to the same training
    frame as create_dataset.py, including SDXL Euler-to-VP conversion.
    """

    def __init__(self, save_ks: set[int], cfg):
        self.save_map = {k - 1: k for k in save_ks}
        self.cfg = cfg
        self.captured: dict[int, tuple[torch.Tensor, float]] = {}

    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        from pid._src.dataprep.fix_batch_generation.pipeline_registry import to_training_frame

        k = self.save_map.get(step_index)
        if k is not None:
            sigmas = pipe.scheduler.sigmas
            sigma_idx = min(step_index + 1, len(sigmas) - 1)
            sigma_val = float(sigmas[sigma_idx].item())
            latent_train, sigma_train = to_training_frame(callback_kwargs["latents"], sigma_val, self.cfg)
            self.captured[k] = (latent_train.detach().cpu(), sigma_train)
        return callback_kwargs


def make_samples(prompts: list[str], num_images_per_prompt: int) -> list[tuple[int, str]]:
    samples = []
    for prompt_idx, prompt in enumerate(prompts):
        for image_idx in range(num_images_per_prompt):
            global_idx = prompt_idx * num_images_per_prompt + image_idx
            samples.append((global_idx, prompt))
    return samples


def local_slice(samples: list[tuple[int, str]], rank: int, world_size: int) -> list[tuple[int, str]]:
    per_rank = (len(samples) + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, len(samples))
    return samples[start:end]


def output_subdir(output_dir: str, step: int | None, hq_resolution: int) -> str:
    step_name = "full_step" if step is None else f"{step}step"
    return os.path.join(output_dir, step_name, str(hq_resolution))


def ensure_output_dirs(output_dir: str, hq_resolution: int, steps: list[int]) -> None:
    os.makedirs(output_subdir(output_dir, None, hq_resolution), exist_ok=True)
    for step in steps:
        os.makedirs(output_subdir(output_dir, step, hq_resolution), exist_ok=True)


def encode_zero_placeholder_png(resolution: int) -> bytes:
    """Encode the fix-batch tensor value 0.0 as PNG without allocating a huge tensor."""
    # encode_tensor_as_png(torch.zeros(...)) maps [-1, 1] value 0.0 to uint8 127.
    image = Image.new("RGB", (resolution, resolution), (127, 127, 127))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def save_fix_batch(
    *,
    output_dir: str,
    sample_idx: int,
    hq_png: bytes,
    lq_image_01: torch.Tensor,
    lq_latent: torch.Tensor,
    prompt: str,
    degrade_sigma: float,
) -> str:
    lq_image = lq_image_01[0].float() * 2.0 - 1.0
    fix_batch = {
        "HQ_video_or_image": hq_png,
        "caption": [prompt],
        "LQ_video_or_image": encode_tensor_as_png(lq_image.cpu()),
        "degrade_sigma": float(degrade_sigma),
        "LQ_latent": lq_latent[0].to(torch.bfloat16).cpu().clone(),
    }

    pt_path = os.path.join(output_dir, f"fix_batch_{sample_idx:04d}.pt")
    torch.save(fix_batch, pt_path)
    return pt_path


def validate_steps(save_xt_steps: list[int], num_inference_steps: int) -> None:
    for step in save_xt_steps:
        if step < 1 or step > num_inference_steps:
            raise ValueError(f"--save_xt_steps value {step} out of range [1, {num_inference_steps}]")


def main() -> None:
    rank, world_size = init_distributed()
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args)
    all_samples = make_samples(prompts, args.num_images_per_prompt)
    local_samples = local_slice(all_samples, rank, world_size)

    from pid._src.dataprep.fix_batch_generation.pipeline_registry import (
        decode_with_pipeline_vae,
        extract_latent,
        load_pipeline,
    )

    if world_size > 1:
        for load_rank in range(world_size):
            if rank == load_rank:
                print(f"[Rank {rank}] Loading pipeline...")
                pipeline, pipe_cfg = load_pipeline(
                    args.backbone, args.backbone_model_id, dtype=dtype, device=device, cpu_offload=args.cpu_offload
                )
            dist.barrier()
    else:
        print("Loading pipeline...")
        pipeline, pipe_cfg = load_pipeline(
            args.backbone, args.backbone_model_id, dtype=dtype, device=device, cpu_offload=args.cpu_offload
        )

    resolution = args.resolution or pipe_cfg.default_resolution[0]
    hq_resolution = resolution * 4
    height = width = resolution
    num_inference_steps = args.num_inference_steps or pipe_cfg.default_num_inference_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else pipe_cfg.default_guidance_scale
    save_xt_steps = sorted(set(args.save_xt_steps or []))
    validate_steps(save_xt_steps, num_inference_steps)

    ensure_output_dirs(args.output_dir, hq_resolution, save_xt_steps)
    hq_png = encode_zero_placeholder_png(hq_resolution)

    print_rank0(
        f"Backbone={args.backbone}, prompts={len(prompts)}, samples={len(all_samples)}, "
        f"resolution={resolution}, hq_placeholder={hq_resolution}, steps={save_xt_steps or 'none'}",
        rank,
    )
    print(f"[Rank {rank}] Processing {len(local_samples)} samples")

    for local_idx, (global_idx, prompt) in enumerate(local_samples, start=1):
        seed = args.seed + global_idx
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        xt_callback = XtCaptureCallback(set(save_xt_steps), pipe_cfg) if save_xt_steps else None

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
        if xt_callback is not None:
            gen_kwargs["callback_on_step_end"] = xt_callback
            gen_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        with torch.no_grad():
            raw_output = pipeline(**gen_kwargs)
            clean_latent = extract_latent(pipeline, raw_output, pipe_cfg, height, width)
            clean_image = decode_with_pipeline_vae(pipeline, clean_latent, pipe_cfg)

            final_dir = output_subdir(args.output_dir, None, hq_resolution)
            save_fix_batch(
                output_dir=final_dir,
                sample_idx=global_idx,
                hq_png=hq_png,
                lq_image_01=clean_image,
                lq_latent=clean_latent,
                prompt=prompt,
                degrade_sigma=0.0,
            )

            if xt_callback is not None:
                for step in save_xt_steps:
                    if step not in xt_callback.captured:
                        raise RuntimeError(f"Did not capture requested xt step {step} for sample {global_idx}")
                    xt_raw_cpu, xt_sigma = xt_callback.captured[step]
                    xt_raw = xt_raw_cpu.to(device=device, dtype=dtype)
                    xt_latent = extract_latent(pipeline, SimpleNamespace(images=xt_raw), pipe_cfg, height, width)
                    xt_image = decode_with_pipeline_vae(pipeline, xt_latent, pipe_cfg)
                    step_dir = output_subdir(args.output_dir, step, hq_resolution)
                    save_fix_batch(
                        output_dir=step_dir,
                        sample_idx=global_idx,
                        hq_png=hq_png,
                        lq_image_01=xt_image,
                        lq_latent=xt_latent,
                        prompt=prompt,
                        degrade_sigma=xt_sigma,
                    )

        if local_idx % 10 == 0 or local_idx == len(local_samples):
            print(f"[Rank {rank}] [{local_idx}/{len(local_samples)}] global_idx={global_idx}, seed={seed}")

    if world_size > 1:
        dist.barrier()

    print_rank0(f"Done. Wrote fix_batch assets under {Path(args.output_dir).absolute()}", rank)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
