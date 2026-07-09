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

"""Official demo: text/class → latent diffusion → PiD pixel-decode.

Single entrypoint for every LDM backbone — pick one with `--backbone`:

    diffusers backbones : flux, flux2, flux2-klein-4b, flux2-klein-9b, sd3, sdxl,
                          qwenimage, qwenimage-2512, zimage, zimage-turbo
    representation AEs  : dinov2, siglip

Runs the backbone on a prompt (or class id for `dinov2`), captures the intermediate
`x_t` at `--save_xt_steps` (early termination) and the final clean `x_0`, then decodes
each captured latent twice — once with the backbone's native VAE/RAE decoder (baseline)
and once with our PiD pixel-diffusion decoder. Outputs are saved side-by-side and
(optionally) async-uploaded to S3.

Per-backbone defaults (resolution, steps, guidance, extra generate kwargs) come from
`pipeline_registry.py`. `--experiment` / `--checkpoint_path` default to the
`checkpoint_registry.py` entry for `(backbone, --pid_ckpt_type)`.

SDXL note: SDXL is the only non-flow-matching backbone. Its captured `x_t` is rescaled
from the variance-exploding Euler frame to the VP frame the PiD-SDXL student trained on
(see `pipeline_registry.to_training_frame`); from the CLI this is transparent.

Qwen note: `--backbone qwenimage-2512` runs the Dec-2025 Qwen-Image refresh (same VAE +
PiD student as `qwenimage`, different transformer). Large models — pass `--cpu_offload`
on single-GPU runs.

Flux2-klein note: `--backbone flux2-klein-4b` / `flux2-klein-9b` run the FLUX.2-klein
distilled models (Flux2KleinPipeline / FLUX.2-klein-4B | -9B) — same Flux2 BN VAE + PiD
student as `flux2`, different transformer. Defaults follow the model cards: 4 steps,
guidance_scale=1.0.

>>> Single GPU, single prompt (Flux, default 2k decoder):
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone flux \
    --prompt "a photorealistic tabby cat on a wooden table, cinematic light" \
    --ldm_inference_steps 28 --save_xt_steps 24 \
    --output_dir ./results/official_demo/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Single GPU, 1024 → 4K (SDXL, 2kto4k decoder):  (--resolution is the final 4K size; LDM runs at 4096/4 = 1024)
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone sdxl \
    --pid_ckpt_type 2kto4k --load_ema_to_reg --resolution 4096 \
    --prompt "an oil painting of a forest at dusk" \
    --ldm_inference_steps 30 --save_xt_steps 24 26 28 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Multi-GPU, prompt file (Qwen-Image-2512):
PYTHONPATH=. torchrun --nproc_per_node=4 -m pid._src.inference.from_ldm \
    --backbone qwenimage-2512 --pid_ckpt_type 2kto4k_v1pt5 --load_ema_to_reg --cpu_offload \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 50 --save_xt_steps 44 46 48 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

(`dinov2` / `siglip` need the upstream RAE / Scale-RAE repos — see docs/dinov2_siglip.md.)
"""

import logging
import os

import torch

from pid._src.inference.cli_utils import maybe_init_distributed, parse_ldm_args
from pid._src.inference.decoder import (
    capture_steps,
    load_our_decoder,
    run_ours_and_save_step,
)
from pid._src.inference.inference_utils import (
    AsyncUploader,
    build_tag,
    get_rank_and_world_size,
    load_prompts,
)
from pid._src.inference.pipeline_registry import (
    PIPELINE_REGISTRY,
    decode_with_pipeline_vae,
    extract_latent,
    load_pipeline,
)
from pid._src.inference.step_capture import XtCaptureCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


def run_ldm_demo(args):
    """Diffusers-backbone demo flow (flux / flux2 / flux2-klein* / sd3 / sdxl / qwenimage* / zimage*)."""
    backbone = args.backbone
    rank, world_size = get_rank_and_world_size()
    maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    # ---- Resolve backbone defaults ----
    pipe_cfg_default = PIPELINE_REGISTRY[backbone]
    # --resolution is the FINAL (post-SR) output resolution, parsed into (H, W); non-square
    # supported. The LDM runs at output / --scale, so the PiD decoder's (baseline * scale)
    # lands back on the requested output size.
    H_out, W_out = args.resolution
    if H_out % args.scale != 0 or W_out % args.scale != 0:
        raise ValueError(
            f"--resolution {H_out}x{W_out} must be divisible by --scale {args.scale} "
            f"(the LDM runs at --resolution / --scale)."
        )
    H, W = H_out // args.scale, W_out // args.scale
    ldm_inference_steps = args.ldm_inference_steps or pipe_cfg_default.default_num_inference_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else pipe_cfg_default.default_guidance_scale
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Validate xt step indices against the resolved ldm_inference_steps.
    save_xt_set = set(args.save_xt_steps) if args.save_xt_steps else set()
    for k in save_xt_set:
        if k < 1 or k > ldm_inference_steps:
            raise ValueError(f"--save_xt_steps value {k} out of range [1, {ldm_inference_steps}]")

    prompts = load_prompts(args)
    tag = build_tag(args, backbone)

    if is_rank0:
        logger.info(
            f"Backbone: {backbone}  output(H={H_out}, W={W_out})  ldm(H={H}, W={W})  "
            f"ldm_steps: {ldm_inference_steps}  guidance: {guidance_scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Prompts: {len(prompts)}  save_xt_steps: {sorted(save_xt_set)}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load HF pipeline (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading {backbone} pipeline ({msg}) ...")
                pipeline, pipe_cfg = load_pipeline(
                    backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload
                )
            dist.barrier()
    else:
        logger.info(f"Loading {backbone} pipeline ...")
        pipeline, pipe_cfg = load_pipeline(backbone, args.backbone_model_id, dtype=dtype, cpu_offload=args.cpu_offload)

    # ---- Load our pixel decoder ----
    model = load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or f"./results/official_demo/{backbone}"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # ---- Shard prompts across ranks (round-robin keeps load balanced when len%world!=0) ----
    indexed_prompts = list(enumerate(prompts))
    if world_size > 1:
        indexed_prompts = indexed_prompts[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_prompts)} prompts")

    for prompt_idx, prompt in indexed_prompts:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        generator = torch.Generator(device="cuda").manual_seed(seed)

        xt_cb = XtCaptureCallback(save_xt_set, pipe_cfg) if save_xt_set else None

        gen_kwargs = dict(
            prompt=prompt,
            height=H,
            width=W,
            num_inference_steps=ldm_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            output_type="latent",
            generator=generator,
        )
        gen_kwargs.update(pipe_cfg.extra_generate_kwargs)
        if xt_cb is not None:
            gen_kwargs["callback_on_step_end"] = xt_cb
            gen_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        logger.info(f"[{prompt_idx}] Running {backbone} pipeline (seed={seed}): {prompt[:80]!r}")
        raw_output = pipeline(**gen_kwargs)
        final_latent = extract_latent(pipeline, raw_output, pipe_cfg, H, W)

        # ---- Decode each step (intermediate xt + final x0) ----
        for step_label, latent, sigma in capture_steps(
            pipeline, pipe_cfg, xt_cb, final_latent, H, W, dtype, ldm_inference_steps
        ):
            # VAE decode (baseline) — returns (1, 3, H, W) in [0, 1]
            with torch.no_grad():
                vae_img_01 = decode_with_pipeline_vae(pipeline, latent, pipe_cfg)

            run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=vae_img_01,
                sigma=sigma,
                caption=prompt,
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="vae_decode",
                baseline_upload_tag_prefix=f"{backbone}_vae_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


def main():
    args = parse_ldm_args()
    # Non-diffusers backbones bypass the diffusers pipeline (own load + capture + decode).
    if args.backbone == "dinov2":
        from pid._src.inference.rae_generation import run_rae_demo

        return run_rae_demo(args)
    if args.backbone == "siglip":
        from pid._src.inference.scale_rae_generation import run_scale_rae_demo

        return run_scale_rae_demo(args)
    return run_ldm_demo(args)


if __name__ == "__main__":
    main()
