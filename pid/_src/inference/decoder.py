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
Shared "PiD pixel decoder" runtime for both demos.

  - load_our_decoder      : instantiate the PiD model from a checkpoint.
  - capture_steps         : iterate (step_label, latent, sigma) over captured xt + final x0 (from_ldm).
  - run_ours_and_save_step: run PiD once on a (latent, baseline) pair, save both, optionally upload.
  - vae_decode / add_noise: VAE round-trip + forward-noising for the from_clean path.
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch

from pid._src.inference.inference_utils import AsyncUploader, maybe_upload_video, save_image
from pid._src.inference.pipeline_registry import extract_latent
from pid._src.inference.step_capture import XtCaptureCallback
from pid._src.utils.model_loader import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def load_our_decoder(args, experiment_opts: list, is_rank0: bool):
    """Load the PiD pixel decoder (and its VAE) from the resolved checkpoint."""
    if is_rank0:
        logger.info(f"Loading our pixel decoder from {args.checkpoint_path} ...")
    model, _config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=args.load_ema_to_reg,
    )
    model.eval()
    if getattr(args, "compile", False):
        model.enable_compile()
    return model


def capture_steps(
    pipeline,
    pipe_cfg,
    xt_callback: Optional[XtCaptureCallback],
    final_latent: torch.Tensor,
    H: int,
    W: int,
    dtype: torch.dtype,
    ldm_inference_steps: int,
):
    """Yield (step_label, latent_unpacked_on_cuda, sigma) for each captured step.

    - Intermediate xt at user K (sorted ascending): label = f"{K:02d}xt", sigma from callback.
    - Final clean x0: label = "x0", sigma ≈ 0 (sigmas[-1] from scheduler).

    Uses extract_latent to unpack any backbone-specific packed format (Flux/Flux2/QwenImage).
    """
    if xt_callback is not None:
        for K in sorted(xt_callback.captured.keys()):
            xt_packed_cpu, sigma = xt_callback.captured[K]
            xt_packed = xt_packed_cpu.to(device="cuda", dtype=dtype)
            xt_latent = extract_latent(pipeline, SimpleNamespace(images=xt_packed), pipe_cfg, H, W)
            yield f"{K:02d}xt", xt_latent, sigma

    final_sigma = float(pipeline.scheduler.sigmas[-1].item())
    yield "x0", final_latent, final_sigma


def run_ours_and_save_step(
    *,
    model,
    args,
    tag: str,
    sample_id: str,
    prompt_idx: int,
    step_label: str,
    latent: torch.Tensor,
    baseline_01: torch.Tensor,  # (1, 3, H, W) in [0, 1]
    sigma: float,
    caption: str,
    output_dir: str,
    uploader: Optional[AsyncUploader],
    baseline_subdir: str,
    baseline_upload_tag_prefix: str,
):
    """Run our pixel decoder once on the captured latent + baseline image, save both,
    optionally upload to S3. Mirrors the per-step body of the diffusers run loop.

    `baseline_subdir` is the local-filesystem subdir name for the native baseline
    (e.g. "vae_decode" for diffusers, "dinov2_decode" for the RAE backbone,
    "siglip_decode" for the Scale-RAE backbone). `baseline_upload_tag_prefix` becomes
    "<prefix>_step_<label>" on S3.
    """
    # PiD conditions on LQ_latent + degrade_sigma + caption; baseline_neg1_1 is only the
    # native VAE/RAE baseline image we save alongside ours for comparison.
    baseline_neg1_1 = baseline_01 * 2.0 - 1.0  # [-1, 1] for saving the baseline image

    data_batch = {
        model.config.input_caption_key: [caption],
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([sigma], device="cuda", dtype=torch.float32),
    }

    lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
    infer_image_size = (lq_h * args.scale, lq_w * args.scale)

    samples = model.generate_samples_from_batch(
        data_batch,
        cfg_scale=args.cfg_scale,
        num_steps=args.pid_inference_steps,
        seed=args.seed,
        shift=args.shift,
        image_size=infer_image_size,
    )
    ours_img = samples[0].float().cpu().clamp(-1, 1)

    ours_path = os.path.join(output_dir, tag, f"step_{step_label}", f"{sample_id}.{args.save_format}")
    save_image(ours_img, ours_path)

    baseline_path = os.path.join(output_dir, baseline_subdir, f"step_{step_label}", f"{sample_id}.{args.save_format}")
    save_image(baseline_neg1_1.float().cpu().squeeze(0).clamp(-1, 1), baseline_path)

    logger.info(f"[{prompt_idx}] step={step_label} sigma={sigma:.4f} -> ours={ours_path}  baseline={baseline_path}")

    if args.upload:
        ours_upload_tag = f"{tag}_step_{step_label}"
        baseline_upload_tag = f"{baseline_upload_tag_prefix}_step_{step_label}"
        if uploader is not None:
            uploader.submit(maybe_upload_video, ours_path, ours_upload_tag, True, args.group_name)
            uploader.submit(maybe_upload_video, baseline_path, baseline_upload_tag, True, args.group_name)
        else:
            maybe_upload_video(ours_path, ours_upload_tag, True, args.group_name)
            maybe_upload_video(baseline_path, baseline_upload_tag, True, args.group_name)


def vae_decode(model, latent_4d: torch.Tensor) -> torch.Tensor:
    """Wrap model.vae_encoder.decode to handle the 5D <-> 4D shape contract.

    Input  latent_4d: [B, C, zH, zW]
    Output recon:     [B, 3, H, W] in [-1, 1]
    """
    z5 = latent_4d.unsqueeze(2)  # [B, C, 1, zH, zW]
    recon5 = model.vae_encoder.decode(z5)  # [B, 3, 1, H, W]
    if recon5.ndim == 5:
        recon5 = recon5[:, :, 0]  # [B, 3, H, W]
    return recon5


def add_noise(
    clean_latent: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
    backbone_tag: str = "",
) -> torch.Tensor:
    """Forward-noise the clean latent at sigma. Form depends on backbone_tag and must
    match the latent_noising backbone the PiD student was trained with.

      backbone_tag == "sdxl":     x_t = sqrt(1 - σ²) x_0 + σ ε   (variance-preserving)
      otherwise (flow-matching):  x_t = (1 - σ)      x_0 + σ ε
    """
    if sigma <= 0.0:
        return clean_latent
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    if backbone_tag == "sdxl":
        mean_coef = float((max(0.0, 1.0 - sigma * sigma)) ** 0.5)
        return mean_coef * clean_latent + sigma * noise
    return (1.0 - sigma) * clean_latent + sigma * noise
