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

"""From-clean demo: input image → VAE encode → optional noise → PiD pixel-decode.

Single entrypoint for every VAE backbone — pick one with `--backbone`:

    flux, flux2, sd3, sdxl, qwenimage, dinov2, siglip

No latent diffusion model is run. The input image is fed at its native resolution (only
center-cropped so each side is a multiple of 16), VAE-encoded via the loaded pixel-decoder
model's own VAE, optionally forward-noised by each σ in `--degrade_sigmas`, then decoded
twice (VAE baseline + PiD) at `--scale * vae_native_resolution`.

Noise form follows the backbone the PiD student trained with: `sdxl` uses the
variance-preserving `x_t = sqrt(1-σ²) x_0 + σ ε`; all others use flow-matching
`x_t = (1-σ) x_0 + σ ε` (see `decoder.add_noise`).

`--experiment` / `--checkpoint_path` default to the `checkpoint_registry.py` entry for
`(backbone, --pid_ckpt_type)`.

>>> Single image, sigma sweep (Flux):
PYTHONPATH=. python -m pid._src.inference.from_clean --backbone flux \
    --manifest assets/clean_image_manifest.jsonl \
    --degrade_sigmas 0.0 0.4 \
    --output_dir ./results/official_demo_from_clean/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4

>>> Single image with explicit prompt (SDXL, VP noising):
PYTHONPATH=. python -m pid._src.inference.from_clean --backbone sdxl \
    --pid_ckpt_type 2kto4k --load_ema_to_reg \
    --input_path some.jpg --prompt "a cat" --degrade_sigmas 0.0 0.4 0.7 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

import logging
import os

import torch

from pid._src.inference.cli_utils import parse_clean_args
from pid._src.inference.decoder import add_noise, load_our_decoder, vae_decode
from pid._src.inference.inference_utils import (
    AsyncUploader,
    build_tag,
    get_rank_and_world_size,
    load_input_image,
    load_samples,
    maybe_upload_video,
    save_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


def run_clean_demo(args):
    backbone_tag = args.backbone
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
    is_rank0 = rank == 0

    tag = build_tag(args, backbone_tag)
    if is_rank0:
        logger.info(
            f"Backbone(VAE): {backbone_tag}  input: native resolution (cropped to 16-multiple)  "
            f"sigmas: {sorted(args.degrade_sigmas)}  scale: {args.scale}  pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    # ---- Load model (also loads its VAE) ----
    model = load_our_decoder(args, experiment_opts, is_rank0)

    # ---- Resolve default text prompt.
    # PiD requires a caption in the data batch. Order of fallback applied per sample:
    #   1) per-sample "prompt" from --manifest line  (manifest mode only)
    #   2) --prompt CLI flag                          (global default)
    #   3) ValueError                                 (no caption resolvable)

    # ---- Output dirs / uploader ----
    output_dir = args.output_dir or f"./results/official_demo_from_clean/{backbone_tag}"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")
    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # ---- Resolve sample list (single image OR JSONL manifest) and shard across ranks ----
    samples_all = load_samples(args)
    indexed_samples = list(enumerate(samples_all))
    if world_size > 1:
        my_samples = indexed_samples[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(my_samples)} of {len(samples_all)} samples")
    else:
        my_samples = indexed_samples
        if is_rank0:
            logger.info(f"Processing {len(my_samples)} sample(s)")

    for idx, (image_path, per_sample_prompt) in my_samples:
        # ---- Resolve caption for this sample ----
        caption = per_sample_prompt or args.prompt
        if caption is None:
            raise ValueError(
                f"Sample idx={idx} image={image_path!r} has no resolvable caption — "
                f"provide a per-line 'prompt' in the manifest, --prompt."
            )

        # ---- Filename layout: under --manifest disambiguate with idx prefix; under
        # --input_path keep the bare basename (preserves the prior single-image UX).
        bn = os.path.splitext(os.path.basename(image_path))[0]
        sample_id = f"{idx:08d}_{bn}" if args.manifest is not None else bn

        # ---- Load + encode ----
        # The image is fed at its native resolution (cropped to a 16-multiple). dinov2 / siglip
        # resize internally to their own fixed native interface, so the same preprocessing works
        # for every backbone.
        input_tensor = load_input_image(image_path).to(dtype=torch.bfloat16, device="cuda")
        clean_latent = model.encode_lq_latent(input_tensor)  # [1, C, zH, zW]

        # ---- Derive VAE-native pixel size from the latent grid times the tokenizer's
        # spatial_compression_factor. For standard VAEs (Flux/SD3/Flux2/SDXL/QwenImage) it equals
        # the (16-multiple) input size; for RAE-style fixed-resolution decoders (Scale-RAE → 256,
        # DINOv2-RAE → 512) it is independent of the input size. We anchor target_hw to this
        # VAE-native size so the LQ image fed to the pixel decoder and the SR output stay
        # consistent with the model's training-time scale = SR_out / vae_native.
        vae_compression = int(model.vae_encoder.spatial_compression_factor)
        vae_h = int(clean_latent.shape[-2]) * vae_compression
        vae_w = int(clean_latent.shape[-1]) * vae_compression
        target_hw = (vae_h * args.scale, vae_w * args.scale)
        logger.info(
            f"[idx={idx}] Clean latent shape={tuple(clean_latent.shape)}  "
            f"vae_native=({vae_h}x{vae_w})  target_hw={target_hw}  caption={caption[:60]!r}"
        )

        # ---- Save the input itself at its NATIVE resolution (no upsample). ----
        input_save = input_tensor.float().cpu().squeeze(0).clamp(-1, 1)
        input_path_out = os.path.join(output_dir, "input", f"{sample_id}.{args.save_format}")
        save_image(input_save, input_path_out)
        if args.upload:
            input_upload_tag = f"{backbone_tag}_input"
            if uploader is not None:
                uploader.submit(maybe_upload_video, input_path_out, input_upload_tag, True, args.group_name)
            else:
                maybe_upload_video(input_path_out, input_upload_tag, True, args.group_name)

        # ---- σ sweep ----
        for sigma in sorted(args.degrade_sigmas):
            sigma_label = f"sigma_{sigma:.3f}"

            # Per-σ deterministic noise generator (re-seeded so the same σ always gives the same noise)
            gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
            latent = add_noise(clean_latent.float(), float(sigma), gen, backbone_tag).to(dtype=torch.bfloat16)

            # VAE decode (baseline)
            with torch.no_grad():
                vae_img = vae_decode(model, latent)  # [1, 3, R, R] in [-1, 1]

            # Pixel decoder (ours). PiD conditions on LQ_latent + degrade_sigma + caption.
            data_batch = {
                model.config.input_caption_key: [caption],
                "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor([float(sigma)], device="cuda", dtype=torch.float32),
            }
            samples_out = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=args.cfg_scale,
                num_steps=args.pid_inference_steps,
                seed=args.seed + idx,
                shift=args.shift,
                image_size=target_hw,
            )
            ours_img = samples_out[0].float().cpu().clamp(-1, 1)  # [C, 1, H_out, W_out]

            # Save ours (native SR resolution)
            ours_path = os.path.join(output_dir, tag, sigma_label, f"{sample_id}.{args.save_format}")
            save_image(ours_img, ours_path)

            # Save VAE baseline at its native resolution (no bicubic upsampling).
            vae_path = os.path.join(output_dir, "vae_decode", sigma_label, f"{sample_id}.{args.save_format}")
            save_image(vae_img.float().cpu().squeeze(0).clamp(-1, 1), vae_path)

            logger.info(f"[idx={idx}] sigma={sigma:.3f} -> ours={ours_path}  vae={vae_path}")

            # S3 upload — flat one-level experiment_name expected by
            #   scripts/comparsion_display_presigned.py:
            #     s3://<bucket>/streamlit_assets/<group>/<experiment_name>/<filename>
            # so we collapse "<tag>/<sigma_label>" into "<tag>_<sigma_label>". The VAE
            # side uses "<backbone_tag>_vae_decode_<sigma_label>" to avoid cross-VAE
            # collisions when sharing --group_name.
            if args.upload:
                ours_upload_tag = f"{tag}_{sigma_label}"
                vae_upload_tag = f"{backbone_tag}_vae_decode_{sigma_label}"
                if uploader is not None:
                    uploader.submit(maybe_upload_video, ours_path, ours_upload_tag, True, args.group_name)
                    uploader.submit(maybe_upload_video, vae_path, vae_upload_tag, True, args.group_name)
                else:
                    maybe_upload_video(ours_path, ours_upload_tag, True, args.group_name)
                    maybe_upload_video(vae_path, vae_upload_tag, True, args.group_name)

    if uploader is not None:
        logger.info(f"[Rank {rank}] Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


def main():
    args = parse_clean_args()
    run_clean_demo(args)


if __name__ == "__main__":
    main()
