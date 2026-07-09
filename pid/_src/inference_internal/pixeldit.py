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
PixelDiT T2I Inference — text-to-image generation using our DCP checkpoint format.

Unlike pixeldit_official.py (which loads official .pth weights), this script uses
load_model_from_checkpoint with DCP format checkpoints from our training runs.

Multi-GPU (torchrun): rank i processes prompts[i::world_size], each rank saves its
own output PNGs. No cross-rank communication is needed.

Usage (single prompt):
  PYTHONPATH=. python pid/_src/inference_internal/pixeldit.py \
      --experiment pixeldit_text_to_image_finetune_res_2048 \
      --checkpoint_path imaginaire4/imaginaire4-output/xxx/checkpoints/iter_000000000 \
      --prompt "A majestic snow-capped mountain range at golden hour" \
      --output_dir ./results/t2i --cfg_scale 2.75 --num_steps 50 --load_ema_to_reg

Usage (prompts file, multi-GPU):
  PYTHONPATH=. torchrun --nproc_per_node=4 \
      pid/_src/inference_internal/pixeldit.py \
      --experiment pixeldit_text_to_image_finetune_res_2048_to_3840 \
      --checkpoint_path imaginaire4/imaginaire4-output/xxx/checkpoints/iter_000000000 \
      --prompts_file pid/_src/dataprep/prompts/prompts_example.txt \
      --output_dir ./results/t2i --cfg_scale 2.75 --num_steps 50 --image_size 3840 --load_ema_to_reg
"""

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image

from pid._src.inference.inference_utils import (
    AsyncUploader,
    generate_tag_from_checkpoint,
    get_rank_and_world_size,
    maybe_upload_video,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="PixelDiT T2I inference")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument(
        "--config_file",
        type=str,
        default="pid/_src/configs/pid_training/config.py",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Single prompt string")
    prompt_group.add_argument("--prompts_file", type=str, help="Text file with one prompt per line")

    parser.add_argument("--output_dir", type=str, default="./results/pixeldit_t2i")
    parser.add_argument("--cfg_scale", type=float, default=2.75)
    parser.add_argument("--num_steps", type=int, default=None, help="Denoising steps (default: model config)")
    parser.add_argument("--shift", type=float, default=None, help="Flow shift (default: model config)")
    parser.add_argument("--image_size", type=int, default=None, help="Output resolution (default: model config)")
    parser.add_argument("--batch_size", type=int, default=4, help="Prompts per forward pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_ema_to_reg", action="store_true", help="Load EMA weights into regular model")

    # S3 upload
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--group_name", type=str, default="pixeldit_t2i", help="S3 group name")
    parser.add_argument("--note", type=str, default="", help="Note appended to tag")

    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown
    return args


def _save_png(tensor: torch.Tensor, save_path: str):
    """Save [C, H, W] or [C, 1, H, W] tensor in [-1, 1] as PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)  # [C, 1, H, W] -> [C, H, W]
    arr = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = arr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(arr).save(save_path)


def main():
    args = parse_arguments()

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
    is_rank0 = rank == 0

    # Build prompt list
    if args.prompt is not None:
        all_prompts = [args.prompt]
    else:
        with open(args.prompts_file) as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    if is_rank0:
        logger.info(f"Total prompts: {len(all_prompts)}")

    # Shard prompts across ranks (interleaved so output indices are stable)
    my_indices = list(range(rank, len(all_prompts), world_size))
    my_prompts = [all_prompts[i] for i in my_indices]
    if world_size > 1:
        logger.info(f"[Rank {rank}/{world_size}] {len(my_prompts)} prompts")

    # Load model
    if is_rank0:
        logger.info(f"Loading model: {args.checkpoint_path}")
    experiment_opts = list(args.extra_experiment_opts or [])
    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=args.load_ema_to_reg,
    )

    model.eval()

    # Build tag (used for output dir name and S3 upload key)
    extra_params = {"cfg": args.cfg_scale}
    if args.num_steps is not None:
        extra_params["steps"] = args.num_steps
    if args.shift is not None:
        extra_params["shift"] = args.shift
    if args.image_size is not None:
        extra_params["res"] = args.image_size
    if args.note:
        extra_params["note_"] = args.note
    tag = generate_tag_from_checkpoint(args.checkpoint_path, extra_params, load_ema=args.load_ema_to_reg)

    output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Tag: {tag}")
        logger.info(f"Output dir: {output_dir}")

    # Async uploader — upload calls run in background threads
    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # Generate in mini-batches
    bs = args.batch_size
    for i in range(0, len(my_prompts), bs):
        batch_prompts = my_prompts[i : i + bs]
        batch_indices = my_indices[i : i + bs]

        data_batch = {model.config.input_caption_key: batch_prompts}
        samples = model.generate_samples_from_batch(
            data_batch,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            shift=args.shift,
            image_size=args.image_size,
        )
        # samples: [B, C, 1, H, W]

        for sample, idx in zip(samples, batch_indices):
            save_path = os.path.join(output_dir, f"{idx:04d}.png")
            _save_png(sample.cpu(), save_path)
            logger.info(f"[Rank {rank}] {save_path}")

            if args.upload:
                uploader.submit(maybe_upload_video, save_path, tag, args.upload, args.group_name)

    # Wait for all background uploads before exiting
    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete...")
        uploader.wait()

    if is_rank0:
        logger.info("Done.")


if __name__ == "__main__":
    main()
