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

"""PiD image super-resolution inference for teacher model or student model

Pipeline: LQ image -> script-side VAE encode -> caption + LQ latent + sigma
          -> PiD model -> HQ image

Supported input formats:
    1. Single image + caption:
       --input_path image.png --caption "description of the image"

    2. Fix-batch directory containing ``fix_batch_*.pt`` files:
       --fix_batch_dir assets/pid_callback_assets/qwenimage/full_step/2048
       The model consumes the saved latent; LQ/HQ images stay on the script side.

The experiment, checkpoint, and callback assets must belong to the same model
family (for example, QwenImage must be paired with QwenImage assets).

Both teacher and distilled student inference are supported. For the examples
below, use ``cfg_scale=5, num_steps=25`` for the teacher and
``cfg_scale=1, num_steps=4`` for the student. The CKPT can be distributed checkpoint.

Example1: Teacher model (single image + caption):

    EXP=pid_v1pt5_teacher_qwenimage_h1024_d4_fix_backbone_res_2048
    CKPT=/path/to/teacher/checkpoints/iter_XXXXXXXXX

    PYTHONPATH=. python -m pid._src.inference_internal.pid_inference \
        --experiment "$EXP" \
        --checkpoint_path "$CKPT" \
        --input_path assets/0072.jpg \
        --caption "A tranquil alpine lakeside scene unfolds where a gravel path winds toward a serene, emerald-green lake framed by towering, forested mountains under a soft, overcast sky" \
        --output_dir results/pid_inference \
        --cfg_scale 5 --num_steps 25

Example2: Teacher model (fix-batch directory):

    PYTHONPATH=. python -m pid._src.inference_internal.pid_inference \
        --experiment "$EXP" \
        --checkpoint_path "$CKPT" \
        --fix_batch_dir assets/pid_callback_assets/qwenimage/full_step/2048 \
        --output_dir results/pid_inference \
        --cfg_scale 5 --num_steps 25

Example3: Student model (fix-batch directory with data parallelism):

    EXP=pid_v1pt5_student_qwenimage_h1024_d4_res_2048_distill
    CKPT=/path/to/student/checkpoints/iter_XXXXXXXXX

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        -m pid._src.inference_internal.pid_inference \
        --experiment "$EXP" \
        --checkpoint_path "$CKPT" \
        --fix_batch_dir assets/pid_callback_assets/qwenimage/full_step/2048 \
        --output_dir results/pid_inference \
        --cfg_scale 1 --num_steps 4
"""

import argparse
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pid._src.inference.inference_utils import (
    AsyncUploader,
    generate_tag_from_checkpoint,
    get_rank_and_world_size,
    load_fix_batch,
    maybe_upload_video,
)
from pid._src.utils.model_loader import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


# =============================================================================
# Argument parsing
# =============================================================================


def parse_arguments():
    parser = argparse.ArgumentParser(description="PiD image super-resolution inference")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment registered under pid/_src/configs/pid_training",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="pid/_src/configs/pid_training/config.py",
        help="Hydra config file",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_path", type=str, help="Path to a single LQ image")
    input_group.add_argument(
        "--fix_batch_dir",
        type=str,
        help="Directory of fix_batch .pt files, typically under assets/pid_callback_assets",
    )

    # Caption options
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Caption for single image (required with --input_path)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./results/pid_inference", help="Output directory")
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="Image format for saved SR / LQ / GT outputs (default: lossy JPEG)",
    )

    # Inference parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=1, help="CFG guidance scale")
    parser.add_argument("--shift", type=float, default=None, help="Flow shift (default: use model config)")
    parser.add_argument("--num_steps", type=int, default=None, help="Denoising steps (default: use model config)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (for fix_batch mode)")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap on total fix_batch .pt files processed across all ranks. "
        "Applied to the sorted file list before rank sharding, so the run is reproducible "
        "regardless of world size. Only effective in --fix_batch_dir mode.",
    )
    parser.add_argument(
        "--degrade_sigma",
        type=float,
        default=None,
        help="Explicit degradation-sigma override. When omitted, single-image mode uses 0.0 "
        "and fix-batch mode reads degrade_sigma from each .pt file. "
        "Any numeric value (including 0.0) overrides the data. "
        "Only effective when model was trained with a sigma-aware gate (lq_gate_type='sigmoid_sigma' or 'sigma_only').",
    )

    # Model options
    parser.add_argument("--load_ema_to_reg", action="store_true", help="Load EMA weights to regular model")
    # S3 upload
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--group_name", type=str, default="pid_inference", help="S3 group name")
    parser.add_argument("--note", type=str, default="", help="Note appended to tag")

    # Parse known args; unknown args forwarded to experiment config
    args, unknown_args = parser.parse_known_args()
    if args.input_path is not None and args.caption is None:
        parser.error("--caption is required when using --input_path")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be greater than 0")
    args.extra_experiment_opts = unknown_args
    return args


# =============================================================================
# Image I/O helpers
# =============================================================================


def load_image_as_tensor(path: str, dtype=torch.float32, device="cuda") -> torch.Tensor:
    """Load an image file as [1, 3, H, W] tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0  # [1, 3, H, W]
    return t.to(dtype)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] tensor in [-1, 1] to PIL Image."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def save_result_image(
    sample: torch.Tensor,
    save_path: str,
    quality: int = 95,
) -> str:
    """Save a single sample tensor [C, H, W] or [C, 1, H, W]. Format inferred from extension.

    JPEG paths (.jpg/.jpeg) are saved with the requested `quality`; PNG paths are lossless.
    """
    if sample.dim() == 4:
        sample = sample.squeeze(1)  # [C, 1, H, W] -> [C, H, W]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img = _tensor_to_pil(sample)
    if save_path.lower().endswith((".jpg", ".jpeg")):
        img.save(save_path, quality=quality)
    else:
        img.save(save_path)
    return save_path


def _is_placeholder_hq(image: torch.Tensor) -> bool:
    """Return whether HQ is a known empty callback placeholder.

    Current callback assets encode an empty HQ image as uniform uint8 value
    127, which decodes to ``-1 / 255`` rather than exactly zero. Older assets
    may contain an actual all-zero tensor.
    """
    image = image.float()
    if image.numel() == 0 or image.abs().max().item() == 0:
        return True
    encoded_zero = 127.0 / 127.5 - 1.0
    return bool(torch.allclose(image, torch.full_like(image, encoded_zero), rtol=0.0, atol=1e-6))


# =============================================================================
# Processing functions
# =============================================================================


def _generate_from_lq_latent(
    model,
    captions: List[str],
    lq_latents: torch.Tensor,
    degrade_sigmas: List[float],
    args,
) -> torch.Tensor:
    """Run PiD with the three-field latent-only inference batch."""
    if args.degrade_sigma is not None:
        degrade_sigmas = [float(args.degrade_sigma)] * len(captions)

    data_batch = {
        model.config.input_caption_key: captions,
        "LQ_latent": lq_latents.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor(degrade_sigmas, dtype=torch.float32, device="cuda"),
    }

    return model.generate_samples_from_batch(
        data_batch,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        seed=args.seed,
        shift=args.shift,
    )


def process_single_image(
    model,
    lq_image: torch.Tensor,
    caption: str,
    args,
    output_path: str,
    tag: str = "",
    uploader: Optional["AsyncUploader"] = None,
) -> str:
    """Run SR on a single LQ image and save results.

    Args:
        model: PiD model
        lq_image: [1, 3, H_lq, W_lq] in [-1, 1]
        caption: text caption
        args: CLI args
        output_path: save path for SR output
        tag: experiment tag for S3 upload

    Returns:
        Path to saved SR image.
    """
    # Encode the single LQ image here so both input modes send the same
    # latent-only batch (caption, LQ latent, degrade sigma) to PiD.
    lq_latent = model.encode_lq_latent(lq_image.to(dtype=torch.bfloat16, device="cuda")).contiguous()
    samples = _generate_from_lq_latent(
        model,
        captions=[caption],
        lq_latents=lq_latent,
        degrade_sigmas=[0.0],
        args=args,
    )

    # samples: [B, C, 1, H, W] -> save first sample
    sr_image = samples[0].float().cpu().clamp(-1, 1)
    save_result_image(sr_image, output_path)
    logger.info(f"Saved SR: {output_path}")

    # Save the bicubic-upsampled LQ input alongside the model output.
    basename = os.path.basename(output_path)
    sr_dir = os.path.dirname(output_path)
    parent_dir = os.path.dirname(sr_dir)

    # LQ input (bicubic upsampled to SR resolution)
    lq_dir = os.path.join(parent_dir, "LQ_input")
    os.makedirs(lq_dir, exist_ok=True)
    lq_save = lq_image[0].float().cpu()
    target_h, target_w = sr_image.shape[-2], sr_image.shape[-1]
    lq_up = F.interpolate(lq_save.unsqueeze(0), size=(target_h, target_w), mode="bicubic", align_corners=False)
    lq_path = os.path.join(lq_dir, basename)
    save_result_image(lq_up.squeeze(0).clamp(-1, 1), lq_path)

    # Upload to S3 (async if uploader provided, else blocking)
    if args.upload and tag:

        def _upload(path, upload_tag):
            if uploader is not None:
                uploader.submit(maybe_upload_video, path, upload_tag, args.upload, args.group_name)
            else:
                maybe_upload_video(path, upload_tag, args.upload, args.group_name)

        _upload(output_path, tag)
        _upload(lq_path, "LQ_input")

    return output_path


def process_fix_batches(
    model,
    fix_batch_paths: list,
    args,
    output_dir: str,
    tag: str = "",
    uploader: Optional["AsyncUploader"] = None,
):
    """Process multiple fix_batch .pt files as a single batch.

    Fix-batch inference requires pre-computed LQ latents. LQ and HQ pixels stay
    in this script for visualization and are never passed to the model.
    """
    records = []
    for pt_path in fix_batch_paths:
        fb = load_fix_batch(pt_path)
        cap = fb.get("caption")
        if cap is None:
            raise ValueError(f"Fix-batch file is missing caption: {pt_path}")
        # caption is stored as list[str] in fix_batch; flatten to str for batching
        if isinstance(cap, list):
            if not cap:
                raise ValueError(f"Fix-batch caption list is empty: {pt_path}")
            caption = cap[0]
        else:
            caption = cap
        lq_latent = fb["LQ_latent"]
        lq_image = fb["LQ_video_or_image"]
        degrade_sigma = fb["degrade_sigma"]
        m = re.search(r"(\d+)", os.path.basename(pt_path))
        records.append(
            {
                "LQ_video_or_image": lq_image,
                "HQ_video_or_image": fb.get("HQ_video_or_image"),
                "LQ_latent": lq_latent,
                "caption": caption,
                "index": int(m.group(1)) if m else 0,
                "degrade_sigma": float(degrade_sigma),
            }
        )
    _run_batch(model, records, args, output_dir, tag, uploader, name_width=4)


def _run_batch(
    model,
    records: List[Dict],
    args,
    output_dir: str,
    tag: str = "",
    uploader: Optional["AsyncUploader"] = None,
    name_width: int = 4,
):
    """Shared batched-inference path for fix-batch files.

    Each record is a dict with:
        LQ_latent: required tensor [1, C, zH, zW]
        LQ_video_or_image: required tensor [1, 3, H_lq, W_lq] for visualization
        caption: required str
        degrade_sigma: required float
        HQ_video_or_image: optional tensor used only for visualization
        index: required int | str; integers are zero-padded to `name_width`.
    """
    if not records:
        return

    captions = [r["caption"] for r in records]
    indices = [r["index"] for r in records]
    lq_images = torch.cat([r["LQ_video_or_image"] for r in records], dim=0)
    lq_latents = torch.cat([r["LQ_latent"] for r in records], dim=0)
    degrade_sigmas = [r["degrade_sigma"] for r in records]
    gt_images_per_record = [r.get("HQ_video_or_image") for r in records]
    B = lq_images.shape[0]
    if lq_latents.shape[0] != B:
        raise ValueError("LQ image and latent batch sizes do not match")

    samples = _generate_from_lq_latent(
        model,
        captions=captions,
        lq_latents=lq_latents,
        degrade_sigmas=degrade_sigmas,
        args=args,
    )
    # samples: [B, C, 1, H, W]

    parent_dir = os.path.dirname(output_dir)
    lq_dir = os.path.join(parent_dir, "LQ_input")
    gt_dir = os.path.join(parent_dir, "GT")

    sr_images = samples.float().cpu().clamp(-1, 1)
    target_h, target_w = sr_images.shape[-2], sr_images.shape[-1]
    lq_up = F.interpolate(lq_images.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)

    for i in range(B):
        idx_i = indices[i]
        stem_i = f"{idx_i:0{name_width}d}" if isinstance(idx_i, int) else str(idx_i)
        basename = f"{stem_i}.{args.save_format}"
        sr_image = sr_images[i]  # [C, 1, H, W]

        output_path = os.path.join(output_dir, basename)
        save_result_image(sr_image, output_path)

        os.makedirs(lq_dir, exist_ok=True)
        lq_path = os.path.join(lq_dir, basename)
        save_result_image(lq_up[i].clamp(-1, 1), lq_path)

        gt_path = None
        gt_image = gt_images_per_record[i]
        if isinstance(gt_image, torch.Tensor) and not _is_placeholder_hq(gt_image):
            os.makedirs(gt_dir, exist_ok=True)
            gt_path = os.path.join(gt_dir, basename)
            save_result_image(gt_image[0].float().clamp(-1, 1), gt_path)

        if args.upload and tag:

            def _upload(path, upload_tag):
                if uploader is not None:
                    uploader.submit(maybe_upload_video, path, upload_tag, args.upload, args.group_name)
                else:
                    maybe_upload_video(path, upload_tag, args.upload, args.group_name)

            _upload(output_path, tag)
            _upload(lq_path, "LQ_input")
            if gt_path is not None:
                _upload(gt_path, "GT")

    def _fmt_idx(x):
        return f"{x:0{name_width}d}" if isinstance(x, int) else str(x)

    logger.info(f"Saved batch of {B}: indices {_fmt_idx(indices[0])}-{_fmt_idx(indices[-1])}")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_arguments()

    if args.input_path is not None and not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input image does not exist: {args.input_path}")
    if args.fix_batch_dir is not None and not os.path.isdir(args.fix_batch_dir):
        raise FileNotFoundError(f"Fix-batch directory does not exist: {args.fix_batch_dir}")

    # Initialize distributed (for data parallel with fix_batch_dir)
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
    is_rank0 = rank == 0

    # Generate tag
    extra_params = {"cfg": args.cfg_scale}
    if args.num_steps is not None:
        extra_params["steps"] = args.num_steps
    if args.shift is not None:
        extra_params["shift"] = args.shift
    if args.degrade_sigma is not None:
        extra_params["sigma"] = args.degrade_sigma
    if args.note:
        extra_params["note_"] = args.note
    tag = generate_tag_from_checkpoint(args.checkpoint_path, extra_params, load_ema=args.load_ema_to_reg)

    if is_rank0:
        logger.info(f"Tag: {tag}")

    # Build experiment options
    experiment_opts = []
    if args.extra_experiment_opts:
        experiment_opts.extend(args.extra_experiment_opts)
        if is_rank0:
            logger.info(f"Extra experiment options: {args.extra_experiment_opts}")

    # Load model
    if is_rank0:
        logger.info(f"Loading model from {args.checkpoint_path} ...")

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        enable_fsdp=False,
        experiment_opts=experiment_opts,
        strict=False,
        load_ema_to_reg=args.load_ema_to_reg,
    )

    model.eval()

    # Create tagged output directory
    output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    # Async uploader — uploads run in background threads so inference is not blocked
    uploader = AsyncUploader(max_workers=8) if args.upload else None

    # ---- Mode 1: Single image + caption ----
    if args.input_path is not None:
        lq_image = load_image_as_tensor(args.input_path, dtype=torch.float32, device="cuda")
        basename = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_sr.{args.save_format}")

        process_single_image(model, lq_image, args.caption, args, output_path, tag=tag, uploader=uploader)

    # ---- Mode 2: Fix batch directory ----
    elif args.fix_batch_dir is not None:
        pt_files = sorted(
            [os.path.join(args.fix_batch_dir, f) for f in os.listdir(args.fix_batch_dir) if f.endswith(".pt")]
        )
        if not pt_files:
            raise ValueError(f"No .pt files found in {args.fix_batch_dir}")

        # Cap before sharding so the selected subset is identical across world sizes.
        if args.max_samples is not None and len(pt_files) > args.max_samples:
            if is_rank0:
                logger.info(f"Capping fix_batch list from {len(pt_files)} to {args.max_samples} (--max_samples)")
            pt_files = pt_files[: args.max_samples]

        # Data parallel sharding
        if world_size > 1:
            pt_files = pt_files[rank::world_size]
            logger.info(f"[Rank {rank}/{world_size}] Processing {len(pt_files)} fix_batch files")

        # Process in batches
        bs = args.batch_size
        for i in range(0, len(pt_files), bs):
            batch_paths = pt_files[i : i + bs]
            logger.info(f"[{i + 1}-{i + len(batch_paths)}/{len(pt_files)}]")
            process_fix_batches(model, batch_paths, args, output_dir, tag=tag, uploader=uploader)

    # Wait for all background uploads to finish before exiting
    if uploader is not None:
        logger.info("Waiting for background uploads to complete...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
