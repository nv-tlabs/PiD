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

"""Boogu-Image text-to-image inference.

By default this entrypoint runs Boogu's own decoder and saves the generated
images directly. With --pid_decode it also captures Boogu's final Flux1-style
VAE latent and routes it through the Flux1 PiD pixel decoder.

Example:
PYTHONPATH=. python -m pid._src.inference.from_boogu \
    --variant base \
    --model models/Boogu-Image-0.1-Base \
    --prompt "A cinematic street photograph of a neon-lit market at night" \
    --resolution 1024 \
    --output_dir results/boogu/base
"""

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pid._src.inference.checkpoint_registry import VALID_CKPT_TYPES, get_pid_checkpoint
from pid._src.inference.cli_utils import parse_resolution
from pid._src.inference.inference_utils import get_rank_and_world_size, load_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)


BOOGU_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {
        "model": "Boogu/Boogu-Image-0.1-Base",
        "num_inference_steps": 50,
        "text_guidance_scale": 4.0,
    },
    "turbo": {
        "model": "Boogu/Boogu-Image-0.1-Turbo",
        "num_inference_steps": 4,
        "text_guidance_scale": 1.0,
    },
}


def _dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _resolve_device(requested_device: str, world_size: int) -> str:
    """Return a concrete per-process CUDA device for torchrun sharded runs."""
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        return requested_device

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"

    if requested_device == "cuda":
        torch.cuda.set_device(0)
        return "cuda:0"

    return requested_device


class BooguVAELatentCapture:
    """Capture the normalized Flux1 latent Boogu feeds into its VAE decode tail."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.latents: list[torch.Tensor] = []
        self._original_decode = None

    def __enter__(self):
        vae = self.pipe.vae
        self._original_decode = vae.decode
        scale = getattr(vae.config, "scaling_factor", None)
        shift = getattr(vae.config, "shift_factor", None)

        def decode_wrapper(latents, *args, **kwargs):
            normalized = latents.detach()
            if shift is not None:
                normalized = normalized - shift
            if scale is not None:
                normalized = normalized * scale
            self.latents.append(normalized.cpu())
            return self._original_decode(latents, *args, **kwargs)

        vae.decode = decode_wrapper
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original_decode is not None:
            self.pipe.vae.decode = self._original_decode
        return False


def _pil_to_tensor_01(image, device: str) -> torch.Tensor:
    """Convert a PIL image to (1, 3, H, W) float tensor in [0, 1]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device)


def _load_boogu_pipeline(args: argparse.Namespace, dtype: torch.dtype, device: str):
    try:
        if args.variant == "turbo":
            from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline as PipelineClass
        else:
            from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline as PipelineClass
    except ImportError as exc:
        raise ImportError(
            "Boogu-Image support requires the Boogu package. Install the Boogu-Image repo "
            "in this environment, for example: `pip install -e /path/to/Boogu-Image`."
        ) from exc

    os.environ["device"] = device
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(".hf_modules_cache").resolve()))

    logger.info("Loading %s from %s (dtype=%s)", PipelineClass.__name__, args.model, dtype)
    pipe = PipelineClass.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    offload_count = int(args.enable_model_cpu_offload) + int(args.enable_sequential_cpu_offload)
    if offload_count > 1:
        raise ValueError("Only one offload strategy can be enabled at a time.")

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload_flag = True
        pipe.enable_model_cpu_offload(device=device)
        logger.info("Boogu pipeline loaded with model CPU offload on %s.", device)
    elif args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload_flag = True
        pipe.enable_sequential_cpu_offload(device=device)
        logger.info("Boogu pipeline loaded with sequential CPU offload on %s.", device)
    else:
        pipe.to(device)
        logger.info("Boogu pipeline loaded on %s.", device)

    return pipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Boogu-Image text-to-image generation, optionally with Flux1 PiD decode")
    parser.add_argument(
        "--variant",
        choices=sorted(BOOGU_VARIANTS),
        default="base",
        help="Boogu model family preset. Controls defaults for --model, --num_inference_steps, and --text_guidance_scale.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model ID or local Boogu pipeline path. Defaults to the selected --variant model.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, default=None, help="Single inline prompt/instruction string")
    prompt_group.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line")

    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(1024, 1024),
        help="Output resolution. Either 'N' or 'W,H'. Default: 1024.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Boogu denoising steps.")
    parser.add_argument("--text_guidance_scale", type=float, default=None, help="Boogu text CFG scale.")
    parser.add_argument("--negative_instruction", type=str, default="", help="Negative prompt/instruction.")
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each prompt.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed, incremented per prompt.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Pipeline dtype.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=None,
        help="Boogu native generation pixel budget. Default: height * width.",
    )
    parser.add_argument(
        "--max_input_image_side_length",
        type=int,
        default=None,
        help="Boogu native generation max side. Default: 2 * max(height, width).",
    )
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Instruction encoder token cap.")
    parser.add_argument(
        "--truncate_instruction_sequence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass truncation=True to Boogu's instruction tokenizer.",
    )
    parser.add_argument(
        "--system_prompt_follows_task_type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Boogu's task-type-specific system prompt.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable Boogu model CPU offload.",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable Boogu sequential CPU offload.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: ./results/boogu_t2i/<variant>",
    )
    parser.add_argument("--save_format", choices=["png", "jpg"], default="png", help="Saved image format.")

    pid_group = parser.add_argument_group("Flux1 PiD decode")
    pid_group.add_argument(
        "--pid_decode",
        action="store_true",
        help="Also decode Boogu's final Flux1 VAE latent with the Flux1 PiD pixel decoder.",
    )
    pid_group.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Flux1 PiD decoder experiment config name. Defaults to checkpoint_registry['flux', --pid_ckpt_type].",
    )
    pid_group.add_argument(
        "--config_file",
        type=str,
        default="pid/_src/configs/pid/config.py",
        help="Hydra config file for the PiD decoder.",
    )
    pid_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Flux1 PiD decoder checkpoint path. Defaults to checkpoint_registry['flux', --pid_ckpt_type].",
    )
    pid_group.add_argument(
        "--pid_ckpt_type",
        type=str,
        choices=list(VALID_CKPT_TYPES),
        default="2k",
        help="Flux1 PiD checkpoint variant to use when --experiment / --checkpoint_path are omitted.",
    )
    pid_group.add_argument("--load_ema_to_reg", action="store_true", help="Load EMA weights into the regular model.")
    pid_group.add_argument("--cfg_scale", type=float, default=1.0, help="PiD decoder CFG scale.")
    pid_group.add_argument("--pid_inference_steps", type=int, default=4, help="PiD decoder denoising steps.")
    pid_group.add_argument("--shift", type=float, default=None, help="PiD decoder flow shift.")
    pid_group.add_argument("--scale", type=int, default=4, help="PiD decoder upscale factor.")
    pid_group.add_argument("--compile", action="store_true", help="torch.compile the PiD decoder.")
    pid_group.add_argument("--upload", action="store_true", help="Upload PiD/baseline outputs to S3.")
    pid_group.add_argument("--group_name", type=str, default="official_demo_boogu", help="S3 group name.")
    pid_group.add_argument("--note", type=str, default="", help="Note appended to the PiD output tag.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    args.extra_experiment_opts = unknown

    defaults = BOOGU_VARIANTS[args.variant]
    if args.model is None:
        args.model = defaults["model"]
    if args.num_inference_steps is None:
        args.num_inference_steps = defaults["num_inference_steps"]
    if args.text_guidance_scale is None:
        args.text_guidance_scale = defaults["text_guidance_scale"]
    if args.num_images_per_prompt < 1:
        parser.error("--num_images_per_prompt must be >= 1")
    if args.pid_decode and (args.experiment is None or args.checkpoint_path is None):
        default_ckpt = get_pid_checkpoint("flux", args.pid_ckpt_type)
        if args.experiment is None:
            args.experiment = default_ckpt.experiment
        if args.checkpoint_path is None:
            args.checkpoint_path = default_ckpt.checkpoint_path
    return args


def main():
    args = parse_args()
    rank, world_size = get_rank_and_world_size()
    device = _resolve_device(args.device, world_size)
    is_rank0 = rank == 0

    prompts = load_prompts(args)
    H, W = args.resolution
    max_input_image_pixels = args.max_input_image_pixels or H * W
    max_input_image_side_length = args.max_input_image_side_length or 2 * max(H, W)
    output_dir = Path(args.output_dir or f"results/boogu_t2i/{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_rank0:
        logger.info(
            "Boogu variant=%s model=%s output=%sx%s steps=%s guidance=%s prompts=%s",
            args.variant,
            args.model,
            W,
            H,
            args.num_inference_steps,
            args.text_guidance_scale,
            len(prompts),
        )
        logger.info("Outputs -> %s", output_dir)

    pipe = _load_boogu_pipeline(args, _dtype_from_arg(args.dtype), device)
    pid_model = None
    pid_tag = None
    uploader = None
    run_pid_and_save = None
    if args.pid_decode:
        from pid._src.inference.decoder import load_our_decoder, run_ours_and_save_step
        from pid._src.inference.inference_utils import AsyncUploader, build_tag

        if not device.startswith("cuda"):
            raise ValueError("--pid_decode requires a CUDA device for the PiD decoder.")
        pid_model = load_our_decoder(args, list(args.extra_experiment_opts), is_rank0)
        pid_tag = build_tag(args, f"boogu_{args.variant}_flux")
        uploader = AsyncUploader(max_workers=8) if args.upload else None
        run_pid_and_save = run_ours_and_save_step
        if is_rank0:
            logger.info("PiD decode enabled: tag=%s checkpoint=%s", pid_tag, args.checkpoint_path)

    indexed_prompts = list(enumerate(prompts))
    if world_size > 1:
        indexed_prompts = indexed_prompts[rank::world_size]
        logger.info("[Rank %s/%s] Processing %s prompts on %s", rank, world_size, len(indexed_prompts), device)

    for prompt_idx, prompt in indexed_prompts:
        seed = args.seed + prompt_idx
        if device.startswith("cuda"):
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)
        logger.info("[%08d] Running Boogu T2I (seed=%s): %r", prompt_idx, seed, prompt[:120])

        boogu_kwargs = dict(
            instruction=prompt,
            negative_instruction=args.negative_instruction,
            height=H,
            width=W,
            max_input_image_pixels=max_input_image_pixels,
            max_input_image_side_length=max_input_image_side_length,
            num_inference_steps=args.num_inference_steps,
            text_guidance_scale=args.text_guidance_scale,
            num_images_per_instruction=args.num_images_per_prompt,
            max_sequence_length=args.max_sequence_length,
            truncate_instruction_sequence=args.truncate_instruction_sequence,
            system_prompt_follows_task_type=args.system_prompt_follows_task_type,
            generator=generator,
            device=device,
            rewriter_device=device,
        )
        if args.variant == "turbo":
            boogu_kwargs["use_dmd_student_inference"] = True

        latent_context = BooguVAELatentCapture(pipe) if args.pid_decode else nullcontext()
        with latent_context as latent_capture:
            output = pipe(**boogu_kwargs)

        pid_latents = None
        if args.pid_decode:
            if not latent_capture.latents:
                raise RuntimeError("Boogu VAE decode was not called; cannot capture a latent for --pid_decode.")
            pid_latents = latent_capture.latents[-1]
            if pid_latents.ndim != 4:
                raise RuntimeError(f"Expected Boogu latent with shape (B, C, H, W), got {tuple(pid_latents.shape)}")
            if pid_latents.shape[0] != len(output.images):
                raise RuntimeError(
                    f"Captured {pid_latents.shape[0]} Boogu latents but got {len(output.images)} output images."
                )
            logger.info("[%08d] Captured Boogu Flux1 latent shape=%s", prompt_idx, tuple(pid_latents.shape))

        for image_idx, image in enumerate(output.images):
            suffix = f"_{image_idx:02d}" if args.num_images_per_prompt > 1 else ""
            save_path = output_dir / f"{prompt_idx:08d}{suffix}.{args.save_format}"
            if args.save_format == "jpg":
                image = image.convert("RGB")
                image.save(save_path, quality=95)
            else:
                image.save(save_path)
            logger.info("[%08d] Saved %s", prompt_idx, save_path)

            if args.pid_decode:
                sample_id = f"{prompt_idx:08d}{suffix}"
                baseline_01 = _pil_to_tensor_01(image, device=device)
                run_pid_and_save(
                    model=pid_model,
                    args=args,
                    tag=pid_tag,
                    sample_id=sample_id,
                    prompt_idx=prompt_idx,
                    step_label="x0",
                    latent=pid_latents[image_idx : image_idx + 1].to(device=device),
                    baseline_01=baseline_01,
                    sigma=0.0,
                    caption=prompt,
                    output_dir=str(output_dir),
                    uploader=uploader,
                    baseline_subdir="boogu_vae_decode",
                    baseline_upload_tag_prefix=f"boogu_{args.variant}_vae_decode",
                )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info("Done! Results saved under %s", output_dir)


if __name__ == "__main__":
    main()
