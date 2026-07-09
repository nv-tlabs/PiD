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
Shared utilities for the PiD inference demos.

This module provides common functions for:
- Image / prompt / manifest I/O (save_image, load_prompts, load_samples, load_input_image)
- Tag generation from checkpoint paths (generate_tag_from_checkpoint, build_tag)
- Distributed rank/world-size lookup (get_rank_and_world_size)
- S3 upload utilities + the AsyncUploader thread pool (optional)
"""

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch as th
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# S3 Upload Configuration
# =============================================================================

# AWS Profile name for S3 access
S3_PROFILE_NAME = "pdx-yifanlu"

# Default S3 bucket name
S3_BUCKET_NAME = "pid"

# S3 root prefix (folder structure: <ROOT_PREFIX>/<group_name>/<experiment_name>/*.mp4)
S3_ROOT_PREFIX = "streamlit_assets"

# Default group name
S3_DEFAULT_GROUP_NAME = "pid_inference"


# =============================================================================
# Tag Generation Functions
# =============================================================================


def generate_tag_from_checkpoint(
    checkpoint_path: str,
    extra_params: Optional[dict] = None,
    load_ema: bool = False,
) -> str:
    """
    Generate tag from checkpoint path and parameters.

    Examples:
        checkpoint_path = ".../flashvsr_0119_stage2_xxx_cp1/checkpoints/iter_000009790"
        -> base_tag = "flashvsr_0119_stage2_xxx_cp1_iter_000009790"

    Args:
        checkpoint_path: Path to model checkpoint
        extra_params: Dictionary of extra parameters to append to tag
        load_ema: Whether EMA weights are loaded

    Returns:
        Generated tag string
    """
    # Normalize path
    path = checkpoint_path.rstrip("/")

    # Extract iter name (last component)
    iter_name = os.path.basename(path)

    # Extract experiment name (parent of checkpoints dir)
    parent = os.path.dirname(path)
    if os.path.basename(parent) == "checkpoints":
        experiment_name = os.path.basename(os.path.dirname(parent))
    else:
        # Fallback: use parent directory name
        experiment_name = os.path.basename(parent)

    # Build base tag
    tag = f"{experiment_name}_{iter_name}"

    # Append extra parameters
    if extra_params:
        for key, value in extra_params.items():
            if value is not None:
                tag += f"_{key}{value}"

    # Append EMA/reg suffix
    tag += "_ema" if load_ema else "_reg"

    return tag


def build_tag(args, backbone: str) -> str:
    """Build the per-run output tag: "<backbone>_<checkpoint-derived tag>".

    The backbone prefix keeps the same checkpoint's results in distinct folders
    across backbones (and distinct S3 experiment dirs when --upload is set).
    Shared by the from_ldm / from_clean demo entrypoints.
    """
    extra_params = {"cfg": args.cfg_scale}
    if args.pid_inference_steps is not None:
        extra_params["steps"] = args.pid_inference_steps
    if args.shift is not None:
        extra_params["shift"] = args.shift
    if args.note:
        extra_params["note_"] = args.note
    base_tag = generate_tag_from_checkpoint(args.checkpoint_path, extra_params, load_ema=args.load_ema_to_reg)
    return f"{backbone}_{base_tag}"


# =============================================================================
# Distributed Processing Functions
# =============================================================================


def get_rank_and_world_size() -> Tuple[int, int]:
    """
    Get rank and world_size from environment (set by torchrun).

    Returns:
        Tuple of (rank, world_size)
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


# =============================================================================
# S3 Upload Functions
# =============================================================================


def _parse_aws_credentials(
    cred_path_or_profile: Union[str, Path, None] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse AWS credentials from file, AWS profile, or environment variables.

    Args:
        cred_path_or_profile: Path to credentials file, AWS profile name, or None

    Returns:
        Tuple of (endpoint_url, access_key, secret_key, region)
    """
    if cred_path_or_profile:
        cred_path = Path(cred_path_or_profile)

        # Check if it's a file path
        if cred_path.exists() and cred_path.is_file():
            import json

            credentials = json.load(open(cred_path))
            endpoint = credentials.get("endpoint_url")
            access_key = credentials.get("aws_access_key_id")
            secret_key = credentials.get("aws_secret_access_key")
            region = credentials.get("region_name", None)
        else:
            # Treat as AWS profile name
            profile = str(cred_path_or_profile)

            credentials_file = Path.home() / ".aws" / "credentials"
            config_file = Path.home() / ".aws" / "config"

            # Parse credentials file
            credentials_parser = ConfigParser()
            if credentials_file.exists():
                credentials_parser.read(credentials_file)
            else:
                raise FileNotFoundError(f"AWS credentials file not found: {credentials_file}")

            # Parse config file
            config_parser = ConfigParser()
            if config_file.exists():
                config_parser.read(config_file)

            # Get credentials from credentials file
            if profile in credentials_parser:
                access_key = credentials_parser[profile].get("aws_access_key_id")
                secret_key = credentials_parser[profile].get("aws_secret_access_key")
                region = credentials_parser[profile].get("region")
                endpoint = credentials_parser[profile].get("endpoint_url")
            else:
                access_key = None
                secret_key = None
                region = None
                endpoint = None

            # If not found in credentials file, try config file
            if not region or not endpoint:
                config_section = f"profile {profile}" if profile != "default" else profile

                if config_section in config_parser:
                    if not region:
                        region = config_parser[config_section].get("region")
                    if not endpoint:
                        endpoint = config_parser[config_section].get("endpoint_url")
    else:
        # Load from environment variables
        endpoint = os.getenv("AWS_ENDPOINT_URL")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")

    # If no endpoint specified, use AWS S3 default endpoint
    if not endpoint:
        endpoint = "https://s3.amazonaws.com"

    return endpoint, access_key, secret_key, region


def get_s3_client(profile_name: Optional[str] = None):
    """
    Get a boto3 S3 client.

    Args:
        profile_name: AWS profile name (defaults to S3_PROFILE_NAME)

    Returns:
        boto3 S3 client
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 upload. Install with: pip install boto3")

    if profile_name is None:
        profile_name = S3_PROFILE_NAME

    endpoint, access_key, secret_key, region = _parse_aws_credentials(profile_name)

    kwargs = {"endpoint_url": endpoint}
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    if region:
        kwargs["region_name"] = region

    return boto3.client("s3", **kwargs)


def upload_file_to_s3(
    local_path: str,
    s3_key: str,
    bucket_name: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Upload a single file to S3.

    Args:
        local_path: Path to local file
        s3_key: S3 key (path in bucket)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        s3_client: boto3 S3 client (will create one if not provided)

    Returns:
        True if upload succeeded, False otherwise
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"Failed to upload {local_path} to s3://{bucket_name}/{s3_key}: {e}")
        return False


def upload_video_to_s3(
    local_path: str,
    group_name: str,
    experiment_name: str,
    bucket_name: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Upload a video file to S3 with standard path structure.

    The file will be uploaded to: s3://<bucket>/<ROOT_PREFIX>/<group_name>/<experiment_name>/<filename>

    Args:
        local_path: Path to local video file
        group_name: Group name (e.g., "large_motion_lq")
        experiment_name: Experiment name (typically the tag)
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME)
        s3_client: boto3 S3 client (will create one if not provided)

    Returns:
        True if upload succeeded, False otherwise
    """
    filename = os.path.basename(local_path)
    s3_key = f"{S3_ROOT_PREFIX}/{group_name}/{experiment_name}/{filename}"

    success = upload_file_to_s3(local_path, s3_key, bucket_name, s3_client)

    if success:
        print(f"Uploaded to s3://{bucket_name or S3_BUCKET_NAME}/{s3_key}")

    return success


def maybe_upload_video(
    local_path: str,
    tag: str,
    upload: bool,
    group_name: Optional[str] = None,
) -> bool:
    """
    Optionally upload a single video to S3 immediately after generation.

    Args:
        local_path: Path to the local video file
        tag: Experiment tag (used as experiment_name in S3)
        upload: Whether to upload
        group_name: S3 group name (defaults to S3_DEFAULT_GROUP_NAME)

    Returns:
        True if upload succeeded or was skipped, False if upload failed
    """
    if not upload:
        return True

    if group_name is None:
        group_name = S3_DEFAULT_GROUP_NAME

    if not os.path.isfile(local_path):
        print(f"Video file not found: {local_path}")
        return False

    try:
        s3_client = get_s3_client()
        success = upload_video_to_s3(local_path, group_name, tag, s3_client=s3_client)
        return success
    except Exception as e:
        print(f"Upload failed for {local_path}: {e}")
        return False


# =============================================================================
# Demo image / prompt / manifest I/O
# =============================================================================
#   - save_image / _tensor_to_pil : write a [-1, 1] tensor to PNG/JPG.
#   - load_prompts                : resolve --prompt / --prompt_file into a list[str]  (from_ldm).
#   - load_samples                : resolve --input_path / --manifest into (image, prompt) pairs (from_clean).
#   - load_input_image            : load an input image at native size (cropped to a 16-multiple) to [-1, 1] (from_clean).


def _tensor_to_pil(tensor: th.Tensor) -> Image.Image:
    """Convert [C, H, W] in [-1, 1] to PIL Image."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def save_image(sample: th.Tensor, save_path: str, quality: int = 95) -> str:
    """Save [C, H, W] or [C, 1, H, W] tensor in [-1, 1]. Format inferred from extension."""
    if sample.dim() == 4:
        sample = sample.squeeze(1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img = _tensor_to_pil(sample)
    if save_path.lower().endswith((".jpg", ".jpeg")):
        img.save(save_path, quality=quality)
    else:
        img.save(save_path)
    return save_path


def load_prompts(args) -> List[str]:
    """Resolve a list of prompts from --prompt (single) or --prompt_file (one per line)."""
    if args.prompt is not None:
        return [args.prompt]
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"--prompt_file {args.prompt_file} is empty after stripping.")
    return prompts


def load_samples(args) -> List[Tuple[str, Optional[str]]]:
    """Resolve the (image_path, per_sample_prompt_or_None) list from argparse.

    --input_path : returns a single-element list, prompt always None (defers to --prompt /
                   fixed_positive_prompt downstream).
    --manifest   : reads a JSONL file; each object must have an "image" key and may
                   optionally carry a "prompt" key. Per-line prompts override --prompt.
    """
    if args.manifest is not None:
        samples: List[Tuple[str, Optional[str]]] = []
        with open(args.manifest, "r") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"--manifest {args.manifest} line {ln} is not valid JSON: {e}")
                if "image" not in obj:
                    raise ValueError(f'--manifest {args.manifest} line {ln} missing "image" key: {obj!r}')
                samples.append((str(obj["image"]), obj.get("prompt")))
        if not samples:
            raise ValueError(f"--manifest {args.manifest} is empty after stripping.")
        return samples
    return [(args.input_path, None)]


def load_input_image(
    path: str,
    pad_to_multiple: int = 16,
) -> th.Tensor:
    """Load image and return [1, 3, H, W] float32 in [-1, 1] on CPU.

    Preserves the image's native H, W, only center-cropping a few pixels so each side is a
    multiple of pad_to_multiple (keeps the VAE latent grid integer). This single native-size
    preprocessing is shared across all backbones; the fixed-resolution encoders
    (dinov2 / siglip) resize internally to their own native interface.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = (w // pad_to_multiple) * pad_to_multiple
    new_h = (h // pad_to_multiple) * pad_to_multiple
    if new_w == 0 or new_h == 0:
        raise ValueError(f"Image {path} size {w}x{h} is smaller than pad_to_multiple={pad_to_multiple}.")
    if (new_w, new_h) != (w, h):
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))

    arr = np.asarray(img, np.uint8).astype("float32")
    t = th.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t


# =============================================================================
# Async S3 uploader
# =============================================================================


class AsyncUploader:
    """Fire-and-forget S3 uploader backed by a thread pool.

    Usage:
        uploader = AsyncUploader(max_workers=8)
        uploader.submit(maybe_upload_video, path, tag, upload, group)
        ...
        uploader.wait()   # block until all queued uploads finish
    """

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []

    def submit(self, fn, *args, **kwargs):
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)

    def wait(self):
        failed = 0
        for fut in as_completed(self._futures):
            try:
                ok = fut.result()
                if ok is False:
                    failed += 1
            except Exception as e:
                logger.error(f"Async upload error: {e}")
                failed += 1
        self._futures.clear()
        self._executor.shutdown(wait=False)
        if failed:
            logger.warning(f"{failed} upload(s) failed")


def decode_image_bytes_to_tensor(png_bytes: bytes, device: str = "cpu") -> th.Tensor:
    """Decode image bytes (PNG/JPG/etc) to [1, C, H, W] float32 tensor in [-1, 1].

    Inverse of encode_tensor_as_png(). Also works with JPEG and other PIL-supported formats.
    """
    pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    t = th.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0  # [C, H, W] in [-1, 1]
    return t.unsqueeze(0).to(device)  # [1, C, H, W]


def load_fix_batch(pt_path: str, device: str = "cpu") -> dict:
    """Load a fix_batch .pt file, auto-detecting image vs video payloads.

    Handles both new format ("HQ_video_or_image") and legacy format ("image").
    Adds exactly one model-facing alias for GT pixels:
    - image fix_batch -> "image"
    - video fix_batch -> "video"

    Returns dict with tensors in [-1, 1] float32:
        "HQ_video_or_image": [1, 3, H, W] or [1, 3, T, H, W]
        "LQ_video_or_image": [1, 3, H_lq, W_lq] or [1, 3, T, H_lq, W_lq]
        "LQ_latent": optional pre-computed latent
        "caption": list[str]
    """
    data = th.load(pt_path, map_location="cpu", weights_only=False)

    for key in ["HQ_video_or_image", "LQ_video_or_image"]:
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, bytes):
            # Image bytes (PNG/JPG) -> decode to tensor
            data[key] = decode_image_bytes_to_tensor(val, device=device)
        elif isinstance(val, th.Tensor):
            # Raw tensor (legacy format)
            if val.dtype == th.uint8:
                data[key] = val.float() / 127.5 - 1.0
            data[key] = data[key].to(device)

    # Move LQ_latent to target device if present
    if "LQ_latent" in data and isinstance(data["LQ_latent"], th.Tensor):
        data["LQ_latent"] = data["LQ_latent"].to(device)
        if data["LQ_latent"].ndim == 3:
            data["LQ_latent"] = data["LQ_latent"].unsqueeze(0)

    if "HQ_video_or_image" in data:
        data["image"] = data["HQ_video_or_image"]
        data.setdefault("media_type", "image")

    return data
