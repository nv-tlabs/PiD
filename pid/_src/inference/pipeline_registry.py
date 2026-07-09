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
Registry of diffusers pipelines for FPD-vs-VAE evaluation on generated images.

Each DiffusionPipelineConfig describes how to load a diffusers pipeline, extract
latents in (B, C, H, W) format, denormalize them, and decode with the pipeline's VAE.

Supported backbones: flux, sdxl, sd3, flux2, flux2-klein-4b, flux2-klein-9b, qwenimage, qwenimage-2512, zimage, zimage-turbo.

Latent normalization conventions:
  - Flux/SDXL/SD3: simple affine scale+shift  →  raw = latent / scale + shift
  - Flux2: BatchNorm-based  →  raw = latent * bn_std + bn_mean
    (running stats stored in AutoencoderKLFlux2.latent_norm)
  - QwenImage / QwenImage-2512: per-channel mean/std  →  raw = latent * std + mean
    (vectors stored in pipeline.vae.config.latents_mean / latents_std)
  - ZImage/ZImage-Turbo: affine scale+shift read from pipeline.vae.config at runtime
    (vae_scale_factor=0 in registry signals runtime lookup)

Inference-vs-training-frame conventions:
  Flow-matching backbones (Flux/Flux2/SD3/QwenImage/ZImage) store intermediate latents
  in the same flow-matching frame the PiD student was trained on — x_t = (1-σ) x_0 + σ ε,
  σ ∈ [0, 1]. SDXL is the exception: its EulerDiscreteScheduler keeps latents in the
  variance-exploding Euler frame x_t = x_0 + σ_eu ε with σ_eu ∈ [0, ~15]. `to_training_frame`
  rescales SDXL captures to the variance-preserving form that the SDXL PiD student trained
  with (x_vp = x_eu / sqrt(σ²+1), σ_vp = σ_eu / sqrt(σ²+1)); for every other backbone it is
  a no-op.

Diffusers `output_type="latent"` returns the denoised latent in the *normalized*
space (same convention as tokenizer.encode()). For FPD the latent is used directly
— no extra denormalization is needed. denormalize_latent() is only needed for VAE
decode when the pipeline's decode path doesn't handle it internally.

Requires diffusers >= 0.37.0 for Flux2/QwenImage/ZImage support.
"""

import importlib
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiffusionPipelineConfig:
    name: str  # "flux", "sdxl", "sd3", "flux2"
    pipeline_class: str  # e.g. "diffusers.FluxPipeline"
    default_model_id: str  # HuggingFace model ID
    latent_channels: int  # 16 (Flux/SD3), 4 (SDXL), 32 (Flux2)
    spatial_compression: int  # 8
    # Affine normalization (Flux1/SDXL/SD3). Set both to 0 for BN-based (Flux2).
    vae_scale_factor: float  # diffusers VAE scaling
    vae_shift_factor: float  # diffusers VAE shift (0 if none)
    # Whether this backbone uses BatchNorm-based latent normalization (Flux2)
    uses_bn_normalization: bool = False
    # Whether this backbone uses per-channel mean/std normalization (QwenImage)
    uses_perchannel_normalization: bool = False
    # Whether the VAE is a video-style 3D VAE that produces 5D latents (QwenImage)
    has_temporal_dim: bool = False
    default_resolution: tuple[int, int] = (1024, 1024)
    default_num_inference_steps: int = 28
    default_guidance_scale: float = 3.5
    # Extra kwargs forwarded to pipeline.__call__
    extra_generate_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIPELINE_REGISTRY: dict[str, DiffusionPipelineConfig] = {
    "flux": DiffusionPipelineConfig(
        name="flux",
        pipeline_class="diffusers.FluxPipeline",
        default_model_id="black-forest-labs/FLUX.1-dev",
        latent_channels=16,
        spatial_compression=8,
        vae_scale_factor=0.3611,
        vae_shift_factor=0.1159,
        default_resolution=(1024, 1024),
        default_num_inference_steps=28,
        default_guidance_scale=3.5,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
    "sdxl": DiffusionPipelineConfig(
        name="sdxl",
        pipeline_class="diffusers.StableDiffusionXLPipeline",
        default_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        latent_channels=4,
        spatial_compression=8,
        vae_scale_factor=0.13025,
        vae_shift_factor=0.0,
        default_resolution=(1024, 1024),
        default_num_inference_steps=30,
        default_guidance_scale=7.5,
    ),
    "sd3": DiffusionPipelineConfig(
        name="sd3",
        pipeline_class="diffusers.StableDiffusion3Pipeline",
        default_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        latent_channels=16,
        spatial_compression=8,
        vae_scale_factor=1.5305,
        vae_shift_factor=0.0609,
        default_resolution=(1024, 1024),
        default_num_inference_steps=28,
        default_guidance_scale=4.0,
    ),
    "flux2": DiffusionPipelineConfig(
        name="flux2",
        pipeline_class="diffusers.Flux2Pipeline",
        default_model_id="black-forest-labs/FLUX.2-dev",
        latent_channels=32,
        spatial_compression=8,
        # Flux2 uses BatchNorm-based normalization, not affine scale/shift.
        # Set to 0 — actual denormalization uses pipeline.vae.latent_norm running stats.
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        uses_bn_normalization=True,
        default_resolution=(1024, 1024),
        default_num_inference_steps=50,
        default_guidance_scale=4.0,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
    # FLUX.2-klein (the distilled "klein" siblings of FLUX.2-dev) — different
    # diffusers entrypoint (Flux2KleinPipeline) but the SAME Flux2
    # AutoencoderKLFlux2 (32-ch BatchNorm latent). The PiD student is identical,
    # so checkpoint_registry aliases these backbone-tags to the flux2 entries.
    # 4B and 9B share the same config; only default_model_id differs. Defaults
    # follow the HF model cards: 1024x1024, guidance_scale=1.0, num_inference_steps=4
    # (distilled few-step models).
    "flux2-klein-4b": DiffusionPipelineConfig(
        name="flux2-klein-4b",
        pipeline_class="diffusers.Flux2KleinPipeline",
        default_model_id="black-forest-labs/FLUX.2-klein-4B",
        latent_channels=32,
        spatial_compression=8,
        # Same BatchNorm-based normalization as flux2 — denorm uses pipeline.vae.bn.
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        uses_bn_normalization=True,
        default_resolution=(1024, 1024),
        default_num_inference_steps=4,
        default_guidance_scale=1.0,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
    "flux2-klein-9b": DiffusionPipelineConfig(
        name="flux2-klein-9b",
        pipeline_class="diffusers.Flux2KleinPipeline",
        default_model_id="black-forest-labs/FLUX.2-klein-9B",
        latent_channels=32,
        spatial_compression=8,
        # Same BatchNorm-based normalization as flux2 — denorm uses pipeline.vae.bn.
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        uses_bn_normalization=True,
        default_resolution=(1024, 1024),
        default_num_inference_steps=4,
        default_guidance_scale=1.0,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
    "qwenimage": DiffusionPipelineConfig(
        name="qwenimage",
        pipeline_class="diffusers.QwenImagePipeline",
        default_model_id="Qwen/Qwen-Image",
        latent_channels=16,
        spatial_compression=8,
        # QwenImage uses per-channel mean/std normalization, not affine scale/shift.
        # Actual denormalization reads pipeline.vae.config.latents_mean / latents_std.
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        uses_perchannel_normalization=True,
        has_temporal_dim=True,
        default_resolution=(1024, 1024),
        default_num_inference_steps=50,
        default_guidance_scale=4.0,
        extra_generate_kwargs={"max_sequence_length": 512, "true_cfg_scale": 4.0, "negative_prompt": " "},
    ),
    # Dec 2025 refresh of Qwen-Image — same AutoencoderKLQwenImage and same
    # QwenImagePipeline class; only the transformer / text-encoder checkpoint
    # differs. The PiD student is identical, so checkpoint_registry aliases
    # this backbone-tag to the qwenimage entry.
    "qwenimage-2512": DiffusionPipelineConfig(
        name="qwenimage-2512",
        pipeline_class="diffusers.QwenImagePipeline",
        default_model_id="Qwen/Qwen-Image-2512",
        latent_channels=16,
        spatial_compression=8,
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        uses_perchannel_normalization=True,
        has_temporal_dim=True,
        default_resolution=(1024, 1024),
        default_num_inference_steps=50,
        default_guidance_scale=4.0,
        extra_generate_kwargs={"max_sequence_length": 512, "true_cfg_scale": 4.0, "negative_prompt": " "},
    ),
    "zimage": DiffusionPipelineConfig(
        name="zimage",
        pipeline_class="diffusers.ZImagePipeline",
        default_model_id="Tongyi-MAI/Z-Image",
        latent_channels=16,
        spatial_compression=8,
        # ZImage uses affine normalization but exact values depend on the pretrained
        # checkpoint. Set to 0 so denormalize_latent() reads from pipeline.vae.config.
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        default_resolution=(1024, 1024),
        default_num_inference_steps=50,
        default_guidance_scale=5.0,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
    "zimage-turbo": DiffusionPipelineConfig(
        name="zimage-turbo",
        pipeline_class="diffusers.ZImagePipeline",
        default_model_id="Tongyi-MAI/Z-Image-Turbo",
        latent_channels=16,
        spatial_compression=8,
        # ZImage-Turbo shares ZImage's VAE/latent convention. Runtime values are
        # read from pipeline.vae.config by denormalize_latent().
        vae_scale_factor=0.0,
        vae_shift_factor=0.0,
        default_resolution=(1024, 1024),
        # The model card describes Turbo as an 8-NFE distilled model. Diffusers'
        # example uses num_inference_steps=9, yielding 8 non-zero scheduler jumps
        # followed by the terminal sigma=0 sample.
        default_num_inference_steps=9,
        default_guidance_scale=0.0,
        extra_generate_kwargs={"max_sequence_length": 512},
    ),
}


def get_config(name: str) -> DiffusionPipelineConfig:
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(PIPELINE_REGISTRY.keys())}")
    return PIPELINE_REGISTRY[name]


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def load_pipeline(
    name: str, model_id: Optional[str] = None, dtype=torch.bfloat16, device: str = "cuda", cpu_offload: bool = False
):
    """Dynamically import and load a diffusers pipeline.

    Args:
        cpu_offload: If True, use enable_model_cpu_offload() instead of .to(device).
            Keeps model weights on CPU and only moves the active component to GPU during
            forward pass. Essential for large models (Flux2, QwenImage, etc.) that exceed
            single-GPU VRAM when all components are loaded simultaneously.

    Returns (pipeline, cfg) where pipeline is ready to call and cfg is the
    DiffusionPipelineConfig for this backbone.
    """
    cfg = get_config(name)
    model_id = model_id or cfg.default_model_id

    # e.g. "diffusers.FluxPipeline" -> module="diffusers", cls="FluxPipeline"
    module_path, cls_name = cfg.pipeline_class.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    PipelineClass = getattr(mod, cls_name)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Loading {cfg.pipeline_class} from {model_id} (dtype={dtype}) ...")
    pipeline = PipelineClass.from_pretrained(model_id, torch_dtype=dtype, token=token)
    if cpu_offload:
        # Only the active component (text encoder / transformer / VAE) lives on GPU at a time.
        # enable_model_cpu_offload() defaults to gpu_id=0 — must pass the correct device
        # explicitly for multi-GPU torchrun, otherwise all ranks pile onto GPU 0.
        gpu_id = torch.cuda.current_device()
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        print(f"Pipeline loaded with model CPU offload (gpu_id={gpu_id}).")
    else:
        pipeline = pipeline.to(device)
        print(f"Pipeline loaded on {device}.")
    return pipeline, cfg


# ---------------------------------------------------------------------------
# Latent handling
# ---------------------------------------------------------------------------


def denormalize_latent(pipeline, latent: torch.Tensor, cfg: DiffusionPipelineConfig) -> torch.Tensor:
    """Reverse the latent normalization applied during VAE encode.

    For Flux1/SDXL/SD3 (affine): raw = latent / scale + shift
    For Flux2 (BatchNorm):        raw = latent * bn_std + bn_mean
        where bn_std/bn_mean come from pipeline.vae.latent_norm running stats.

    Only needed when manually feeding latent to the pipeline's VAE.decode(),
    which expects the *raw* (un-normalized) latent space.
    """
    if cfg.uses_bn_normalization:
        # Flux2: denormalize via BatchNorm running statistics.
        # diffusers 0.37+: stored as pipeline.vae.bn (BatchNorm2d, affine=False).
        bn = pipeline.vae.bn
        # running_mean/var are (C_packed,) where C_packed = latent_channels * patch_h * patch_w
        # The latent from output_type="latent" is already in packed BN-normalized space.
        bn_mean = bn.running_mean.to(latent.device, latent.dtype)
        bn_var = bn.running_var.to(latent.device, latent.dtype)
        bn_std = (bn_var + bn.eps).sqrt()
        # Reshape to broadcast: (1, C_packed, 1, 1)
        bn_mean = bn_mean.view(1, -1, 1, 1)
        bn_std = bn_std.view(1, -1, 1, 1)
        return latent * bn_std + bn_mean
    elif cfg.uses_perchannel_normalization:
        # QwenImage: denormalize via per-channel mean/std from VAE config
        latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1).to(latent.device, latent.dtype)
        latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1).to(latent.device, latent.dtype)
        return latent * latents_std + latents_mean
    else:
        # Affine scale/shift
        scale = cfg.vae_scale_factor
        shift = cfg.vae_shift_factor
        if scale == 0.0:
            # Fallback: read from pipeline's VAE config at runtime (e.g., ZImage)
            scale = pipeline.vae.config.scaling_factor
            shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
        return latent / scale + shift


def to_training_frame(latent: torch.Tensor, sigma: float, cfg: DiffusionPipelineConfig) -> tuple[torch.Tensor, float]:
    """Map an inference-loop intermediate latent + its scheduler σ to the frame the PiD
    student was trained on.

    Flow-matching backbones (Flux / Flux2 / SD3 / QwenImage / ZImage) already use the
    σ ∈ [0, 1], x_t = (1-σ) x_0 + σ ε convention — identity for them.

    SDXL is the exception. Its `EulerDiscreteScheduler` keeps latents in the variance-
    exploding Euler frame x_t^eu = x_0 + σ_eu ε with σ_eu ∈ [0, ~15], but the SDXL PiD
    student was trained with the VP/DDPM frame x_t^vp = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε,
    σ_vp ∈ [0, 1]. The two frames share the same noise realization differing only by a
    scalar: x_t^vp = x_t^eu / sqrt(σ²+1), σ_vp = σ_eu / sqrt(σ²+1).
    """
    if cfg.name == "sdxl":
        denom = float((sigma**2 + 1.0) ** 0.5)
        if denom == 0.0:
            # σ_eu == 0 → final clean latent; Euler and VP frames coincide.
            return latent, 0.0
        return latent / denom, sigma / denom
    return latent, sigma


def extract_latent(pipeline, raw_output, cfg: DiffusionPipelineConfig, height: int, width: int) -> torch.Tensor:
    """Normalize pipeline output_type="latent" to (B, C, H, W).

    Flux1 packs latents into (B, seq_len, C) — needs _unpack_latents().
    Flux2 packs latents into (B, seq_len, C) — needs _unpack_latents_with_ids().
    SDXL / SD3 already return (B, C, H, W).
    """
    latent = raw_output.images  # could be packed for Flux/Flux2

    if cfg.name == "flux":
        # Flux1: packed (B, seq_len, C) → (B, C, H, W)
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        latent = FluxPipeline._unpack_latents(
            latent,
            height=height,
            width=width,
            vae_scale_factor=pipeline.vae_scale_factor,
        )
    elif cfg.name in ("flux2", "flux2-klein-4b", "flux2-klein-9b"):
        # Flux2-family target is the packed BN-normalized (B, 128, H/16, W/16) latent the
        # PiD student trained on. Two cases by rank:
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        if latent.ndim == 4:
            # Flux2KleinPipeline's output_type="latent" already runs _unpack_latents_with_ids
            # + BN-denormalize + _unpatchify internally, so it returns the RAW (B, 32, H/8, W/8)
            # VAE latent — unlike Flux2Pipeline, which returns the packed normalized (B, seq, 128)
            # latent. Re-derive the packed + normalized representation: patchify back to
            # (B, 128, H/16, W/16), then BN-normalize with the same running stats
            # denormalize_latent() will later invert.
            patched = Flux2Pipeline._patchify_latents(latent)
            bn = pipeline.vae.bn
            bn_mean = bn.running_mean.view(1, -1, 1, 1).to(patched.device, patched.dtype)
            bn_std = (bn.running_var.view(1, -1, 1, 1) + bn.eps).sqrt().to(patched.device, patched.dtype)
            latent = (patched - bn_mean) / bn_std
        else:
            # Packed (B, seq_len, C) → (B, C, H, W) using position IDs. This is Flux2-dev's
            # final output AND every flux2-family intermediate xt capture (mid-loop latents are
            # packed for both pipelines). diffusers 0.37+ API: _unpack_latents_with_ids(x, x_ids)
            # where x_ids are (B, H*W, 4) coordinates from _prepare_latent_ids.
            vae_sf = pipeline.vae_scale_factor  # typically 8
            latent_h = height // (vae_sf * 2)
            latent_w = width // (vae_sf * 2)
            # _prepare_latent_ids takes a (B, C, H, W) tensor and reads .shape
            dummy = torch.zeros(latent.shape[0], 1, latent_h, latent_w, device=latent.device)
            latent_ids = Flux2Pipeline._prepare_latent_ids(dummy).to(latent.device)
            result = Flux2Pipeline._unpack_latents_with_ids(latent, latent_ids)
            # _unpack_latents_with_ids returns a list/stacked tensor (B, C, H, W)
            latent = result if isinstance(result, torch.Tensor) else torch.stack(result, dim=0)
    elif cfg.name in ("qwenimage", "qwenimage-2512"):
        # QwenImage: packed (B, seq_len, C) → (B, C, 1, H, W) with temporal dim
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

        latent = QwenImagePipeline._unpack_latents(
            latent,
            height=height,
            width=width,
            vae_scale_factor=pipeline.vae_scale_factor,
        )
        # Squeeze temporal dim: (B, C, 1, H, W) → (B, C, H, W)
        latent = latent.squeeze(2)

    # ZImage: already (B, C, H, W), no unpacking needed.

    if latent.ndim != 4:
        raise RuntimeError(f"Expected 4-D latent (B, C, H, W) after extraction, got shape {latent.shape}")
    return latent


def decode_with_pipeline_vae(pipeline, latent: torch.Tensor, cfg: DiffusionPipelineConfig) -> torch.Tensor:
    """Standard VAE decode using the pipeline's own VAE.

    Takes the *normalized* latent (as returned by output_type="latent"),
    denormalizes it, and decodes to pixel space.

    Returns: (B, 3, H, W) float tensor in [0, 1].
    """
    raw_latent = denormalize_latent(pipeline, latent, cfg)

    if cfg.uses_bn_normalization:
        # Flux2 VAE: unpatch before decoding.
        # raw_latent is (B, C_packed, pH, pW) — C_packed = latent_channels * patch_h * patch_w.
        # Must undo patchification to get (B, latent_channels, H/8, W/8) before vae.decode().
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        raw_latent = Flux2Pipeline._unpatchify_latents(raw_latent)

    if cfg.has_temporal_dim:
        # Video-style 3D VAE (e.g., QwenImage): expects (B, C, T, H, W)
        raw_latent = raw_latent.unsqueeze(2)

    # Match VAE dtype — schedulers often output float32 while VAE weights are bfloat16.
    raw_latent = raw_latent.to(pipeline.vae.dtype)

    with torch.no_grad():
        decoded = pipeline.vae.decode(raw_latent, return_dict=False)[0]

    if cfg.has_temporal_dim:
        # 3D VAE returns (B, 3, T, H, W) — take first frame
        decoded = decoded[:, :, 0]

    # diffusers VAE outputs in [-1, 1] — map to [0, 1]
    decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
    return decoded


def print_latent_stats(latent: torch.Tensor, label: str = "latent"):
    """Print mean/std/min/max for latent debugging."""
    with torch.no_grad():
        print(
            f"  [{label}] shape={list(latent.shape)} "
            f"mean={latent.mean().item():.4f} std={latent.std().item():.4f} "
            f"min={latent.min().item():.4f} max={latent.max().item():.4f}"
        )
