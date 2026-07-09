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
Shared official PID checkpoint registry.

Single source of truth for the (experiment_name, checkpoint_path) pair used by
every pixel-decoder demo in `pid/_src/inference/`. The registry is keyed by
(backbone, ckpt_type):

  ckpt_type = "2k"             Original 2048px-trained decoders, used as
                               512→2048 (4×) decoder for diffusers-style
                               backbones, or 256→2048 (8×) for Scale-RAE.
  ckpt_type = "2kto4k"         Multi-resolution-trained v1 decoders (data
                               bucketing 2048→3840 + SD3-style dynamic shift).
  ckpt_type = "2kto4k_v1pt5"   Multi-resolution-trained v1.5 decoders.
                               The 4K variants are designed for 1024 LDM →
                               4K (3840) decoding. dinov2 / siglip have no
                               4K variant.

Backbone-tag namespace:
  flux           Flux1-dev (16-ch VAE)                  LDM + from_clean   (2k + 2kto4k_v1pt5)
  flux2          Flux2-dev (128-ch BN VAE)              LDM + from_clean   (2k + 2kto4k_v1pt5)
  flux2-klein-4b FLUX.2-klein-4B (same Flux2 BN VAE)    LDM only — aliases to flux2 (2k + 2kto4k_v1pt5)
  flux2-klein-9b FLUX.2-klein-9B (same Flux2 BN VAE)    LDM only — aliases to flux2 (2k + 2kto4k_v1pt5)
  sd3            SD3 medium (16-ch VAE)                 LDM + from_clean   (2k + 2kto4k)
  sdxl           SDXL (4-ch VAE, VP-frame student)      LDM + from_clean   (2kto4k only)
  qwenimage      Qwen-Image (16-ch 3D VAE)              LDM + from_clean   (2kto4k_v1pt5 only)
  qwenimage-2512 Qwen-Image-2512 (Dec 2025 refresh)     LDM only — aliases to qwenimage (2kto4k_v1pt5 only)
  zimage         ZImage (Flux1's 16-ch VAE)             LDM only — reuses Flux1 model (2k + 2kto4k_v1pt5)
  zimage-turbo   ZImage-Turbo (same 16-ch VAE)          LDM only — reuses Flux1 model (2k + 2kto4k_v1pt5)
  dinov2         DINOv2-B + RAE ViT-XL (768-ch RAE)     LDM + from_clean   (2k only, sr4x)
  siglip         SigLIP-2 So400M + Scale-RAE ViT-XL     LDM + from_clean   (2k only, sr8x)

`pid_scale` is the spatial upscaling factor baked into the PID network
(sr4x → 4, sr8x → 8) and is forwarded to the demo's --scale argument.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDCheckpoint:
    experiment: str
    checkpoint_path: str
    pid_scale: int


_CKPT_ROOT = "checkpoints"

VALID_CKPT_TYPES = ("2k", "2kto4k", "2kto4k_v1pt5")


PID_CHECKPOINT_REGISTRY: dict[tuple[str, str], PIDCheckpoint] = {
    # ---- 2k (the original 2048-trained release) ----
    ("flux", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sd3", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("zimage", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("dinov2", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_dinov2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_dinov2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("siglip", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr8x_official_siglip_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth",
        pid_scale=8,
    ),
    # ---- 2kto4k v1 (multi-res-trained, dynamic_shift-aware) ----
    ("sd3", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sdxl", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sdxl_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    # ---- 2kto4k v1.5 (multi-res-trained, dynamic_shift-aware) ----
    ("flux", "2kto4k_v1pt5"): PIDCheckpoint(
        experiment="PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2kto4k_v1pt5"): PIDCheckpoint(
        experiment="PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("qwenimage", "2kto4k_v1pt5"): PIDCheckpoint(
        experiment="PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
}
# ZImage and ZImage-Turbo use Flux1's 16-ch VAE for both ckpt types → alias to
# the flux entries. Keep explicit aliases (vs. duplicating) so updating "flux"
# updates these backbones too.
PID_CHECKPOINT_REGISTRY[("zimage-turbo", "2k")] = PID_CHECKPOINT_REGISTRY[("flux", "2k")]
PID_CHECKPOINT_REGISTRY[("zimage", "2kto4k_v1pt5")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k_v1pt5")]
PID_CHECKPOINT_REGISTRY[("zimage-turbo", "2kto4k_v1pt5")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k_v1pt5")]
# Qwen-Image-2512 (Dec 2025 refresh) shares the AutoencoderKLQwenImage and uses the
# same PiD student as Qwen-Image — only the transformer/text-encoder differ.
PID_CHECKPOINT_REGISTRY[("qwenimage-2512", "2kto4k_v1pt5")] = PID_CHECKPOINT_REGISTRY[("qwenimage", "2kto4k_v1pt5")]
# FLUX.2-klein (Flux2KleinPipeline / FLUX.2-klein-4B & -9B) shares the AutoencoderKLFlux2
# and uses the same PiD student as Flux2 — only the transformer/text-encoder differ.
# Both size variants alias to the same flux2 decoders.
for _klein in ("flux2-klein-4b", "flux2-klein-9b"):
    PID_CHECKPOINT_REGISTRY[(_klein, "2k")] = PID_CHECKPOINT_REGISTRY[("flux2", "2k")]
    PID_CHECKPOINT_REGISTRY[(_klein, "2kto4k_v1pt5")] = PID_CHECKPOINT_REGISTRY[("flux2", "2kto4k_v1pt5")]


def get_pid_checkpoint(backbone: str, ckpt_type: str = "2k") -> PIDCheckpoint:
    """Return the registered official PID checkpoint for `(backbone, ckpt_type)`.

    `ckpt_type` defaults to `"2k"` so existing call sites keep their pre-2kto4k
    behavior. Raises KeyError with the list of valid keys when the pair is
    unknown — typical cause is asking for a 4K variant of a backbone
    that doesn't ship one (dinov2 / siglip).
    """
    if ckpt_type not in VALID_CKPT_TYPES:
        raise KeyError(f"Unknown ckpt_type {ckpt_type!r}. Valid: {VALID_CKPT_TYPES}")
    try:
        return PID_CHECKPOINT_REGISTRY[(backbone, ckpt_type)]
    except KeyError as exc:
        valid = ", ".join(sorted(f"{b}+{t}" for b, t in PID_CHECKPOINT_REGISTRY))
        raise KeyError(f"Unknown (backbone, ckpt_type)=({backbone!r}, {ckpt_type!r}). Valid: {valid}") from exc


__all__ = [
    "PIDCheckpoint",
    "PID_CHECKPOINT_REGISTRY",
    "VALID_CKPT_TYPES",
    "get_pid_checkpoint",
]
