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
2D LQ projection for pixel-space image super-resolution.

Takes LQ VAE latent [B, z_dim, zH, zW], projects them to patch-aligned tokens
for injection into the PixDiT_T2I transformer.

Supports two gate types for ControlNet-style injection:
  "sigma_aware_per_token":         x + sigmoid(Linear([x, lq]) - exp(log_alpha)*sigma) * lq  (B,N,1)
  "sigma_aware_per_token_per_dim": x + sigmoid(Linear([x, lq]) - exp(log_alpha)*sigma) * lq  (B,N,D)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_CONV_PADDING_MODES = {"zeros", "reflect", "replicate"}


def _validate_conv_padding_mode(conv_padding_mode: str) -> None:
    if conv_padding_mode not in _VALID_CONV_PADDING_MODES:
        raise ValueError(
            f"conv_padding_mode must be one of {sorted(_VALID_CONV_PADDING_MODES)}, got {conv_padding_mode!r}"
        )


# ---------------------------------------------------------------------------
# Gate modules
# ---------------------------------------------------------------------------


class SigmaAwarePerTokenGate(nn.Module):
    """Per-token scalar sigma-aware gate. Used in PiD v1.5, reducing gate parameter.

    Init: content_proj.bias=2.0, log_alpha=log(5) →
          gate ≈ sigmoid(2.0 - 5*sigma): ~0.88 at sigma=0, ~0.5 at sigma=0.4, ~0.05 at sigma=1.
    """

    def __init__(self, dim: int):
        super().__init__()
        import math

        self.content_proj = nn.Linear(dim * 2, 1)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def compute_gate_scalar(
        self, x: torch.Tensor, lq: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert sigma is not None, "SigmaAwarePerTokenGate requires degrade_sigma input"
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))  # (B, N, 1)
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)  # (B, 1, 1)
        return torch.sigmoid(content_logit + sigma_offset)  # (B, N, 1)

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.compute_gate_scalar(x, lq, sigma) * lq


class SigmaAwarePerTokenAndDimGate(nn.Module):
    """Per-token per-dim sigma-aware gate. Used in PiD v1.

    Content branch projects to dim instead of 1, so the gate is independent per
    (token, channel) instead of shared across channels. Sigma branch stays scalar
    per sample and broadcasts (B, 1, 1) → (B, N, D).
    """

    def __init__(self, dim: int):
        super().__init__()
        import math

        self.content_proj = nn.Linear(dim * 2, dim)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def compute_gate_scalar(
        self, x: torch.Tensor, lq: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert sigma is not None, "SigmaAwarePerTokenAndDimGate requires degrade_sigma input"
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))  # (B, N, D)
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)  # (B, 1, 1)
        return torch.sigmoid(content_logit + sigma_offset)  # (B, N, D)

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.compute_gate_scalar(x, lq, sigma) * lq


def _build_gate(gate_type: str, dim: int, zero_init: bool = True) -> nn.Module:
    # zero_init is intentionally not forwarded to gate constructors: gate zero-init is
    # redundant when output_heads is zero-init (lq=0 already kills the injection term)
    # and creates a gradient dead zone if both are zero-init simultaneously.
    if gate_type == "sigma_aware_per_token":
        return SigmaAwarePerTokenGate(dim)
    elif gate_type == "sigma_aware_per_token_per_dim":
        return SigmaAwarePerTokenAndDimGate(dim)
    else:
        raise ValueError(
            f"Unknown gate_type: {gate_type!r}. Must be one of 'sigma_aware_per_token', 'sigma_aware_per_token_and_dim'."
        )


class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip."""

    def __init__(self, channels: int, num_groups: int = 4, conv_padding_mode: str = "zeros"):
        super().__init__()
        _validate_conv_padding_mode(conv_padding_mode)
        self.conv_padding_mode = conv_padding_mode
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# LQ Projection 2D
# ---------------------------------------------------------------------------


class LQProjection2D(nn.Module):
    """2D LQ projection for image super-resolution in pixel space.

    Spatial alignment strategy (lossless, no bilinear interpolation):

    Image branch:
      LQ image is at H_lq = H_hq / sr_scale. Patch grid is pH = H_hq / patch_size.
      Ratio = H_lq / pH = patch_size / sr_scale.
      - If ratio >= 1 (LQ res >= patch grid): PixelUnshuffle(ratio) to fold spatial
        dims into channels. E.g. sr_scale=4, ps=16: ratio=4, unshuffle folds 4x4 pixels
        into channels: [B, 3, 256, 256] → [B, 3*16, 64, 64] = [B, 48, 64, 64].
      - If ratio < 1 (LQ res < patch grid): Conv2d with PixelShuffle to upsample.

    Latent branch:
      LQ latent is at zH = H_lq / lsdf. Patch grid is pH = H_hq / patch_size.
      Optional latent_unpatchify_factor moves patchified latent channels back to
      spatial dims first. For Flux2 normalized latents, factor=2 converts
      [B, 128, H/16, W/16] -> [B, 32, H/8, W/8] without BN inverse normalization.
      z_patch_ratio = pH / zH = (sr_scale * effective_lsdf) / patch_size,
      where effective_lsdf = latent_spatial_down_factor / latent_unpatchify_factor.
      - If z_patch_ratio <= 1 (latent res >= patch grid): fold z_patch_ratio×z_patch_ratio
        spatial elements into channels (same as FastPixelDecoder._align_z_to_patch_grid).
      - If z_patch_ratio > 1 (latent res < patch grid): nearest interpolate to upsample.

    Args:
        in_channels: LQ image channels (3 for RGB, 0 to disable image branch).
        latent_channels: LQ latent channels (e.g. 16 for Wan VAE, 0 to disable).
        hidden_dim: internal feature dimension for conv processing.
        out_dim: output dimension (must match transformer hidden_size).
        patch_size: spatial patch size of the transformer (e.g. 16).
        sr_scale: super-resolution scale factor (LQ is sr_scale times smaller).
        latent_spatial_down_factor: VAE spatial downscale factor (default 8).
        latent_unpatchify_factor: optional spatial unpatchify factor for patchified
            latents. 1 disables it. Flux2 normalized latents should use 2.
        num_res_blocks: number of ResBlocks appended after the initial conv pair in each
            branch (image / latent / merge). 0 = shallow (Conv-SiLU-Conv only). Must
            stay at 4 to load checkpoints saved before May-2026 (state-dict keys at
            `image_conv.{3..6}.*`, `latent_proj.{3..6}.*`, `merge.{2..5}.*`).
        num_outputs: number of output feature sets.
            1 = single output (for token_add mode).
            N = one per transformer block (for controlnet mode).
        gate_type: "sigma_aware_per_token" | "sigma_aware_per_token_and_dim".
        interval: inject every N blocks (only relevant when num_outputs > 1).
        zero_init: if True, zero-init all output projections for safe pretrained start.
        conv_padding_mode: padding mode for all Conv2d layers in image / latent / merge branches.
        pit_output: if True, add a dedicated output head for PiT block injection.
            The PiT head output is appended as the last element of forward() output.
        lq_aux_rgb_head: if True, add a training-only head that reconstructs the
            LQ RGB image from the shared patch-grid hidden features.
        lq_aux_rgb_head_latent_block_idx: 1-based ResBlock index inside latent_proj
            where the aux RGB head should tap features. -1 keeps the old behaviour:
            use the final merged/shared hidden feature. Positive values are only
            supported for latent-only LQProjection2D.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 0,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: int = 16,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        latent_unpatchify_factor: int = 1,
        num_res_blocks: int = 4,
        num_outputs: int = 1,
        gate_type: str = "sigma_aware_per_token_and_dim",
        interval: int = 1,
        zero_init: bool = True,
        conv_padding_mode: str = "zeros",
        pit_output: bool = False,
        lq_aux_rgb_head: bool = False,
        lq_aux_rgb_head_latent_block_idx: int = -1,
    ):
        super().__init__()
        assert in_channels > 0 or latent_channels > 0, "At least one of in_channels or latent_channels must be > 0"

        _validate_conv_padding_mode(conv_padding_mode)

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.sr_scale = sr_scale
        self.latent_spatial_down_factor = latent_spatial_down_factor
        self.latent_unpatchify_factor = latent_unpatchify_factor
        self.num_res_blocks = num_res_blocks
        if latent_unpatchify_factor > 1 and latent_spatial_down_factor % latent_unpatchify_factor != 0:
            raise ValueError(
                "latent_spatial_down_factor must be divisible by latent_unpatchify_factor, got "
                f"{latent_spatial_down_factor} and {latent_unpatchify_factor}."
            )
        self.effective_latent_spatial_down_factor = latent_spatial_down_factor // latent_unpatchify_factor
        self.num_outputs = num_outputs
        self.interval = interval
        self.zero_init = zero_init
        self.conv_padding_mode = conv_padding_mode
        self.pit_output = pit_output
        self.lq_aux_rgb_head_enabled = bool(lq_aux_rgb_head)
        self.lq_aux_rgb_head_latent_block_idx = lq_aux_rgb_head_latent_block_idx
        self.lq_aux_patch_factor = patch_size // sr_scale if lq_aux_rgb_head else 0

        # --- Image branch (not used in PiD) ---
        # PixelUnshuffle → Conv proj → ResBlocks for deep feature extraction
        if in_channels > 0:
            assert patch_size >= sr_scale and patch_size % sr_scale == 0, (
                f"patch_size ({patch_size}) must be >= sr_scale ({sr_scale}) and divisible"
            )
            self.image_unshuffle_factor = patch_size // sr_scale
            unshuffle_ch = in_channels * self.image_unshuffle_factor**2
            layers = [
                nn.Conv2d(
                    unshuffle_ch,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=conv_padding_mode,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=conv_padding_mode,
                ),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.image_conv = nn.Sequential(*layers)
        else:
            self.image_conv = None
            self.image_unshuffle_factor = 0

        # --- Latent branch (Used in PiD) ---
        # Spatial alignment (fold / upsample) → Conv proj
        if latent_channels > 0:
            if latent_unpatchify_factor > 1 and latent_channels % (latent_unpatchify_factor**2) != 0:
                raise ValueError(
                    "latent_channels must be divisible by latent_unpatchify_factor**2, got "
                    f"{latent_channels} and {latent_unpatchify_factor}."
                )
            effective_latent_channels = latent_channels // (latent_unpatchify_factor**2)
            z_to_patch_ratio = (sr_scale * self.effective_latent_spatial_down_factor) / patch_size
            self.z_to_patch_ratio = z_to_patch_ratio

            if z_to_patch_ratio > 1:
                # Latent is lower res than patch grid → nearest upsample (no learnable params).
                # LearnedLatentUpsampler (PixelShuffle) caused DDP numerical issues on multi-node.
                self.latent_upsampler = None
                self.latent_upsample_ratio = int(z_to_patch_ratio)
                latent_proj_in_ch = effective_latent_channels
            elif z_to_patch_ratio == 1:
                self.latent_upsampler = None
                latent_proj_in_ch = effective_latent_channels
            else:
                fold_factor = int(1 / z_to_patch_ratio)
                assert fold_factor * z_to_patch_ratio == 1.0, (
                    f"fold_factor {fold_factor} * z_to_patch_ratio {z_to_patch_ratio} != 1"
                )
                self.latent_upsampler = None
                self.latent_fold_factor = fold_factor
                latent_proj_in_ch = effective_latent_channels * fold_factor**2

            layers = [
                nn.Conv2d(
                    latent_proj_in_ch,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=conv_padding_mode,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=conv_padding_mode,
                ),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.latent_proj = nn.Sequential(*layers)
        else:
            self.latent_proj = None
            self.z_to_patch_ratio = 0
            self.latent_upsampler = None

        # --- Merge + shared ResBlocks (if both branches active) ---
        if in_channels > 0 and latent_channels > 0:
            layers = [
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, padding_mode=conv_padding_mode),
                nn.SiLU(),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
            self.merge = nn.Sequential(*layers)
        else:
            self.merge = None

        # --- Output heads ---
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_outputs)])

        # --- Dedicated PiT output head (separate from DiT heads) ---
        if pit_output:
            self.pit_head = nn.Linear(hidden_dim, out_dim)
        else:
            self.pit_head = None

        if lq_aux_rgb_head:
            self.lq_aux_rgb_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3 * self.lq_aux_patch_factor * self.lq_aux_patch_factor),
            )
        else:
            self.lq_aux_rgb_head = None

        # --- Gate modules (one per injection point, for controlnet-style injection) ---
        # Using a ModuleList instead of a single shared module allows each block to learn
        # independent gating behaviour (different content_proj weights and log_alpha).
        self.gate_modules = nn.ModuleList(
            [_build_gate(gate_type, out_dim, zero_init=zero_init) for _ in range(num_outputs)]
        )

    def init_weights(self):
        """Initialize weights. Zero-init output heads when zero_init=True.

        Conv layers use truncated normal (std=0.02) instead of kaiming_normal_
        to keep intermediate activations small under bfloat16 autocast.
        With zero-init output heads the forward output is zero regardless of
        conv init scale, but large conv activations cause grad overflow in
        bfloat16 backward (output_head.weight.grad ∝ conv_features).
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for head in self.output_heads:
            if self.zero_init:
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
            else:
                # Small init so LQ signal is present from the start but doesn't
                # overwhelm the pretrained base model.
                nn.init.trunc_normal_(head.weight, std=0.02)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

        # PiT head follows same init strategy
        if self.pit_head is not None:
            if self.zero_init:
                nn.init.zeros_(self.pit_head.weight)
                if self.pit_head.bias is not None:
                    nn.init.zeros_(self.pit_head.bias)
            else:
                nn.init.trunc_normal_(self.pit_head.weight, std=0.02)
                if self.pit_head.bias is not None:
                    nn.init.zeros_(self.pit_head.bias)

        if self.lq_aux_rgb_head is not None:
            for module in self.lq_aux_rgb_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def is_gate_active(self, block_idx: int) -> bool:
        """Whether gate() should be called for this block index."""
        if self.interval > 1:
            return block_idx % self.interval == 0
        return True

    def _get_output_index(self, block_idx: int) -> int:
        """Map block_idx to output head index, respecting interval."""
        if self.interval > 1:
            return block_idx // self.interval
        return block_idx

    def gate(
        self, x: torch.Tensor, lq: torch.Tensor, sigma: Optional[torch.Tensor] = None, out_idx: int = 0
    ) -> torch.Tensor:
        """Apply gating: inject lq features into transformer hidden state x."""
        return self.gate_modules[out_idx](x, lq, sigma=sigma)

    def _align_image_to_patch_grid(
        self, lq_video_or_image: torch.Tensor, target_pH: int, target_pW: int
    ) -> torch.Tensor:
        """Align LQ image to patch grid via PixelUnshuffle.

        [B, C, H_lq, W_lq] → pad if needed → PixelUnshuffle(factor) → [B, C*f*f, pH, pW]
        Then conv to [B, hidden_dim, pH, pW].

        Multi-AR images may have H_lq not divisible by unshuffle_factor. We pad to
        target_pH * f, target_pW * f to ensure exact alignment with the patch grid.
        """
        f = self.image_unshuffle_factor
        B, C, H_lq, W_lq = lq_video_or_image.shape
        target_H_lq = target_pH * f
        target_W_lq = target_pW * f

        # Pad or crop to exact target size if needed (multi-AR may not align perfectly)
        if H_lq != target_H_lq or W_lq != target_W_lq:
            lq_video_or_image = F.interpolate(
                lq_video_or_image, size=(target_H_lq, target_W_lq), mode="bilinear", align_corners=False
            )

        x = F.pixel_unshuffle(lq_video_or_image, f)  # [B, C*f*f, target_pH, target_pW]
        return self.image_conv(x)  # [B, hidden_dim, target_pH, target_pW]

    def _unpatchify_latent_if_needed(self, lq_latent: torch.Tensor) -> torch.Tensor:
        f = self.latent_unpatchify_factor
        if f == 1:
            return lq_latent

        B, C, H, W = lq_latent.shape
        if C != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} LQ latent channels, got {C}.")
        if C % (f * f) != 0:
            raise ValueError(f"LQ latent channels ({C}) must be divisible by unpatchify factor squared ({f * f}).")

        # Flux2-style normalized latent unpatchify:
        # [B, C*f*f, H, W] -> [B, C, H*f, W*f]. No BN inverse normalization.
        lq_latent = lq_latent.reshape(B, C // (f * f), f, f, H, W)
        lq_latent = lq_latent.permute(0, 1, 4, 2, 5, 3).contiguous()
        return lq_latent.reshape(B, C // (f * f), H * f, W * f)

    def _align_latent_spatial_to_patch_grid(self, lq_latent: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        """Align LQ latent to patch grid via nearest interpolate or fold.

        Returns [B, latent_proj_in_ch, pH, pW].
        """
        lq_latent = self._unpatchify_latent_if_needed(lq_latent)
        B, z_dim = lq_latent.shape[:2]

        if self.z_to_patch_ratio > 1:
            # Upsample: latent is lower res than patch grid → nearest interpolate
            z_aligned = F.interpolate(lq_latent, size=(pH, pW), mode="nearest")
        elif self.z_to_patch_ratio == 1:
            z_aligned = lq_latent
            if z_aligned.shape[2] != pH or z_aligned.shape[3] != pW:
                z_aligned = F.interpolate(z_aligned, size=(pH, pW), mode="bilinear", align_corners=False)
        else:
            # Fold: latent is higher res than patch grid
            f = self.latent_fold_factor
            # Ensure latent spatial matches expected fold size
            zH_expected, zW_expected = pH * f, pW * f
            if lq_latent.shape[2] != zH_expected or lq_latent.shape[3] != zW_expected:
                lq_latent = F.interpolate(
                    lq_latent, size=(zH_expected, zW_expected), mode="bilinear", align_corners=False
                )
            z_aligned = lq_latent.reshape(B, z_dim, pH, f, pW, f)
            z_aligned = z_aligned.permute(0, 1, 3, 5, 2, 4)
            z_aligned = z_aligned.reshape(B, z_dim * f * f, pH, pW)

        return z_aligned

    def _run_latent_proj(
        self, z_aligned: torch.Tensor, capture_aux_feature: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run latent_proj and optionally tap a 1-based ResBlock output for aux RGB supervision."""
        if not capture_aux_feature:
            return self.latent_proj(z_aligned), None

        aux_feature = None
        res_block_idx = 0
        x = z_aligned
        for layer in self.latent_proj:
            x = layer(x)
            if isinstance(layer, ResBlock):
                res_block_idx += 1
                if res_block_idx == self.lq_aux_rgb_head_latent_block_idx:
                    aux_feature = x

        if aux_feature is None:
            raise RuntimeError(
                "Failed to capture latent aux RGB feature. Check "
                f"lq_aux_rgb_head_latent_block_idx={self.lq_aux_rgb_head_latent_block_idx} "
                f"against num_res_blocks={self.num_res_blocks}."
            )
        return x, aux_feature

    def _align_latent_to_patch_grid(self, lq_latent: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        """Align LQ latent to patch grid and project it to [B, hidden_dim, pH, pW]."""
        z_aligned = self._align_latent_spatial_to_patch_grid(lq_latent, pH, pW)
        latent_feature, _ = self._run_latent_proj(z_aligned, capture_aux_feature=False)
        return latent_feature

    def _predict_lq_rgb(self, tokens: torch.Tensor, target_pH: int, target_pW: int) -> torch.Tensor:
        if self.lq_aux_rgb_head is None:
            raise RuntimeError("lq_aux_rgb_head is disabled.")
        patch_factor = self.lq_aux_patch_factor
        bsz = tokens.shape[0]
        rgb_patch = self.lq_aux_rgb_head(tokens).type_as(tokens)
        rgb_patch = rgb_patch.view(bsz, target_pH, target_pW, 3, patch_factor, patch_factor)
        rgb = rgb_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        return rgb.view(bsz, 3, target_pH * patch_factor, target_pW * patch_factor)

    def forward(
        self,
        lq_video_or_image: Optional[torch.Tensor] = None,
        lq_latent: Optional[torch.Tensor] = None,
        target_pH: int = 0,
        target_pW: int = 0,
        return_aux: bool = False,
    ) -> List[torch.Tensor]:
        """Project LQ inputs to patch-aligned token features.

        Args:
            lq_video_or_image: [B, C, H_lq, W_lq] LQ image at original low resolution. Can be None.
            lq_latent: [B, z_dim, zH, zW] LQ VAE latent. Can be None.
            target_pH: target patch grid height (H_hq / patch_size).
            target_pW: target patch grid width (W_hq / patch_size).
            return_aux: if True, return (outputs, aux). aux contains
                "pred_lq_rgb" when lq_aux_rgb_head is enabled.

        Returns:
            List of [B, N, out_dim] tensors where N = target_pH * target_pW.
            Length = num_outputs (+ 1 if pit_output=True).
        """
        assert target_pH > 0 and target_pW > 0, "Must provide target_pH and target_pW"
        features = []
        aux_rgb_feature = None

        # Image branch: PixelUnshuffle → Conv
        if self.image_conv is not None and lq_video_or_image is not None:
            features.append(self._align_image_to_patch_grid(lq_video_or_image, target_pH, target_pW))

        # Latent branch: Fold/Upsample → Conv
        if self.latent_proj is not None and lq_latent is not None:
            capture_aux_feature = return_aux and self.lq_aux_rgb_head_latent_block_idx > 0
            if capture_aux_feature:
                z_aligned = self._align_latent_spatial_to_patch_grid(lq_latent, target_pH, target_pW)
                latent_feature, aux_rgb_feature = self._run_latent_proj(
                    z_aligned, capture_aux_feature=capture_aux_feature
                )
            else:
                latent_feature = self._align_latent_to_patch_grid(lq_latent, target_pH, target_pW)
            features.append(latent_feature)

        # Merge or select single branch
        if len(features) == 2 and self.merge is not None:
            merged = self.merge(torch.cat(features, dim=1))  # [B, hidden_dim, pH, pW]
        elif len(features) == 1:
            merged = features[0]
        else:
            # Both inputs are None — return zero features
            ref = lq_video_or_image if lq_video_or_image is not None else lq_latent
            if ref is None:
                raise ValueError("LQProjection2D requires at least one LQ input or a reference tensor.")
            B, device, dtype = ref.shape[0], ref.device, ref.dtype
            N = target_pH * target_pW
            num_total = self.num_outputs + (1 if self.pit_output else 0)
            outputs = [torch.zeros(B, N, self.out_dim, device=device, dtype=dtype) for _ in range(num_total)]
            if return_aux:
                aux = None
                if self.lq_aux_rgb_head is not None:
                    aux_h = target_pH * self.lq_aux_patch_factor
                    aux_w = target_pW * self.lq_aux_patch_factor
                    aux = {"pred_lq_rgb": torch.zeros(B, 3, aux_h, aux_w, device=device, dtype=dtype)}
                return outputs, aux
            return outputs

        # Flatten to tokens: [B, hidden_dim, pH, pW] -> [B, N, hidden_dim]
        tokens = merged.flatten(2).transpose(1, 2)

        # Project through output heads
        outputs = [head(tokens) for head in self.output_heads]

        # Append dedicated PiT head output as last element
        if self.pit_head is not None:
            outputs.append(self.pit_head(tokens))

        if return_aux:
            aux = None
            if self.lq_aux_rgb_head is not None:
                if aux_rgb_feature is not None:
                    aux_tokens = aux_rgb_feature.flatten(2).transpose(1, 2)
                else:
                    aux_tokens = tokens
                aux = {"pred_lq_rgb": self._predict_lq_rgb(aux_tokens, target_pH, target_pW)}
            return outputs, aux
        return outputs
