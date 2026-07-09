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
Discriminator networks for DMD2 Projected GAN / APT-style GAN loss.

Uses teacher intermediate features as the discrimination space (not pixel/latent space).
The discriminator is a lightweight head that operates on unpatchified features
from specific transformer blocks of the teacher network.

Reference: FastGen's discriminators.py
"""

from typing import List, Optional, Set

import torch
from torch import nn


class Discriminator(torch.nn.Module):
    """Base class for Discriminators operating on teacher intermediate features."""

    def __init__(self, feature_indices: Optional[Set[int]] = None):
        super().__init__()
        self.feature_indices = feature_indices

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")


def _get_optimal_groups(num_channels: int) -> int:
    """Calculate optimal number of groups for GroupNorm."""
    if num_channels <= 32:
        groups = max(1, num_channels // 4)
    else:
        groups = 32
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
    assert num_channels % groups == 0, f"{num_channels} not divisible by {groups}"
    return groups


def _build_dit_simple_conv3d_discriminator_head(
    inner_dim: int,
    kernel_size=(2, 4, 4),
    stride=(2, 2, 2),
    padding=(0, 1, 1),
) -> nn.Sequential:
    """Simple 2-layer Conv3D discriminator head (~1M params).

    Args:
        inner_dim: Input channel dimension (e.g. 384 for Wan 1.3B).
        kernel_size: Kernel size for the first conv3d layer.
        stride: Stride for the first conv3d layer.
        padding: Padding for the first conv3d layer.
    """
    hidden_channels = inner_dim // 2
    return nn.Sequential(
        nn.Conv3d(
            kernel_size=kernel_size,
            in_channels=inner_dim,
            out_channels=hidden_channels,
            stride=stride,
            padding=padding,
        ),
        nn.GroupNorm(num_groups=_get_optimal_groups(hidden_channels), num_channels=hidden_channels),
        nn.LeakyReLU(0.2),
        nn.Conv3d(kernel_size=1, in_channels=hidden_channels, out_channels=1, stride=1, padding=0),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
    )


class Discriminator_VideoDiT(Discriminator):
    """
    Discriminator for video features from video diffusion models (DiT, Wan, etc.).

    Supports multiple architectures with different computational characteristics.
    Operates on unpatchified teacher intermediate features [B, inner_dim, T, H, W].

    Default config for Wan 1.3B: num_blocks=30, inner_dim=384 (1536//4), disc_type="dit_simple_conv3d"

    Available architectures:
    - dit_simple_conv3d: Simple 2-layer conv3d (~1M params, recommended starting point)
    """

    ARCHITECTURES = {
        "dit_simple_conv3d": {
            "type": "dit_simple_conv3d",
            "kernel_size": (2, 4, 4),
            "stride": (2, 2, 2),
            "padding": (0, 1, 1),
        },
    }

    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        num_blocks: int = 30,
        disc_type: str = "dit_simple_conv3d",
        inner_dim: int = 384,
    ):
        """
        Args:
            feature_indices: Which block indices to extract features from. Defaults to middle block.
            num_blocks: Total number of transformer blocks in the teacher.
            disc_type: Architecture type. See ARCHITECTURES.
            inner_dim: Input channel dim = teacher hidden_dim // (patch_h * patch_w).
                       Wan 1.3B: 1536 // 4 = 384
        """
        super().__init__(feature_indices=feature_indices)

        if self.feature_indices is None:
            self.feature_indices = {int(num_blocks // 2)}
        self.feature_indices = {i for i in self.feature_indices if i < num_blocks}
        self.num_features = len(self.feature_indices)
        self.disc_type = disc_type
        self.inner_dim = inner_dim

        if disc_type not in self.ARCHITECTURES:
            available = ", ".join(self.ARCHITECTURES.keys())
            raise ValueError(f"Unknown disc_type '{disc_type}'. Available: {available}")

        config = self.ARCHITECTURES[disc_type]

        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            head = self._build_discriminator_head(config, inner_dim)
            self.cls_pred_heads.append(head)

    def _build_discriminator_head(self, config: dict, inner_dim: int) -> nn.Module:
        arch_type = config["type"]
        if arch_type == "dit_simple_conv3d":
            return _build_dit_simple_conv3d_discriminator_head(
                inner_dim=inner_dim,
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: List of unpatchified feature tensors, each [B, inner_dim, T, H, W].
                   Length must match num_features (= len(feature_indices)).

        Returns:
            Concatenated logits [B, num_features] for discrimination.
        """
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}"
            )

        all_logits = []
        for head, feat in zip(self.cls_pred_heads, feats):
            logits = head(feat)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)


class Discriminator_ImageDiT(Discriminator):
    """Discriminator for image features from image DiT models (e.g., Flux, PixDiT).

    Default architecture (num_conv_layers=2, patch_logits=False) is the lightweight
    2-layer Conv2D head (~0.5M params per head): one stride-2 downsample conv +
    1x1 channel-reduce conv + global avg pool → scalar logit per head.

    Tunable via kwargs:
      - num_conv_layers: controls head depth. N means (N-1) stride-2 downsample
        convs (each halving channels) followed by a final 1x1 conv → 1 channel.
        N=2 is the default shallow head; N=3/4 gives "deep head" variants.
      - patch_logits: if True, omit the global avg pool — disc emits PatchGAN-style
        spatial logits ([B, Hs*Ws] after flatten), giving finer-grained adversarial
        signal. softplus(±logits).mean() and R1 MSE naturally accept either shape.

    For PixelDiT SR with patch_depth=14 (feature_indices tap patch_blocks only):
        inner_dim=1152, num_blocks=14, feature_indices={13}
    """

    def __init__(
        self,
        feature_indices: Optional[Set[int]] = None,
        num_blocks: int = 57,
        inner_dim: int = 3072,
        num_conv_layers: int = 2,
        patch_logits: bool = False,
    ):
        """
        Args:
            feature_indices: Block indices to extract features from. Defaults to middle block.
            num_blocks: Total patch_blocks that can be tapped by feature_indices.
            inner_dim: Feature channel dimension (= teacher hidden_size for image DiTs).
            num_conv_layers: Total Conv2D layers per head (>=2). See class docstring.
            patch_logits: If True, skip global avg pool and emit PatchGAN spatial logits.
        """
        super().__init__(feature_indices=feature_indices)

        if self.feature_indices is None:
            self.feature_indices = {int(num_blocks // 2)}
        self.feature_indices = {i for i in self.feature_indices if i < num_blocks}
        self.num_features = len(self.feature_indices)
        self.inner_dim = inner_dim
        assert num_conv_layers >= 2, f"num_conv_layers must be >= 2, got {num_conv_layers}"
        self.num_conv_layers = num_conv_layers
        self.patch_logits = patch_logits

        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            self.cls_pred_heads.append(self._build_head(inner_dim, num_conv_layers, patch_logits))

    @staticmethod
    def _build_head(inner_dim: int, num_conv_layers: int, patch_logits: bool) -> nn.Sequential:
        layers: list = []
        ch_in = inner_dim
        ch_out = inner_dim // 2
        for _ in range(num_conv_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(num_groups=_get_optimal_groups(ch_out), num_channels=ch_out),
                    nn.LeakyReLU(0.2),
                ]
            )
            ch_in = ch_out
            ch_out = max(ch_out // 2, 64)
        layers.append(nn.Conv2d(ch_in, 1, kernel_size=1, stride=1, padding=0))
        if not patch_logits:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: List of feature tensors, each [B, inner_dim, H, W].
                   Length must match num_features (= len(feature_indices)).

        Returns:
            Concatenated logits: [B, num_features] when patch_logits=False, or
            [B, num_features * Hs * Ws] when patch_logits=True.
        """
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats)} with length {len(feats) if isinstance(feats, list) else 'N/A'}"
            )

        all_logits = []
        for head, feat in zip(self.cls_pred_heads, feats):
            all_logits.append(head(feat))

        return torch.cat(all_logits, dim=1)
