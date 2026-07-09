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

from __future__ import annotations

import attrs
import torch
import torch.nn.functional as F


@attrs.define(slots=False)
class TrainDegradationConfig:
    # PixelDiT SR only needs a clean, deterministic LQ image: bicubic
    # downsampled from the HQ target.
    downscale: float = 4.0


def get_simple_downsample_size(
    height: int,
    width: int,
    downscale: float,
) -> tuple[int, int]:
    """Return the legacy simple_downsample target size for an HQ image."""
    target_h = int(height / downscale)
    target_w = int(width / downscale)

    return target_h, target_w


def simple_downsample_image(image: torch.Tensor, downscale: float) -> torch.Tensor:
    """Bicubic downsample an NCHW image tensor in normalized [-1, 1] range."""
    if image.ndim != 4:
        raise ValueError(f"simple_downsample_image expects [B, C, H, W], got shape {tuple(image.shape)}.")

    height, width = image.shape[-2:]
    target_size = get_simple_downsample_size(height, width, downscale)
    lq_image = F.interpolate(
        image.contiguous(),
        size=target_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return lq_image.clamp(-1.0, 1.0)


__all__ = [
    "TrainDegradationConfig",
    "get_simple_downsample_size",
    "simple_downsample_image",
]
