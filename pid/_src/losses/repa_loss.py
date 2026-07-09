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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PixelDiTREPALoss(nn.Module):
    """REPA loss adapted for PixelDiT's MMDiTBlockT2I blocks.

    MMDiTBlockT2I.forward returns (x, y) tuple; the hook captures the image stream x.
    We register a hook on patch_blocks[i_extract] and project features to DINOv2 space.
    """

    def __init__(self, net: nn.Module, i_extract: int = 6, n_layers: int = 2, cache_dir=None, **kwargs):
        super().__init__()
        features_dim = net.hidden_size

        # Frozen DINOv2 feature extractor
        from pid._src.networks.repa.feature_encoder import DinoEncoder, Frozen

        self.features_extractor = Frozen(DinoEncoder(cache_dir=cache_dir), allow_grad=False)

        # Projection MLP: hidden_size -> 768 (DINOv2 dim)
        self.repa_mlp = nn.Sequential()
        self.repa_loss_fn = nn.CosineSimilarity(dim=2, eps=1e-5)
        dino_dim = self.features_extractor.module.out_dim  # 768
        for i in range(n_layers):
            in_dim = features_dim
            out_dim = features_dim if i < n_layers - 1 else dino_dim
            self.repa_mlp.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                self.repa_mlp.append(nn.SiLU())

        # Register hook on the specified patch block.
        # MMDiTBlockT2I returns (x, y); we capture the image-stream tensor x.
        i_extract = min(i_extract, len(net.patch_blocks) - 1)
        net.patch_blocks[i_extract].register_forward_hook(self._hook_repa)
        self._repa_layer_output = None

    def _hook_repa(self, module, input, output):
        if self.training:
            self._repa_layer_output = output[0] if isinstance(output, tuple) else output

    def forward(self, x_gt: Tensor) -> Tensor:
        """Compute REPA loss. Call after model forward, after the hook has captured features."""
        repa_val = self._repa_layer_output
        if repa_val is None:
            return torch.tensor(0.0, device=x_gt.device)

        # Ensure 1D tokens: [B, L, C]
        if repa_val.dim() == 4:
            B, C, H, W = repa_val.shape
            repa_val = repa_val.permute(0, 2, 3, 1).reshape(B, H * W, C)

        repa_val = self.repa_mlp(repa_val.float())

        with torch.no_grad():
            repa_ref = self.features_extractor(x_gt.float(), target_n_tokens=repa_val.shape[1])

        # Handle spatial size mismatch via interpolation.
        B, Tu, Cu = repa_val.shape
        _, Td, Cd = repa_ref.shape
        if Tu != Td:
            h_u = int(Tu**0.5)
            h_d = int(Td**0.5)
            if h_u * h_u == Tu and h_d * h_d == Td:
                if Td > Tu:
                    dino_2d = repa_ref.permute(0, 2, 1).reshape(B, Cd, h_d, h_d)
                    dino_resized = F.interpolate(dino_2d, size=(h_u, h_u), mode="bilinear", align_corners=False)
                    repa_ref = dino_resized.flatten(2).permute(0, 2, 1)
                else:
                    usit_2d = repa_val.permute(0, 2, 1).reshape(B, Cu, h_u, h_u)
                    usit_resized = F.interpolate(usit_2d, size=(h_d, h_d), mode="bilinear", align_corners=False)
                    repa_val = usit_resized.flatten(2).permute(0, 2, 1)

        repa_val = F.normalize(repa_val, dim=-1)
        repa_ref = F.normalize(repa_ref, dim=-1)

        self._repa_layer_output = None
        with torch.autocast("cuda", enabled=False):
            return 1 - self.repa_loss_fn(repa_val.to(torch.float32), repa_ref.to(torch.float32)).mean()
