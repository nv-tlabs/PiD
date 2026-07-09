# SSDD losses: DinoEncoder (frozen DINOv2 for REPA), REPALoss (representation alignment),
# SSDDLosses (container for optional REPA + LPIPS), GanLoss (wraps NLayerDiscriminator).
#
# Changes from SSDD source:
# - Removed accelerator parameter from Frozen and SSDDLosses (Imaginaire handles distribution)
# - Uses lazy import for transformers (REPA only needed when enabled)
# - Keeps lpips library (original SSDD was validated with it)
# - Removed checkpoint loading from SSDDLosses init (handled by Imaginaire)
#
# Original source: SSDD/ssdd/models/ssdd/losses.py
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from torchvision.transforms import Normalize

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def mark_initialized(model):
    for m in model.modules():
        m._w_init = True


def freeze_model(model, freeze=True, mark_init="auto"):
    for param in model.parameters():
        param.requires_grad = not freeze
    if mark_init == "auto":
        mark_init = freeze
    if mark_init:
        mark_initialized(model)


class Frozen(nn.Module):
    """Wraps a pre-trained module and freezes it.
    If allow_grad is True, gradients can be computed through the module, but weights are not updated.
    """

    def __init__(self, module, allow_grad=True):
        super().__init__()
        freeze_model(module)
        self._module = (module.eval(),)
        self.allow_grad = allow_grad

    def __getattr__(self, name):
        if name == "module":
            return self._module[0]
        return getattr(self._module[0], name)

    def _apply(self, fn, recurse=True):
        # Propagate .to() / .cuda() / .float() etc. to the wrapped module
        # which is stored in a tuple (not registered as nn.Module submodule).
        self._module[0]._apply(fn)
        return super()._apply(fn, recurse=recurse)

    def forward(self, *args, **kwargs):
        assert not self.module.training
        if not self.allow_grad:
            with torch.no_grad():
                return self.module(*args, **kwargs)
        return self.module(*args, **kwargs)

    def __repr__(self):
        m = self.module
        name = []
        while m is not None:
            name.append(m.__class__.__name__)
            if hasattr(m, "module"):
                m = m.module
            elif hasattr(m, "model"):
                m = m.model
            elif hasattr(m, "_orig_mod"):
                m = m._orig_mod
            else:
                m = None
        name = "/".join(name)
        return f"Frozen({name})"


class DinoEncoder(nn.Module):
    """Frozen DINOv2 (facebook/dinov2-base) feature extractor for REPA loss."""

    def __init__(self, cache_dir=None):
        super().__init__()

        self.out_dim = 768
        self.base_patch_size = 14
        self.base_res = 224

        # Lazy import to avoid requiring transformers when REPA is disabled
        import transformers

        self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
        freeze_model(self)

    def rescale_and_process_image(self, x, target_n_tokens):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)((x + 1) / 2)
        _, _, h, w = x.shape
        r = math.sqrt(target_n_tokens * self.base_patch_size**2 / (h * w))
        H, W = round(h * r), round(w * r)
        x = torch.nn.functional.interpolate(x, (H, W), mode="bicubic")
        return x

    def forward(self, x, target_n_tokens=None):
        if self.training:
            self.eval()
        x = self.rescale_and_process_image(x, target_n_tokens)
        z = self.model(x).last_hidden_state[:, 1:]  # Remove CLS token
        return z
