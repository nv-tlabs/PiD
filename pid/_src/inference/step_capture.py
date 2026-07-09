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
Diffusers `callback_on_step_end` callbacks for capturing intermediate denoising state.

Used by the demo entrypoint from_ldm.py to capture noisy xt at user-specified steps
(XtCaptureCallback) — and, for Flux, the model's x0 prediction (X0CaptureCallback) —
so each can be decoded alongside the final clean x0.
"""

import torch


class XtCaptureCallback:
    """Callback for pipeline.__call__() that captures noisy xt after K inference steps.

    User semantics: `K in save_ks` means "capture the latent AFTER K forward passes of
    the network". diffusers' `callback_on_step_end(step_index=i)` fires after step_index=i
    executes, so K steps have completed when step_index == K - 1 fires. At that point
    `callback_kwargs["latents"]` is already at `sigmas[K]` and that's what we store as
    the latent's `degrade_sigma`.

    Frame conversion: SDXL's `EulerDiscreteScheduler` holds latents in the variance-
    exploding Euler frame, but the SDXL PiD student was trained on VP/DDPM-frame latents.
    `to_training_frame(latent, σ, cfg)` rescales captures so the stored tensor and the
    stored σ match the frame the model expects — for SDXL it divides both by
    sqrt(σ²+1); for every other backbone (Flux / Flux2 / SD3 / QwenImage / ZImage) it's
    a no-op since the scheduler already uses flow-matching σ ∈ [0, 1].

    Captured dict is keyed by the user-facing K (not step_index), so output dirs and
    caption JSON land at `flux-{K}step_xt/` with sigma[K] — matching cheatsheet semantics.
    """

    def __init__(self, save_ks: set[int], cfg=None):
        # Map internal step_index -> user K so the caller can key by K.
        self.save_map = {k - 1: k for k in save_ks}
        self.cfg = cfg  # DiffusionPipelineConfig, needed for backbone-aware frame conversion
        self.captured: dict[int, tuple[torch.Tensor, float]] = {}  # keyed by K

    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        from pid._src.inference.pipeline_registry import to_training_frame

        k = self.save_map.get(step_index)
        if k is not None:
            sigmas = pipe.scheduler.sigmas
            sigma_idx = min(step_index + 1, len(sigmas) - 1)  # == K
            sigma_val = float(sigmas[sigma_idx].item())
            latent = callback_kwargs["latents"]
            if self.cfg is not None:
                latent, sigma_val = to_training_frame(latent, sigma_val, self.cfg)
            self.captured[k] = (latent.cpu(), sigma_val)
        return callback_kwargs


class X0CaptureCallback:
    """Capture x_0 prediction from the K-th transformer forward pass (Flux only).

    `callback_on_step_end` only exposes `latents` (xt AFTER the scheduler step), so to
    reach the velocity output of the transformer we register a forward post-hook on
    `pipeline.transformer` that stashes `(last_x_input, last_v_output)` on every call.
    Then in the callback (which fires after each step) we use the most recent stash to
    compute x_0_pred for the just-completed step:

        x_t  = (1 - sigma) * x_0 + sigma * noise         (flow matching)
        v    = noise - x_0                                (Flux predicts velocity)
        =>   x_0_pred = x_input - sigmas[step_index] * v

    User-facing K is 1-indexed (K=1 means "x_0 from the 1st forward pass"); the callback
    fires for step_index = K - 1. degrade_sigma stored is sigmas[K-1] — the sigma of the
    *input* that produced this prediction (matches each_step_vis.py:150-151).

    Flux-only because (a) Flux runs one transformer forward per step (guidance is
    distilled into the model), so the hook records exactly one (x, v) per step; and
    (b) latents stay packed (B, seq_len, 64) — extract_latent handles unpacking later.
    """

    def __init__(self, save_ks: set[int], transformer):
        self.save_map = {k - 1: k for k in save_ks}  # step_index -> user K
        self.captured: dict[int, tuple[torch.Tensor, float]] = {}
        self._last_x: torch.Tensor | None = None
        self._last_v: torch.Tensor | None = None
        self._handle = transformer.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        x = kwargs.get("hidden_states")
        if x is None and args:
            x = args[0]
        v = output[0] if isinstance(output, tuple) else output
        self._last_x = x.detach()
        self._last_v = v.detach()

    def __call__(self, pipe, step_index: int, timestep: torch.Tensor, callback_kwargs: dict) -> dict:
        k = self.save_map.get(step_index)
        if k is not None and self._last_x is not None and self._last_v is not None:
            sigma = float(pipe.scheduler.sigmas[step_index].item())
            x_0_pred = self._last_x.float() - sigma * self._last_v.float()
            self.captured[k] = (x_0_pred.to(self._last_v.dtype).cpu(), sigma)
        return callback_kwargs

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def compose_callbacks(*callbacks):
    """Chain multiple callback_on_step_end-compatible callables into one."""

    def combined(pipe, step_index, timestep, callback_kwargs):
        for cb in callbacks:
            callback_kwargs = cb(pipe, step_index, timestep, callback_kwargs)
        return callback_kwargs

    return combined
