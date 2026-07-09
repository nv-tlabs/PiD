# Flow matching utilities: rectified flow training and Euler sampling.
#
# TimeSamplerLogitNormal: LogitNormal time sampling: t = sigmoid(N(mean, std))
# FlowMatchingTrainer: Rectified flow: x_t = (1-t)*x + sigma(t)*noise, velocity prediction
#   Supports prediction_type="x0" (JiT paradigm, arxiv 2511.13720): network predicts clean
#   image x0 instead of velocity, loss is computed in velocity space. The x0->velocity
#   conversion v = (x0_pred - x_t) / t gives implicit SNR-like time-dependent weighting.
#   Our time convention: t=0 clean, t=1 noise (JiT uses opposite), so singularity is at t≈0.
# FMEulerSampler: Multi-step Euler sampling with power-shifted time steps
#
# IMPORTANT: t and t*timescale (0~1000) are ALWAYS kept in float32 to avoid bfloat16
# truncation errors. x_t follows model dtype for the network forward, but a float32 copy
# is kept for loss computation (avoids bfloat16 quantization amplified by 1/t division).
# v_pred, target, and loss are computed in float32.
#
# Original source: SSDD/ssdd/flow.py
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import torch


class TimeSamplerLogitNormal:
    def __init__(self, t_mean=0, t_std=1.0, **kwargs):
        self.t_std = t_std
        self.t_mean = t_mean

    def __call__(self, batch_size, device):
        # Always float32 for time steps
        t = torch.randn(batch_size, device=device, dtype=torch.float32) * self.t_std + self.t_mean
        return torch.sigmoid(t)


class TimeSamplerBeta:
    """Beta(alpha, beta) distribution. alpha > beta biases toward t=1 (noise end).
    Examples: alpha=2,beta=1 → linear ramp toward 1; alpha=5,beta=1 → strong bias toward 1.
    """

    def __init__(self, alpha=2.0, beta=1.0, **kwargs):
        self.dist = torch.distributions.Beta(alpha, beta)

    def __call__(self, batch_size, device):
        return self.dist.sample((batch_size,)).to(device=device, dtype=torch.float32)


class TimeSamplerPower:
    """t = u^(1/k) where u ~ Uniform(0,1). PDF ∝ t^(k-1), so higher k biases toward t=1.
    Examples: k=2 → PDF ∝ t (moderate bias); k=4 → PDF ∝ t³ (strong bias toward 1).
    """

    def __init__(self, k=2.0, **kwargs):
        self.k = k

    def __call__(self, batch_size, device):
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        return u.pow(1.0 / self.k)


TIME_SAMPLER_REGISTRY = {
    "logit_normal": TimeSamplerLogitNormal,
    "beta": TimeSamplerBeta,
    "power": TimeSamplerPower,
}


class FlowMatchingTrainer:
    def __init__(
        self,
        *,
        timescale: float = 1_000,
        sigma_min: float = 0.0,
        t_sampler_args=None,
        t_sampler_type: str = "logit_normal",
        prediction_type: str = "velocity",
    ):
        assert prediction_type in ("velocity", "x0"), f"Unknown prediction_type: {prediction_type}"
        self.prediction_type = prediction_type
        assert t_sampler_type in TIME_SAMPLER_REGISTRY, (
            f"Unknown t_sampler_type: {t_sampler_type}, available: {list(TIME_SAMPLER_REGISTRY.keys())}"
        )
        self.t_sampler = TIME_SAMPLER_REGISTRY[t_sampler_type](**(t_sampler_args or {}))

        self.timescale = timescale
        self.sigma_min = sigma_min

    def alpha(self, t):
        return 1.0 - t

    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)

    def A(self, t):
        return 1.0

    def B(self, t):
        return -(1.0 - self.sigma_min)

    def add_noise(self, x, t, noise=None):
        # Convention: t=0.0 -> clean ; t=1.0 -> noise
        # It's DDPM convention, not the same as the rectified flow community.
        # Compute interpolation in float32 for precision, then cast result to x's dtype.
        # Returns (x_t_model_dtype, noise, x_t_f32) — the float32 copy avoids bfloat16
        # quantization noise being amplified by the 1/t division in velocity computation.
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        alpha_t = self.alpha(t).view(*s)  # float32 (t is always float32)
        sigma_t = self.sigma(t).view(*s)  # float32
        x_t_f32 = alpha_t * x.float() + sigma_t * noise.float()  # x_t = (1 - t) * x + t * noise
        return x_t_f32.to(dtype=x.dtype), noise, x_t_f32

    def loss(self, fn, x, t=None, fn_kwargs=None, noise=None):
        if fn_kwargs is None:
            fn_kwargs = {}

        # t is always float32 — precision-sensitive (scaled to 0~1000 for the network)
        if t is None:
            t = torch.rand(x.shape[0], device=x.device, dtype=torch.float32)
        t = t.float()

        x_t, noise, x_t_f32 = self.add_noise(x, t, noise=noise)

        # t * timescale passed to network as float32 conditioning; x_t in model dtype
        output = fn(x_t, t=t * self.timescale, **fn_kwargs)

        if self.prediction_type == "x0":
            # JiT paradigm: network predicts x0, convert to velocity for loss.
            # v = (x0_pred - x_t) / t, with t clamped at 5e-2 to avoid singularity at t≈0.
            # Use x_t_f32 (not x_t) to avoid bfloat16 quantization noise amplified by 1/t.
            # Keep v_pred in float32 — no cast back to model dtype, avoiding gradient bottleneck.
            x0_pred = output
            s = [x.shape[0]] + [1] * (x.dim() - 1)
            denom = t.float().view(*s).clamp(min=5e-2)
            v_pred = (output.float() - x_t_f32) / denom
        else:
            v_pred = output
            # Convert velocity to x0 prediction: x0 = x_t + v_pred * t (Euler step to t=0)
            s = [x.shape[0]] + [1] * (x.dim() - 1)
            t_expand = t.float().view(*s)
            x0_pred = (x_t_f32 + v_pred.float() * t_expand).to(dtype=output.dtype)

        # Compute target in float32 to avoid bfloat16 precision loss
        target = self.A(t) * x.float() + self.B(t) * noise.float()  # -dxt/dt = x - noise

        # reference of correctness: https://gemini.google.com/app/ab532748d9f6b417
        # t = 0 (clean) ------> x_t = (1 - t) * x + t * noise  -----> t = 1 (noise)
        # so the loss is | (x0_pred - x_t) / t - (x - noise) |
        # is equivalent to | (x_t - x0_pred) / t - (noise - x) |

        loss = ((v_pred.float() - target) ** 2).mean()
        return loss, (x_t, noise, t, v_pred, x0_pred)

    def sample_t(self, batch_size, device):
        return self.t_sampler(batch_size, device=device)

    def get_prediction(self, fn, x_t, t, fn_kwargs=None):
        # t must be float32 before scaling by timescale
        output = fn(x_t, t=t.float() * self.timescale, **(fn_kwargs or {}))
        if self.prediction_type == "x0":
            # Convert x0 prediction to velocity: v = (x0_pred - x_t) / t
            # No clamp here (unlike training loss): in inference the Euler step
            # multiplies v by dt, so the division by t cancels exactly and
            # clamping would only distort the result.
            s = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
            denom = t.float().view(*s)
            return ((output.float() - x_t.float()) / denom).to(dtype=output.dtype)
        return output

    def step(self, x_t, v_pred, cur_t, next_t=0):
        if not isinstance(v_pred, torch.Tensor):
            v_pred = torch.tensor(v_pred, device=x_t.device)
        # Compute step in float32, cast back to x_t dtype
        cur_t_f = cur_t.float().reshape((-1,) + (1,) * (x_t.dim() - 1))
        next_t_f = next_t.float() if isinstance(next_t, torch.Tensor) else next_t
        next_xt = x_t.float() + v_pred.float() * (cur_t_f - next_t_f)
        return next_xt.to(dtype=x_t.dtype)
