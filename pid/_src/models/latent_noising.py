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
Latent noising — forward-noise a (VAE-encoded) latent at a sampled σ and
return it together with the per-sample σ magnitude. Used to build the
degraded LQ_latent + degrade_sigma conditioning inputs for PixelDiTSRModel.

Both backbones sample σ ~ U[add_sigma_min, add_sigma_max] (with optional
per-URL overrides via `url_sigma_overrides`) and only differ in the x_t
formula:

  "flow_matching"   x_t = (1 - σ) x_0 + σ ε         σ ∈ [0, 1]
                    The standard flow-matching forward.
                    `degrade_sigma` returned equals the sampled σ.
                    Used by all flux / flux2 / sd3 / qwen / rae configs.

  "sdxl"            x_t = sqrt(1 - σ²) x_0 + σ ε    σ ∈ [0, 1]   (interpreted as σ_vp)
                    The variance-preserving (DDPM/VP) form, in σ-space.
                    Equivalent to `sqrt(α̅) x_0 + sqrt(1 - α̅) ε` with
                    σ_vp = sqrt(1 - α̅) — but parameterised directly in σ_vp,
                    so there is no discrete training-timestep schedule to
                    look up. Variance preserving: (1-σ²) + σ² = 1.
                    `degrade_sigma` returned equals the sampled σ_vp.

Both forms recover the input at σ = 0 and pure noise at σ = 1, and both
return `degrade_sigma ∈ [0, 1]` so a downstream σ-conditioning embedding
in the student model has consistent meaning across backbones.

Inference-time frame conversion (read this before consuming an SDXL inference
pipeline's intermediate latent as a conditioning input):

  SDXL's default *inference* loop uses EulerDiscreteScheduler, which holds
  intermediate latents in the variance-exploding Euler frame
      x_t^eu = x_0 + σ_eu ε,   σ_eu = sqrt((1 - α̅_t) / α̅_t)
  Our backbone="sdxl" path expects the variance-preserving training frame
      x_t^vp = sqrt(1 - σ_vp²) x_0 + σ_vp ε
  They differ only by a scalar:
      x_t^vp = x_t^eu / sqrt(σ_eu² + 1),   σ_vp = σ_eu / sqrt(σ_eu² + 1)
  The factor 1 / sqrt(σ_eu² + 1) is exactly what diffusers'
  `EulerDiscreteScheduler.scale_model_input(latent, t)` already applies, so
  converting an Euler-frame inference latent to the frame the student model
  was conditioned on is a single call to that method — no custom math.
"""

from __future__ import annotations

from typing import Optional, Union

import attrs
import torch
from torch import Tensor

from pid._ext.imaginaire.utils import log as logger

# Number of __call__ invocations per worker process for which the
# url_sigma_overrides debug log fires. Each rank's logs land in its own train
# log file, so the first few iters per worker is enough to verify routing
# without flooding logs.
_OVERRIDE_LOG_MAX_CALLS = 50


@attrs.define(slots=False)
class UrlSigmaOverride:
    """Per-source σ override rule.

    A sample whose `__url__` field contains `url_substring` (case-sensitive)
    uses [add_sigma_min, add_sigma_max] in place of the global default. First
    match wins; samples that match nothing fall back to the global default.

    σ is interpreted as σ_vp ∈ [0, 1] under backbone="sdxl" (controls
    sqrt(1 - σ²)·x_0 + σ·ε) and as the flow-matching interpolation
    coefficient under backbone="flow_matching" ((1 - σ)·x_0 + σ·ε). The
    same override values therefore mean the same thing — "how strongly to
    noise this source" — across both backbones.
    """

    url_substring: str = ""
    add_sigma_min: float = 0.0
    add_sigma_max: float = 1.0


@attrs.define(slots=False)
class LatentNoisingConfig:
    """Configuration for latent forward-noising.

    σ is sampled per-sample from U[add_sigma_min, add_sigma_max] (or, if
    `url_sigma_overrides` matches the sample's `__url__`, from that rule's
    range). The `backbone` field only selects the x_t formula:

      backbone="flow_matching":   x_t = (1 - σ) x_0 + σ ε
      backbone="sdxl":            x_t = sqrt(1 - σ²) x_0 + σ ε   (VP form)
    """

    enabled: bool = False

    # "flow_matching" or "sdxl" — selects the x_t functional form only.
    backbone: str = "flow_matching"

    # Probability of applying noising to each sample (per-sample Bernoulli).
    apply_prob: float = 0.75

    # Independent per-sample probability of forcing the returned latent to be
    # the clean latent with sigma=0, regardless of how sigma would otherwise be
    # sampled. Drawn independently of apply_prob: a sample is clean if its
    # apply draw says "don't degrade" OR its clean draw says "force clean".
    # Useful for mixing in a small fraction of clean conditioning during
    # training so the model sees sigma=0 as a valid input.
    clean_latent_ratio: float = 0.0

    # Global σ range. Applies to whichever backbone is selected.
    add_sigma_min: float = 0.0
    add_sigma_max: float = 1.0

    # Per-source [add_sigma_min, add_sigma_max] overrides. For each sample the
    # first rule whose `url_substring` is contained in data_batch["__url__"]
    # wins; samples that match nothing fall back to the global range above.
    # Default None (treated as empty list at use-site) — OmegaConf cannot wrap
    # attrs.Factory(list) as a structured-config default, hence None over [].
    url_sigma_overrides: Optional[list] = None


# =============================================================================
# Helpers for url-based sigma overrides
# =============================================================================


def _broadcast_urls(urls, B: int) -> Optional[list[str]]:
    """Normalize the `__url__` field into a length-B list[str], or None if absent.

    The aspect-ratio collator collapses identical urls into a single str and
    keeps differing urls as a list — handle both. Non-string fallback returns
    None.
    """
    if urls is None:
        return None
    if isinstance(urls, str):
        return [urls] * B
    try:
        url_list = list(urls)
    except TypeError:
        return None
    if not all(isinstance(u, str) for u in url_list):
        return None
    if len(url_list) == 1 and B > 1:
        return url_list * B
    assert len(url_list) == B, f"urls length {len(url_list)} != batch size {B}"
    return url_list


def _parse_url_sigma_override(rule) -> tuple[str, float, float]:
    """Extract (substring, sigma_min, sigma_max) from a UrlSigmaOverride or dict-like.

    OmegaConf may hand us a DictConfig instead of an attrs instance — accept both.
    """
    if hasattr(rule, "url_substring"):
        return (
            str(rule.url_substring),
            float(rule.add_sigma_min),
            float(rule.add_sigma_max),
        )
    return (
        str(rule["url_substring"]),
        float(rule["add_sigma_min"]),
        float(rule["add_sigma_max"]),
    )


# =============================================================================
# Main noiser class
# =============================================================================


class LatentNoiser:
    """Forward-noise a clean VAE latent at a sampled σ.

    Usage in PixelDiTSRModel.latent_degrade_inplace:
        if self.latent_noiser is not None:
            degraded, sigma = self.latent_noiser(data_batch["LQ_latent"],
                                                 urls=data_batch.get("__url__"))
            data_batch["LQ_latent"] = degraded
            data_batch["degrade_sigma"] = sigma   # (B,), float32, σ ∈ [0, 1]
    """

    def __init__(
        self,
        config: LatentNoisingConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if config.backbone not in ("flow_matching", "sdxl"):
            raise ValueError(f"Unknown backbone {config.backbone!r}. Expected 'flow_matching' or 'sdxl'.")
        self.config = config
        self.device = device
        self.dtype = dtype
        self._override_log_calls: int = 0

    @torch.no_grad()
    def __call__(
        self,
        latent: Tensor,
        urls: Optional[Union[str, list[str]]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Noise the latent. Returns (noised_latent, sigma) with sigma ∈ [0,1].

        Args:
            latent: (B, C, H, W) clean VAE latent.
            urls:   optional WebDataset `__url__` (str or list[str]); used for
                    per-source σ-range routing via `url_sigma_overrides`.

        Returns:
            (result, sigma) where
              result: (B, C, H, W) noised latent, dtype/device matching `latent`
              sigma:  (B,) float32; 0 for samples where noising was not applied
        """
        B = latent.shape[0]
        device = latent.device

        # Two independent Bernoulli draws:
        #   apply_draw       — "do we noise this sample at all?" (apply_prob)
        #   force_clean_draw — "force sigma=0 / clean latent?" (clean_latent_ratio)
        # A sample is actually noised iff apply_draw is True AND force_clean_draw
        # is False.
        apply_draw = torch.rand(B, device=device) < self.config.apply_prob
        if self.config.clean_latent_ratio > 0.0:
            force_clean_draw = torch.rand(B, device=device) < self.config.clean_latent_ratio
            apply_mask = apply_draw & (~force_clean_draw)
        else:
            apply_mask = apply_draw
        if not apply_mask.any():
            return latent, torch.zeros(B, device=device, dtype=torch.float32)

        sigmas = self._sample_sigmas(B, device, urls)

        # σ is in [0, 1] under both backbones; only the mean-coefficient differs.
        s = sigmas.to(latent.dtype).reshape(B, *([1] * (latent.ndim - 1)))
        if self.config.backbone == "sdxl":
            # VP form: x_t = sqrt(1 - σ²) x_0 + σ ε. Clamp guards against fp
            # roundoff producing tiny negatives when σ → 1.
            mean_coef = torch.sqrt(torch.clamp(1.0 - sigmas**2, min=0.0))
            m = mean_coef.to(latent.dtype).reshape(B, *([1] * (latent.ndim - 1)))
            x_t = m * latent + s * torch.randn_like(latent)
        else:  # flow_matching
            x_t = (1.0 - s) * latent + s * torch.randn_like(latent)

        mask = apply_mask.reshape(B, *([1] * (latent.ndim - 1)))
        result = torch.where(mask, x_t, latent)
        out_sigma = torch.where(apply_mask, sigmas, torch.zeros_like(sigmas))
        return result.to(dtype=latent.dtype, device=latent.device), out_sigma

    # ------------------------------------------------------------------
    # σ sampling (shared across both backbones)
    # ------------------------------------------------------------------

    def _sample_sigmas(self, B: int, device: torch.device, urls: Optional[Union[str, list[str]]]) -> Tensor:
        """Sample σ ~ U[add_sigma_min, add_sigma_max] per-sample, optionally
        overridden per-URL.

        Same σ semantics for both backbones (∈ [0, 1]) — the backbone choice
        only changes how that σ is consumed in __call__.
        """
        sigma_min = torch.full((B,), float(self.config.add_sigma_min), device=device, dtype=torch.float32)
        sigma_max = torch.full((B,), float(self.config.add_sigma_max), device=device, dtype=torch.float32)

        overrides = self.config.url_sigma_overrides or []
        if overrides:
            url_list = _broadcast_urls(urls, B)
            if url_list is not None:
                parsed = [_parse_url_sigma_override(r) for r in overrides]
                matched_subs: list[Optional[str]] = [None] * B
                for i, u in enumerate(url_list):
                    for sub, lo, hi in parsed:
                        if sub and sub in u:
                            sigma_min[i] = lo
                            sigma_max[i] = hi
                            matched_subs[i] = sub
                            break  # first match wins
                self._maybe_log_url_overrides(B, parsed, url_list, matched_subs, sigma_min, sigma_max)

        u01 = torch.rand(B, device=device, dtype=torch.float32)
        return sigma_min + (sigma_max - sigma_min) * u01

    # ------------------------------------------------------------------
    # url_sigma_overrides debug log (throttled)
    # ------------------------------------------------------------------

    def _maybe_log_url_overrides(
        self,
        B: int,
        parsed: list[tuple[str, float, float]],
        url_list: list[str],
        matched_subs: list[Optional[str]],
        sigma_min: Tensor,
        sigma_max: Tensor,
    ) -> None:
        if self._override_log_calls >= _OVERRIDE_LOG_MAX_CALLS:
            return
        self._override_log_calls += 1
        rules_repr = ", ".join(f"{s!r}->[{lo:.3g},{hi:.3g}]" for s, lo, hi in parsed)
        per_sample = []
        for i, u in enumerate(url_list):
            sub = matched_subs[i]
            tag = f"hit({sub!r})" if sub else "default"
            per_sample.append(f"  [{i}] {tag} σ∈[{sigma_min[i].item():.3g},{sigma_max[i].item():.3g}] url={u}")
        # rank0_only=False so every rank emits its own line (loguru sink prefixes
        # with [RANK X]); useful for verifying per-source σ routing across ranks.
        logger.info(
            "[LatentNoiser] url_sigma_overrides call "
            f"#{self._override_log_calls}/{_OVERRIDE_LOG_MAX_CALLS} "
            f"(B={B}, backbone={self.config.backbone!r}, default σ∈"
            f"[{self.config.add_sigma_min:.3g},{self.config.add_sigma_max:.3g}], "
            f"rules: {rules_repr})\n" + "\n".join(per_sample),
            rank0_only=False,
        )
