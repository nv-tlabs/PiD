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
PID (PixelDiT SR) training and inference model.

LQ pixels are used only by training/callback code to create latents and
visualizations. The network-facing PiD condition is always latent-only.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import Any, Optional

import attrs
import torch
from torch import Tensor

from pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from pid._ext.imaginaire.utils import misc
from pid._src.degradation import (
    TrainDegradationConfig,
    simple_downsample_image,
)
from pid._src.models.latent_noising import LatentNoisingConfig
from pid._src.models.pixeldit_model import PixelDiTModel, PixelDiTModelConfig

try:
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer
except ImportError:
    LoraConfig = None
    inject_adapter_in_model = None
    BaseTunerLayer = None

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@attrs.define(slots=False)
class PixelDiTSRLoraConfig:
    enabled: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    # Image-stream attn + FFN + pixel-block FFN. lq_proj is always frozen separately.
    # mlp_x.w* uses dotted path to avoid matching mlp_y.w* (PEFT suffix-matches keys).
    lora_target_modules: list[str] = attrs.Factory(
        lambda: [
            "qkv_x",
            "proj_x",  # MMDiT image stream attention
            "mlp_x.w1",
            "mlp_x.w2",
            "mlp_x.w3",  # MMDiT image stream FFN (SwiGLU)
            "fc1",
            "fc2",  # PiT pixel block FFN
        ]
    )
    adapter_name: str = "fidelity"


@attrs.define(slots=False)
class PixelDiTSRLQLatentImageAlignConfig:
    # Training-only auxiliary supervision for latent-only LQ projection. The
    # projector predicts the raw LQ RGB image from its latent features; the image
    # is only a target for this loss, not a conditioning input to the network.
    enabled: bool = False
    weight: float = 0.1
    sigma_power: float = 1.0
    y_weight: float = 0.25
    chroma_weight: float = 1.0
    eps: float = 1e-3


@attrs.define(slots=False)
class PidModelConfig(PixelDiTModelConfig):
    # Degradation config — only fixed bicubic simple_downsample is supported.
    train_degradation_config: TrainDegradationConfig = attrs.Factory(TrainDegradationConfig)

    # Frozen VAE config for encoding LQ images to latent.
    # Uses same LazyDict format as SSDDModel tokenizer config.
    tokenizer: Any = None

    # VAE latent channels (must match tokenizer.latent_ch)
    state_ch: int = 16

    # Latent forward-noising: noises the LQ latent at a sampled σ (either
    # flow-matching `(1-σ)x_0 + σε` or SDXL VP `sqrt(α̅)x_0 + sqrt(1-α̅)ε`) and
    # writes both LQ_latent and a per-sample degrade_sigma into the batch.
    # See latent_noising.py for details.
    latent_noising: Optional[LatentNoisingConfig] = None

    # LQ latent feature alignment to raw LQ RGB.
    lq_latent_image_align_config: PixelDiTSRLQLatentImageAlignConfig = attrs.Factory(PixelDiTSRLQLatentImageAlignConfig)

    # Fidelity LoRA config: train a small LoRA on the backbone.
    lora_config: PixelDiTSRLoraConfig = attrs.Factory(PixelDiTSRLoraConfig)


# =============================================================================
# Model
# =============================================================================


class PidModel(PixelDiTModel):
    """PixelDiT SR training/inference model.

    Extends PixelDiTModel with:
    - Degradation pipeline for constructing LQ images from HQ
    - Frozen VAE for encoding LQ images to latent
    - LQ condition injection into the network
    """

    def __init__(self, config: PidModelConfig):
        super().__init__(config)

        # --- Simple LQ generation ---
        self.downscale = float(self._cfg_get(config.train_degradation_config, "downscale"))
        if self.downscale <= 0:
            raise ValueError(f"train_degradation_config.downscale must be positive, got {self.downscale}.")

        # --- Frozen VAE for LQ latent encoding ---
        # Named `vae_encoder` to avoid collision with parent's `self.tokenizer` (HuggingFace text tokenizer).
        if config.tokenizer is not None:
            with misc.timer("PixelDiTSRModel: load_vae"):
                from pid._src.tokenizers.base_vae import BaseVAE

                self.vae_encoder: BaseVAE = lazy_instantiate(config.tokenizer)
                if config.state_ch > 0:
                    assert self.vae_encoder.latent_ch == config.state_ch, (
                        f"latent_ch {self.vae_encoder.latent_ch} != state_ch {config.state_ch}"
                    )
        else:
            self.vae_encoder = None
            logger.warning("No VAE configured — LQ latent encoding disabled.")

        # --- Latent forward-noising (LQ degradation) ---
        self.latent_noiser = None
        if config.latent_noising is not None and config.latent_noising.enabled:
            from pid._src.models.latent_noising import LatentNoiser

            self.latent_noiser = LatentNoiser(config.latent_noising)

        # --- Re-apply train_lq_proj_only freeze ---
        # Parent PixelDiTModel.__init__ calls self.net.requires_grad_(True) which
        # overrides the freeze set in PixDiT_T2I_SR.__init__. Re-apply here.
        if hasattr(self.net, "train_lq_proj_only") and self.net.train_lq_proj_only:
            for p in self.net.parameters():
                p.requires_grad_(False)
            for p in self.net.lq_proj.parameters():
                p.requires_grad_(True)
            if hasattr(self.net, "pit_lq_gate") and self.net.pit_lq_gate is not None:
                if hasattr(self.net.pit_lq_gate, "parameters"):
                    for p in self.net.pit_lq_gate.parameters():
                        p.requires_grad_(True)
            trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.net.parameters())
            logger.info(f"train_lq_proj_only: {trainable:,} trainable / {total:,} total params")

        # --- Fidelity LoRA injection ---
        # Must come AFTER train_lq_proj_only block since it overrides requires_grad.
        # lq_proj stays frozen: inject_adapter_in_model only unfreezes LoRA A/B matrices.
        if config.lora_config.enabled:
            self._inject_fidelity_lora()

    # =========================================================================
    # Fidelity LoRA
    # =========================================================================

    def _inject_fidelity_lora(self):
        assert LoraConfig is not None, "peft is not installed, cannot use lora_config.enabled=True"
        cfg = self.config.lora_config

        # Freeze entire network first; PEFT will only unfreeze LoRA A/B matrices
        self.net.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            init_lora_weights=True,
        )
        inject_adapter_in_model(lora_cfg, self.net, adapter_name=cfg.adapter_name)

        # Inject LoRA into EMA model too so parameter lists stay aligned for
        # FastEmaModelUpdater.update_average (zips both models' parameters).
        if hasattr(self, "net_ema") and self.net_ema is not None:
            inject_adapter_in_model(lora_cfg, self.net_ema, adapter_name=cfg.adapter_name)
            self.net_ema.requires_grad_(False)
            # Re-sync so EMA LoRA weights start from the same init as net
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        trainable_names = [n for n, p in self.net.named_parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        logger.info(
            f"Fidelity LoRA injected: {trainable:,} trainable / {total:,} total "
            f"(rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, targets={cfg.lora_target_modules})"
        )
        logger.info(f"Trainable params ({len(trainable_names)}):\n  " + "\n  ".join(trainable_names))

    def set_lora_scale(self, scale: float):
        """Inference-time fidelity/creativity control. scale=0 → creative, scale=1 → fidelity."""
        if BaseTunerLayer is None:
            return
        for module in self.net.modules():
            if isinstance(module, BaseTunerLayer):
                module.set_scale(self.config.lora_config.adapter_name, scale)

    # =========================================================================
    # Degradation
    # =========================================================================

    @torch.no_grad()
    def generate_lr_sample(self, data_batch: dict, iteration: Optional[int] = None) -> dict:
        """Generate LQ image from HQ image via fixed bicubic simple_downsample.

        Args:
            data_batch: must contain self.config.input_data_key with HQ images
                [B, C, H, W] or [B, C, 1, H, W], range [-1, 1].
            iteration: ignored; kept for the shared training/inference call sites.

        Returns:
            data_batch with "LQ_video_or_image" key added, shape [B, C, H_lq, W_lq], range [-1, 1].
        """
        image = data_batch[self.config.input_data_key]
        image = self._normalize_image(image).to(device=image.device)
        if image.ndim == 5:
            image = image[:, :, 0, :, :]  # [B, C, 1, H, W] -> [B, C, H, W]

        lq_image = simple_downsample_image(image, self.downscale)
        data_batch["LQ_video_or_image"] = lq_image.to(device=image.device, dtype=image.dtype)
        return data_batch

    @torch.no_grad()
    def encode_lq_latent(self, lq_image: Tensor) -> Tensor:
        """Encode LQ image through frozen VAE to get LQ latent.

        Args:
            lq_image: [B, C, H_lq, W_lq] normalized to [-1, 1].

        Returns:
            LQ latent [B, z_dim, zH, zW].
        """
        # VAE expects [B, C, T, H, W] for video models
        if lq_image.ndim == 4:
            lq_image = lq_image.unsqueeze(2)  # [B, C, 1, H, W]
        latent = self.vae_encoder.encode(lq_image)
        if latent.ndim == 5:
            latent = latent[:, :, 0, :, :]  # [B, z_dim, 1, zH, zW] -> [B, z_dim, zH, zW]
        return latent

    def prepare_data_batch_for_training(self, data_batch: dict, training_iteration: Optional[int] = None) -> dict:
        """Create and degrade LQ conditions for a training step.

        Args:
            data_batch: raw data batch from dataloader.
            training_iteration: current training iteration.

        Returns:
            The batch with ``LQ_latent`` populated.
        """
        if training_iteration is None:
            raise ValueError("prepare_data_batch_for_training requires training_iteration")

        if not isinstance(data_batch.get("LQ_video_or_image"), torch.Tensor):
            data_batch = self.generate_lr_sample(data_batch, iteration=training_iteration)

        if not isinstance(data_batch.get("LQ_latent"), torch.Tensor):
            if self.vae_encoder is None:
                raise ValueError("Cannot create LQ_latent because this model has no VAE encoder")
            data_batch["LQ_latent"] = (
                self.encode_lq_latent(data_batch["LQ_video_or_image"]).contiguous().to(**self.tensor_kwargs)
            )

        assert self.training, "We should only degrade latent in training"
        self.latent_degrade_inplace(data_batch)

        return data_batch

    def _validate_inference_data_batch(self, data_batch: dict) -> None:
        """Validate the strict three-field latent-only PiD inference contract."""
        expected_keys = {self.config.input_caption_key, "LQ_latent", "degrade_sigma"}
        actual_keys = set(data_batch)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                "PiD inference batch must contain exactly "
                f"{sorted(expected_keys)}; missing={missing}, unexpected={unexpected}"
            )
        if not isinstance(data_batch.get("LQ_latent"), torch.Tensor):
            raise ValueError("PiD inference requires a pre-computed LQ_latent tensor")
        if data_batch["LQ_latent"].ndim != 4:
            raise ValueError(
                f"PiD inference expects LQ_latent with shape [B, C, H, W], got {tuple(data_batch['LQ_latent'].shape)}"
            )

    def _resolve_inference_image_size(self, lq_latent: Tensor, image_size=None) -> tuple[int, int]:
        """Resolve output H/W from an override or from latent/VAE/SR scales."""
        if image_size is not None:
            if isinstance(image_size, (list, tuple)):
                if len(image_size) != 2:
                    raise ValueError(f"image_size must be a scalar or (H, W), got {image_size}")
                img_h, img_w = int(image_size[0]), int(image_size[1])
            else:
                img_h = img_w = int(image_size)
        else:
            if self.vae_encoder is None:
                raise ValueError("Cannot infer output size without a configured VAE; pass image_size explicitly")
            vae_scale = int(self.vae_encoder.spatial_compression_factor)
            net = self.net.module if hasattr(self.net, "module") else self.net
            if not hasattr(net, "sr_scale"):
                raise ValueError("Cannot infer output size because the PiD network has no sr_scale")
            sr_scale = int(net.sr_scale)
            img_h = int(lq_latent.shape[-2]) * vae_scale * sr_scale
            img_w = int(lq_latent.shape[-1]) * vae_scale * sr_scale

        if img_h <= 0 or img_w <= 0:
            raise ValueError(f"Resolved image_size must be positive, got {(img_h, img_w)}")
        return img_h, img_w

    def latent_degrade_inplace(self, data_batch) -> dict:
        """Apply latent forward-noising to LQ latent."""
        if self.latent_noiser is not None:
            urls = data_batch.get("__url__")
            degraded_latent, degrade_sigma = self.latent_noiser(
                data_batch["LQ_latent"],
                urls=urls,
            )
            data_batch["LQ_latent"] = degraded_latent
            data_batch["degrade_sigma"] = degrade_sigma
        elif "degrade_sigma" not in data_batch and "LQ_latent" in data_batch:
            # No degradation applied — sigma = 0 for all samples
            B = data_batch["LQ_latent"].shape[0]
            data_batch["degrade_sigma"] = torch.zeros(B, device=data_batch["LQ_latent"].device, dtype=torch.float32)

    # =========================================================================
    # Training
    # =========================================================================

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        # CP setup: enable on the network (idempotent) and broadcast inputs
        # from the lowest-rank CP peer so every rank in a CP group sees the
        # exact same HQ image, captions, etc. The dataloader produces an
        # independent sample per global rank, so without this broadcast the
        # ranks within a CP group would diverge.
        self._maybe_enable_cp_on_nets([self.net])
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            data_batch[self.config.input_data_key] = self._broadcast_tensor_for_cp(
                data_batch[self.config.input_data_key]
            )
            data_batch[self.config.input_caption_key] = self._broadcast_object_for_cp(
                data_batch.get(self.config.input_caption_key)
            )
            # Drop derived LQ products so they are regenerated from the
            # broadcast HQ before the optional latent-noising step.
            data_batch.pop("LQ_video_or_image", None)
            data_batch.pop("LQ_latent", None)
            data_batch.pop("degrade_sigma", None)

        _shift = self.config.shift
        if self.config.dynamic_shift is not None:
            _raw = data_batch[self.config.input_data_key]
            _h, _w = _raw.shape[-2], _raw.shape[-1]
            _ds = self.config.dynamic_shift
            _shift = _ds["base_shift"] * math.sqrt(math.sqrt(_h * _w) / _ds["base_image_size_for_shift_calc"])

        # 1. Get and normalize HQ image
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]  # [B, C, 1, H, W] -> [B, C, H, W]

        # 2. Prepare LQ conditions (simple downsample + VAE encode)
        data_batch = self.prepare_data_batch_for_training(data_batch, training_iteration=iteration)

        # Broadcast derived LQ products across the CP group. simple_downsample is
        # deterministic, but latent_noising can still draw per-rank noise.
        if cp_group is not None and cp_group.size() > 1:
            for _key in ("LQ_video_or_image", "LQ_latent", "degrade_sigma"):
                if isinstance(data_batch.get(_key), torch.Tensor):
                    data_batch[_key] = self._broadcast_tensor_for_cp(data_batch[_key])

        condition = self.conditioner(data_batch)
        captions = condition.caption
        # AR collator collapses identical caption strings (e.g. all arxiv samples
        # share a fixed prompt) into a single str / len-1 list — broadcast back
        # to per-sample list of B before the text encoder.
        B = x0.shape[0]
        if isinstance(captions, str):
            captions = [captions] * B
        elif isinstance(captions, list) and len(captions) == 1 and B > 1:
            captions = captions * B
        caption_embs, emb_masks = self._encode_text_raw(captions)
        caption_embs = caption_embs.to(**self.tensor_kwargs)

        # 4. PiD always conditions the network on the LQ latent. The LQ image
        # remains a training-only source/target for VAE encoding and RGB alignment.
        lq_latent = condition.lq_latent
        if lq_latent is None:
            raise ValueError("PiD conditioner did not produce lq_latent")
        lq_condition_keep_mask = lq_latent.detach().abs().flatten(1).any(dim=1)
        lq_latent = lq_latent.to(**self.tensor_kwargs)

        lq_align_cfg = self.config.lq_latent_image_align_config
        use_lq_image_align = bool(self._cfg_get(lq_align_cfg, "enabled")) and self._cfg_get(lq_align_cfg, "weight") > 0
        lq_aux_target = data_batch["LQ_video_or_image"] if use_lq_image_align else None

        # 5. Sample t and apply flow shift (per-step _shift comes from multi-res block above).
        # Under CP, t and the FM noise must be identical across CP ranks — we
        # pre-sample both and broadcast them, then pass `noise=` explicitly to
        # `fm_trainer.loss` to bypass its internal `randn_like`.
        t = self.fm_trainer.sample_t(x0.shape[0], device=x0.device)
        if _shift != 1.0:
            t = (_shift * t) / (1.0 + (_shift - 1.0) * t)
        noise = torch.randn_like(x0)
        if cp_group is not None and cp_group.size() > 1:
            t = self._broadcast_tensor_for_cp(t)
            noise = self._broadcast_tensor_for_cp(noise)

        # 6. Flow matching loss
        degrade_sigma = data_batch.get("degrade_sigma")
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        lq_aux_cache = {}

        def _net_fn(x_t, t, **kwargs):
            # velocity mode: net predicts v = noise - x0; FM trainer expects x0 - noise → negate.
            # x0 mode: net predicts x0 directly; FM trainer (prediction_type="x0") consumes it as-is.
            out = self.net(
                x_t,
                t,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
                return_lq_aux=use_lq_image_align,
            )
            if use_lq_image_align:
                out, lq_aux = out
                lq_aux_cache["lq_aux"] = lq_aux
            if self.config.prediction_type == "x0":
                return out
            elif self.config.prediction_type == "velocity":
                return -out
            else:
                raise ValueError(f"Invalid prediction_type: {self.config.prediction_type}")

        with autocast_ctx:
            diff_loss, _ = self.fm_trainer.loss(
                fn=_net_fn,
                x=x0,
                t=t,
                noise=noise,
            )

        loss_dict = {"diffusion_loss": diff_loss}
        total_loss = self.config.loss_weights.get("diffusion", 1.0) * diff_loss
        lq_image_align_weighted_loss = None

        # 7. Optional REPA loss
        if self.repa_loss is not None:
            repa_loss = self.repa_loss(x0)
            loss_dict["repa_loss"] = repa_loss
            total_loss = total_loss + self.config.loss_weights.get("repa", 0.25) * repa_loss

        # 7b. Optional LQ latent feature alignment to raw LQ image.
        if use_lq_image_align:
            lq_image_align_val = self._lq_latent_image_align_loss(
                lq_aux=lq_aux_cache.get("lq_aux"),
                target_lq=lq_aux_target,
                degrade_sigma=degrade_sigma,
                condition_keep_mask=lq_condition_keep_mask,
            )
            loss_dict["lq_latent_image_align_loss"] = lq_image_align_val
            lq_image_align_weighted_loss = self._cfg_get(lq_align_cfg, "weight") * lq_image_align_val
            total_loss = total_loss + lq_image_align_weighted_loss

        loss_dict["total_loss"] = total_loss
        output_batch = {"fm_loss": total_loss.detach(), "loss_dict": loss_dict}
        # Scale loss by cp_size before backward — every gradient flowing into
        # `self.net` is a 1/cp_size slice of the full-batch gradient (the
        # network gathers L tokens at the end via `cat_outputs_cp_with_grad`,
        # which preserves grad only on the local slice). FSDP averages
        # gradients over `world_size`, so multiplying the loss by `cp_size`
        # restores the correct full-batch gradient. No-op when cp_size==1.
        backward_loss = total_loss * self._cp_loss_scale
        if lq_image_align_weighted_loss is not None:
            # The aux RGB reconstruction is computed on every CP rank using the
            # full LQ image, so it should not receive the patch-token CP scale.
            backward_loss = backward_loss - lq_image_align_weighted_loss * (self._cp_loss_scale - 1.0)
        return output_batch, backward_loss

    def validation_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        return self.training_step(data_batch, iteration)

    def _lq_latent_image_align_loss(
        self,
        lq_aux: Optional[dict],
        target_lq: Optional[Tensor],
        degrade_sigma,
        condition_keep_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Color-focused reconstruction loss from LQ latent features to raw LQ image."""
        if lq_aux is None or lq_aux.get("pred_lq_rgb") is None:
            raise RuntimeError(
                "lq_latent_image_align_config is enabled, but the network did not return pred_lq_rgb. "
                "Set model.config.net.lq_aux_rgb_head=True on an LQ projection that supports aux RGB output."
            )
        if target_lq is None:
            raise RuntimeError(
                "lq_latent_image_align_config is enabled, but the training batch has no LQ image target."
            )

        pred = lq_aux["pred_lq_rgb"].float()
        target = target_lq
        if target.ndim == 5:
            target = target[:, :, 0, :, :]
        target = target.to(device=pred.device).float()
        if target.shape[-2:] != pred.shape[-2:]:
            target = torch.nn.functional.interpolate(target, size=pred.shape[-2:], mode="area")

        align_cfg = self.config.lq_latent_image_align_config
        pred_ycc = self._rgb_to_ycbcr((pred + 1.0) * 0.5)
        target_ycc = self._rgb_to_ycbcr((target.clamp(-1, 1) + 1.0) * 0.5)
        weights = pred_ycc.new_tensor(
            [
                self._cfg_get(align_cfg, "y_weight"),
                self._cfg_get(align_cfg, "chroma_weight"),
                self._cfg_get(align_cfg, "chroma_weight"),
            ]
        ).view(1, 3, 1, 1)

        eps = float(self._cfg_get(align_cfg, "eps"))
        per_sample = torch.sqrt(((pred_ycc - target_ycc) * weights).pow(2) + eps * eps) - eps
        per_sample = per_sample.mean(dim=(1, 2, 3))

        valid = target.abs().flatten(1).mean(dim=1) > 1e-6
        if condition_keep_mask is not None:
            valid = valid & condition_keep_mask.to(device=valid.device, dtype=torch.bool).view(-1)
        sample_weight = valid.to(dtype=per_sample.dtype, device=per_sample.device)
        if degrade_sigma is not None:
            sigma = degrade_sigma.to(device=per_sample.device).float().view(-1)
            sample_weight = sample_weight * (1.0 - sigma).clamp_min(0.0).pow(
                float(self._cfg_get(align_cfg, "sigma_power"))
            )
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)

    @staticmethod
    def _cfg_get(config: Any, key: str) -> Any:
        if isinstance(config, dict):
            return config[key]
        return getattr(config, key)

    @staticmethod
    def _rgb_to_ycbcr(x: Tensor) -> Tensor:
        """Differentiable RGB [0,1] to YCbCr-like opponent channels."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)

    def forward(self, x, t, y, **kwargs):
        """Direct network forward pass."""
        kwargs.pop("lq_video_or_image", None)
        return self.net(x, t, y, lq_video_or_image=None, **kwargs)

    # =========================================================================
    # Inference
    # =========================================================================

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        cfg_scale: float = None,
        num_steps: int = None,
        seed: int = 0,
        image_size=None,
        shift: float = None,
        **kwargs,
    ) -> Tensor:
        """Generate SR images from LQ input using DPM-Solver with CFG.

        The data_batch should contain:
        - caption key: text captions
        - "LQ_latent": pre-computed LQ VAE latent
        - "degrade_sigma": per-sample degradation level

        Returns:
            SR images [B, 3, 1, H, W] in [-1, 1].
        """
        from pid._src.modules.dpmsolver import DPMS

        self._validate_inference_data_batch(data_batch)

        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        num_steps = num_steps if num_steps is not None else self.config.num_sample_steps
        _shift_override = shift  # None means "not explicitly passed"

        # Enable CP on the network and broadcast inputs from CP rank 0 so all
        # ranks step the same noise/timesteps and emit identical samples.
        self._maybe_enable_cp_on_nets([self.net])
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            data_batch[self.config.input_caption_key] = self._broadcast_object_for_cp(
                data_batch.get(self.config.input_caption_key)
            )
            for _key in ("LQ_latent", "degrade_sigma"):
                if isinstance(data_batch.get(_key), torch.Tensor):
                    data_batch[_key] = self._broadcast_tensor_for_cp(data_batch[_key])

        lq_latent = data_batch["LQ_latent"].to(**self.tensor_kwargs)
        img_h, img_w = self._resolve_inference_image_size(lq_latent, image_size=image_size)

        # Determine shift: explicit arg > SD3 formula (if dynamic_shift) > config default
        if _shift_override is not None:
            shift = _shift_override
        elif self.config.dynamic_shift is not None:
            _ds = self.config.dynamic_shift
            shift = _ds["base_shift"] * math.sqrt(math.sqrt(img_h * img_w) / _ds["base_image_size_for_shift_calc"])
        else:
            shift = self.config.shift

        net = self.net
        net.eval()

        # Determine batch size from the latent, not from captions: collators may
        # collapse identical captions into a single string/list entry.
        captions = data_batch[self.config.input_caption_key]
        B = lq_latent.shape[0]
        if isinstance(captions, str):
            captions = [captions] * B
        elif isinstance(captions, tuple):
            captions = list(captions)
        if isinstance(captions, list) and len(captions) == 1 and B > 1:
            captions = captions * B
        if not isinstance(captions, list) or len(captions) != B:
            raise ValueError(f"Expected {B} captions for LQ_latent batch, got {captions!r}")

        caption_embs, emb_masks = self._encode_text_raw(captions)
        caption_embs = caption_embs.unsqueeze(1)  # [B, 1, L, C]

        # Null conditioning for CFG
        null_y = self._null_caption_embs.unsqueeze(1).repeat(B, 1, 1, 1)

        model_dtype = next(net.parameters()).dtype

        # Degradation sigma conditioning for inference creativity control.
        # Source of truth is data_batch["degrade_sigma"]; accepts float / list / tensor.
        sigma_val = data_batch["degrade_sigma"]
        if isinstance(sigma_val, torch.Tensor):
            degrade_sigma_tensor = sigma_val.to(device="cuda", dtype=torch.float32).reshape(-1)
            if degrade_sigma_tensor.numel() == 1:
                degrade_sigma_tensor = degrade_sigma_tensor.expand(B).contiguous()
            assert degrade_sigma_tensor.shape == (B,), (
                f"data_batch['degrade_sigma'] expected [B={B}], got {tuple(degrade_sigma_tensor.shape)}"
            )
        elif isinstance(sigma_val, (list, tuple)):
            degrade_sigma_tensor = torch.tensor(sigma_val, device="cuda", dtype=torch.float32)
            assert degrade_sigma_tensor.shape == (B,), (
                f"data_batch['degrade_sigma'] expected length {B}, got {len(sigma_val)}"
            )
        else:
            degrade_sigma_tensor = torch.full((B,), float(sigma_val), device="cuda", dtype=torch.float32)

        # Sample initial noise. Use a dedicated CUDA Generator so --seed controls the
        # sampler regardless of what else in the process has drawn from the global RNG
        # (text encoder, VAE encode, etc.). DPM-Solver runs as an ODE here, so there is
        # no intermediate stochasticity — only this initial draw matters.
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        z = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()

        with autocast_ctx:
            # Simple forward for no-guidance case (cfg_scale=1.0) where DPMS
            # does NOT double the batch.
            def _forward_fn(x, timestep, y, mask=None, **kw):
                x = x.to(model_dtype)
                timestep = timestep.to(model_dtype)
                if y.dim() == 4:
                    y = y.squeeze(1)
                y = y.to(model_dtype)
                return net(
                    x,
                    timestep,
                    y,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )

            # CFG model_fn: DPMS doubles the batch as [uncond, cond].
            # Text-only CFG: LQ conditions always present (weight=1),
            # only text varies between unconditional and conditional paths.
            def _cfg_model_fn(x, timestep, y, mask=None, **kw):
                half_B = x.shape[0] // 2
                if y.dim() == 4:
                    y = y.squeeze(1)

                # First half = unconditional (null text + real LQ)
                out_uncond = net(
                    x[:half_B].to(model_dtype),
                    timestep[:half_B].to(model_dtype),
                    y[:half_B].to(model_dtype),
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )
                # Second half = conditional (real text + real LQ)
                out_cond = net(
                    x[half_B:].to(model_dtype),
                    timestep[half_B:].to(model_dtype),
                    y[half_B:].to(model_dtype),
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )
                return torch.cat([out_uncond, out_cond], dim=0)

            # Use simple forward when cfg_scale=1.0 (DPMS won't double batch)
            model_fn = _forward_fn if cfg_scale == 1.0 else _cfg_model_fn

            model_kwargs = dict(mask=emb_masks)
            # x0 mode: DPMS's "x_start" branch converts x0 → noise internally using FLOW schedule.
            if self.config.prediction_type == "x0":
                dpms_model_type = "x_start"
            elif self.config.prediction_type == "velocity":
                dpms_model_type = "flow"
            else:
                raise ValueError(f"Invalid prediction_type: {self.config.prediction_type}")
            dpm_solver = DPMS(
                model_fn,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=cfg_scale,
                model_type=dpms_model_type,
                guidance_type="classifier-free",
                model_kwargs=model_kwargs,
                schedule="FLOW",
                interval_guidance=[0, 1],
            )
            # Multistep DPM-Solver requires `steps >= order`. At num_steps=1 the
            # second-order multistep solver can't bootstrap, so clamp order at the
            # step count: order=1 == DPM-Solver-1 == DDIM == Euler step for flow
            # matching, which is the natural one-shot fallback.
            samples = dpm_solver.sample(
                z,
                steps=num_steps,
                order=min(num_steps, 2),
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=shift,
            )

        return samples.clamp(-1, 1).unsqueeze(2)

    # =========================================================================
    # Callback interface
    # =========================================================================

    def get_data_and_condition(self, data_batch: dict, **kwargs):
        """Extract GT image and LQ condition for visualization callbacks."""
        if not isinstance(data_batch.get("LQ_video_or_image"), torch.Tensor):
            raise ValueError("PiD visualization requires LQ_video_or_image")
        if not isinstance(data_batch.get("LQ_latent"), torch.Tensor):
            raise ValueError("PiD visualization requires LQ_latent")
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]
        raw_data = x0.unsqueeze(2)  # [B, C, 1, H, W]

        condition = self.conditioner(data_batch, override_dropout_rate={n: 0.0 for n in self.conditioner.embedders})
        return raw_data, x0, condition

    # =========================================================================
    # Optimizer
    # =========================================================================

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        from pid._ext.imaginaire.utils.optim_instantiate import get_base_scheduler

        trainable_modules = [self.net]
        if self.repa_loss is not None:
            trainable_modules.append(self.repa_loss.repa_mlp)
        optim_model = torch.nn.ModuleList(trainable_modules)
        optimizer = lazy_instantiate(optimizer_config, model=optim_model)

        class _SchedulerProxy:
            def __init__(self, model):
                self._model = model

            @property
            def sample_counter(self):
                return getattr(self._model, "sample_counter", 0)

            def __getstate__(self):
                return {"sample_counter": self.sample_counter}

            def __setstate__(self, state):
                self._model = None

        scheduler = get_base_scheduler(optimizer, _SchedulerProxy(self), scheduler_config)
        return optimizer, scheduler

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def state_dict(self, *args, **kwargs):
        sd = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            sd.update(self.net_ema.state_dict(prefix="net_ema."))
        if self.repa_loss is not None:
            sd.update(self.repa_loss.state_dict(prefix="repa_loss."))
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        """Load checkpoint. Handles T2I-only checkpoints (missing LQ keys).

        When loading a T2I checkpoint into SR model, LQ projection keys will
        be missing — use strict=False to ignore them. The zero-initialized LQ
        projections ensure the model starts from pretrained T2I behavior.
        """
        # Detect format
        has_core_keys = any(k.startswith("core.") for k in state_dict)
        has_net_keys = any(k.startswith("net.") for k in state_dict)

        if has_core_keys and not has_net_keys:
            # Original PixelDiT checkpoint (core.* prefix)
            logger.info("Loading original PixelDiT checkpoint (core.* prefix) into SR model")
            net_sd = {}
            repa_sd = {}
            for k, v in state_dict.items():
                if k == "pos_embed":
                    continue
                if k.startswith("core."):
                    net_sd[k[len("core.") :]] = v
                elif k.startswith("_repa_projector."):
                    new_key = k.replace("_repa_projector.", "repa_mlp.")
                    repa_sd[new_key] = v
            # Always load with strict=False for SR (LQ keys will be missing)
            missing, unexpected = self.net.load_state_dict(net_sd, strict=False, assign=assign)
            if missing:
                lq_missing = [k for k in missing if "lq_proj" in k]
                other_missing = [k for k in missing if "lq_proj" not in k]
                if lq_missing:
                    logger.info(f"Expected missing LQ keys ({len(lq_missing)} keys)")
                if other_missing:
                    logger.warning(f"Unexpected missing keys in net: {other_missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in net: {unexpected}")
            if self.repa_loss is not None and repa_sd:
                self.repa_loss.load_state_dict(repa_sd, strict=False, assign=assign)
            if self.config.ema.enabled and hasattr(self, "net_ema"):
                self.net_ema.load_state_dict(net_sd, strict=False, assign=assign)
        else:
            # Our checkpoint format (net.*, net_ema.*, repa_loss.*)
            _net_sd = {
                k[len("net.") :]: v
                for k, v in state_dict.items()
                if k.startswith("net.") and not k.startswith("net_ema.")
            }
            _ema_sd = {k[len("net_ema.") :]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
            _repa_sd = {k[len("repa_loss.") :]: v for k, v in state_dict.items() if k.startswith("repa_loss.")}

            if _net_sd:
                missing, unexpected = self.net.load_state_dict(_net_sd, strict=False, assign=assign)
                if missing:
                    lq_missing = [k for k in missing if "lq_proj" in k]
                    other_missing = [k for k in missing if "lq_proj" not in k]
                    if lq_missing:
                        logger.info(f"Expected missing LQ keys ({len(lq_missing)} keys) — loading T2I into SR")
                    if other_missing and strict:
                        logger.warning(f"Missing keys in net: {other_missing}")
            if _ema_sd and self.config.ema.enabled and hasattr(self, "net_ema"):
                self.net_ema.load_state_dict(_ema_sd, strict=False, assign=assign)
            if _repa_sd and self.repa_loss is not None:
                self.repa_loss.load_state_dict(_repa_sd, strict=False, assign=assign)
