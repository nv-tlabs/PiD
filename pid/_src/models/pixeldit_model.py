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

import logging
import math
from contextlib import contextmanager, nullcontext
from typing import Any

import attrs
import numpy as np
import torch
from torch import Tensor

from pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import misc
from pid._ext.imaginaire.utils.ema import FastEmaModelUpdater
from pid._src.losses.repa_loss import PixelDiTREPALoss
from pid._src.modules.ema import EMAConfig
from pid._src.networks.flow_matching import FlowMatchingTrainer
from pid._src.utils.context_parallel import broadcast as cp_broadcast
from pid._src.utils.context_parallel import robust_broadcast

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None  # CP is opt-in; gracefully degrade when megatron is absent

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@attrs.define(slots=False)
class PixelDiTModelConfig:
    # Network (lazy config -> PixDiT_T2I)
    net: Any = None

    # Precision: "bfloat16" uses autocast, net stays float32
    precision: str = "bfloat16"

    # Data keys
    input_data_key: str = "image"
    input_caption_key: str = "caption"

    # Text encoder config (Gemma-2-2b-it)
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    model_max_length: int = 300
    chi_prompt: list = attrs.Factory(list)
    # Conditioner: handles caption dropout via CaptionStringDrop embedder for CFG training.
    # Use {"override /conditioner": "pixeldit_caption"} in experiment config.
    conditioner: Any = None

    # Flow matching config
    # fm_timescale: original PixelDiT uses discrete timesteps 0-999 (timescale=1000).
    # FlowMatchingTrainer samples t in [0,1] then passes t*timescale to the network.
    fm_timescale: float = 1000.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # prediction_type: "velocity" — network predicts v = noise - x0 (current PixelDiT convention).
    #   "x0" (JiT paradigm, arxiv 2511.13720) — network predicts clean image x0; FlowMatchingTrainer
    #   internally converts to velocity via v = (x0_pred - x_t)/t for loss, giving implicit SNR-like
    #   time weighting. Must match between training and inference (DPMS switches to model_type="x_start").
    prediction_type: str = "velocity"

    # Inference config
    shift: float = 4.0
    cfg_scale: float = 2.75
    # int -> square; [H, W] list/tuple -> rectangular. Consumed only by
    # generate_samples_from_batch's shape fallback. Any so OmegaConf accepts a list.
    image_size: Any = 1024
    negative_prompt: str = "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, watermark"
    num_sample_steps: int = 50

    # REPA loss config (same pattern as ssdd_model.py)
    # e.g. {"i_extract": 8, "n_layers": 2}. disabled in 1024 finetune
    repa_config: dict | None = None
    loss_weights: dict = attrs.Factory(lambda: {"diffusion": 1.0, "repa": 0.5})

    # EMA
    ema: EMAConfig = attrs.Factory(EMAConfig)

    # Dynamic per-step shift via SD3 formula based on actual batch H, W.
    # Format: {"base_shift": float, "base_image_size_for_shift_calc": int}
    # Per-step shift = base_shift * sqrt(sqrt(H * W) / base_image_size_for_shift_calc).
    # Use this when the dataloader feeds varying resolutions per batch (e.g. the
    # multi-resolution dataloader at image_caption_multi_resolution_augmentor).
    # If None, the static config.shift is used.
    dynamic_shift: dict | None = None


# =============================================================================
# Text encoder helper
# =============================================================================

# Map of supported text encoder names to HuggingFace model IDs
_TEXT_ENCODER_DICT = {
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
}


def _load_text_encoder(name: str, device: str = "cuda"):
    """Load tokenizer and text encoder (decoder-only LM, extract decoder layers).

    Only rank 0 downloads from HuggingFace; other ranks wait behind a barrier
    then load from the local cache. This avoids 429 rate-limit errors when many
    ranks hit the HF API simultaneously.
    """
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert name in _TEXT_ENCODER_DICT, f"Unsupported text encoder: {name}"
    model_id = _TEXT_ENCODER_DICT[name]

    # Rank 0 downloads first; others wait then read from cache.
    is_distributed = dist.is_initialized()
    is_rank0 = (not is_distributed) or (dist.get_rank() == 0)

    if is_distributed and not is_rank0:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    text_encoder = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).get_decoder().to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    if is_distributed and is_rank0:
        dist.barrier()

    return tokenizer, text_encoder


# =============================================================================
# Model
# =============================================================================


class PixelDiTModel(ImaginaireModel):
    """PixelDiT T2I training/inference model.

    Pixel-space flow matching with MMDiT architecture.
    Text conditioning via frozen Gemma-2-2b-it encoder.
    """

    # Context-parallel: PixDiT_T2I.forward splits the patch tokens along L after
    # s_embedder and gathers before the final fold; training_step /
    # generate_samples_from_batch broadcast inputs (HQ image, captions, t,
    # noise) across the CP group and scale the loss by cp_size to compensate
    # for FSDP gradient averaging. ED (encoder-decoder) path is not CP-aware —
    # PixDiT_T2I.enable_context_parallel asserts when use_ed=True.
    SUPPORTS_CONTEXT_PARALLEL: bool = True

    def __init__(self, config: PixelDiTModelConfig):
        super().__init__()
        self.config = config

        # 1. Precision setup (same pattern as SSDDModel)
        _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        requested_dtype = _dtype_map[config.precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
            self.precision = torch.float32
        else:
            self.autocast_dtype = None
            self.precision = torch.float32
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # 2. Build network
        with misc.timer("PixelDiTModel: build_net"):
            self.net = lazy_instantiate(config.net)
            self.net = self.net.to(device="cuda", dtype=torch.float32)
            self.net.requires_grad_(True)
            if hasattr(self.net, "init_weights"):
                self.net.init_weights()
            if getattr(self.net, "patch_blocks", None):
                last_patch_block = self.net.patch_blocks[-1]
                if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                    last_patch_block.freeze_unused_text_output_branch()
            logger.info(f"PixDiT_T2I params: {sum(p.numel() for p in self.net.parameters()):,}")

        # 3. Text encoder (frozen)
        # Store tokenizer and text_encoder outside nn.Module registration to prevent
        # DCP checkpointer from saving them (they are frozen, ~2.6B params, and contain
        # non-picklable HuggingFace objects). We use object.__setattr__ to bypass
        # nn.Module.__setattr__ which would register nn.Module subclasses as children.
        with misc.timer("PixelDiTModel: load_text_encoder"):
            _tokenizer, _text_encoder = _load_text_encoder(config.text_encoder_name, device="cuda")
            object.__setattr__(self, "tokenizer", _tokenizer)
            object.__setattr__(self, "text_encoder", _text_encoder)
            # Pre-compute CHI prompt token count
            self._chi_prompt_str = "\n".join(config.chi_prompt) if config.chi_prompt else ""
            self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str)) if self._chi_prompt_str else 0
            # Pre-compute null caption embeddings for CFG
            self._null_caption_embs = self._encode_text_raw([config.negative_prompt if config.negative_prompt else ""])[
                0
            ]  # [1, L, C]

        # 4. Flow matching trainer
        self.fm_trainer = FlowMatchingTrainer(
            timescale=config.fm_timescale,
            sigma_min=0.0,
            t_sampler_args={"t_mean": config.logit_mean, "t_std": config.logit_std},
            t_sampler_type="logit_normal",
            prediction_type=config.prediction_type,
        )

        # 5. Optional REPA loss
        self.repa_loss = None
        if config.repa_config is not None:
            self.repa_loss = PixelDiTREPALoss(self.net, **config.repa_config)
            self.repa_loss = self.repa_loss.to(device="cuda")

        # 6. Conditioner (handles caption dropout for CFG training)
        self.conditioner = lazy_instantiate(config.conditioner)
        logger.info(f"PixelDiT conditioner: {self.conditioner}")

        # 7. Dynamic shift config (resolved per-step in training/inference).
        if config.dynamic_shift is not None:
            _ds = config.dynamic_shift
            logger.info(
                f"PixelDiT dynamic shift: base_shift={_ds['base_shift']} "
                f"base_image_size={_ds['base_image_size_for_shift_calc']}"
            )

        # 8. EMA
        if config.ema.enabled:
            self.net_ema = lazy_instantiate(config.net)
            self.net_ema = self.net_ema.to(device="cuda", dtype=torch.float32)
            self.net_ema.requires_grad_(False)
            self.net_ema_worker = FastEmaModelUpdater()
            s = config.ema.rate
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

    # =========================================================================
    # Text encoding
    # =========================================================================

    @torch.no_grad()
    def _encode_text_raw(self, captions: list[str]) -> tuple[Tensor, Tensor]:
        """Encode captions through text encoder.

        Returns:
            caption_embs: [B, model_max_length, caption_channels]
            emb_masks: [B, model_max_length]
        """
        # Optionally prepend CHI prompt
        if self._chi_prompt_str:
            prompts_all = [self._chi_prompt_str + cap for cap in captions]
            max_length_all = self._num_chi_tokens + self.config.model_max_length - 2
        else:
            prompts_all = captions
            max_length_all = self.config.model_max_length

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[
            0
        ]  # [B, max_length_all, C]

        # Select relevant tokens: BOS + last (model_max_length - 1) tokens
        select_index = [0] + list(range(-self.config.model_max_length + 1, 0))
        caption_embs = caption_embs[:, select_index]  # [B, model_max_length, C]
        emb_masks = caption_token.attention_mask[:, select_index]

        # Note: y_norm / y_norm_scale_factor are NOT applied to raw embeddings.
        # The original PixelDiT inference pipeline passes embeddings through unmodified.
        # The PixDiT_T2I network internally projects them via y_embedder + learned pos embedding.

        return caption_embs, emb_masks

    # =========================================================================
    # Data helpers
    # =========================================================================

    def _normalize_image(self, img: Tensor) -> Tensor:
        """Normalize image to [-1, 1]. Handles uint8 [0,255] or float [0,1]."""
        if img.dtype == torch.uint8:
            return img.float() / 127.5 - 1.0
        elif img.max() > 1.0:
            return img.float() / 127.5 - 1.0
        else:
            # Assume already in [-1, 1] or [0, 1]
            if img.min() >= 0:
                return img.float() * 2.0 - 1.0
            return img.float()

    # =========================================================================
    # Context-parallel helpers
    # ---------------------------------------------------------------------------
    # The pixel-diffusion network (`PixDiT_T2I_SR`) splits image patch tokens
    # internally and gathers the output before the final fold, so the model
    # layer's job is just (a) toggling CP on/off across the relevant networks
    # before each forward, (b) broadcasting input tensors so every CP rank holds
    # identical data, and (c) scaling the loss by `cp_size` before backward to
    # compensate for FSDP's gradient averaging — see plan file. Mirrors
    # `wan_t2v_model.py:get_context_parallel_group / broadcast_split_for_*`.
    # =========================================================================
    @staticmethod
    def get_context_parallel_group():
        if parallel_state is not None and parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    @property
    def _cp_size(self) -> int:
        cp_group = self.get_context_parallel_group()
        return cp_group.size() if cp_group is not None else 1

    @property
    def _cp_loss_scale(self) -> float:
        # FSDP averages gradients across `world_size`, but each CP rank's compute
        # graph spans only `1/cp_size` of the L tokens. Scaling the loss by
        # `cp_size` recovers the correct full-batch gradient (see plan file).
        return float(self._cp_size)

    def _maybe_enable_cp_on_nets(self, nets: list) -> None:
        """Enable CP on every network in `nets` if a CP group is initialized,
        otherwise disable. Idempotent — safe to call at the start of every step.
        """
        cp_group = self.get_context_parallel_group()
        for net in nets:
            if net is None:
                continue
            if cp_group is None or cp_group.size() <= 1:
                if hasattr(net, "disable_context_parallel") and getattr(net, "is_context_parallel_enabled", False):
                    net.disable_context_parallel()
            else:
                if hasattr(net, "enable_context_parallel"):
                    net.enable_context_parallel(cp_group)

    def _broadcast_tensor_for_cp(self, t: Tensor | None) -> Tensor | None:
        """Broadcast a tensor from the lowest-rank CP peer so every rank holds
        identical bytes. Used for HQ image, LQ image, LQ latent, etc."""
        cp_group = self.get_context_parallel_group()
        if t is None or cp_group is None or cp_group.size() <= 1:
            return t
        from torch.distributed import get_process_group_ranks

        src = min(get_process_group_ranks(cp_group))
        return robust_broadcast(t.contiguous(), src=src, pg=cp_group)

    def _broadcast_object_for_cp(self, obj):
        """Broadcast a python object (e.g. caption strings) across the CP group."""
        return cp_broadcast(obj, self.get_context_parallel_group())

    # =========================================================================
    # Callback interface: get_data_and_condition
    # =========================================================================

    class _EmptyCondition:
        """Minimal condition object for EveryNDrawSample callback compatibility."""

        pass

    def get_data_and_condition(self, data_batch: dict, **kwargs):
        """Extract GT image for visualization. Used by EveryNDrawSample callback.

        Returns:
            raw_data: pixel-space GT image [B, C, 1, H, W] for visualization grid
            x0: same as raw_data (no latent space for pixel diffusion)
            condition: empty object (no LQ condition for T2I)
        """
        # Prompt-only fix_batch (pure T2I viz): no GT image is provided, so there is
        # nothing to extract. Return raw_data=None — generate_samples_from_batch then
        # falls back to config.image_size for the output shape, and the callback skips
        # the GT row.
        if data_batch.get(self.config.input_data_key) is None:
            return None, None, self._EmptyCondition()
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]
        # Callback expects [B, C, T, H, W] with T=1 for images
        raw_data = x0.unsqueeze(2)
        return raw_data, x0, self._EmptyCondition()

    # =========================================================================
    # Training
    # =========================================================================

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        # CP setup: enable on the network (idempotent) and broadcast inputs
        # from the lowest-rank CP peer so every rank in a CP group sees the
        # exact same HQ image and captions. The dataloader produces an
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

        # 0. Resolve per-step shift. SD3 formula reads actual H, W from the batch
        # (which may vary per step when the dataloader does multi-resolution
        # bucketing). Falls back to config.shift if dynamic_shift is None.
        _shift = self.config.shift
        if self.config.dynamic_shift is not None:
            _raw = data_batch[self.config.input_data_key]
            _h, _w = _raw.shape[-2], _raw.shape[-1]
            _ds = self.config.dynamic_shift
            _shift = _ds["base_shift"] * math.sqrt(math.sqrt(_h * _w) / _ds["base_image_size_for_shift_calc"])

        # 1. Get and normalize image
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)

        # 2. Get captions with classifier-free guidance dropout (via conditioner)
        condition = self.conditioner(data_batch)

        captions = condition.caption
        # AR collator may collapse identical caption strings into a single str /
        # len-1 list — broadcast back to per-sample list of B for the encoder.
        B = x0.shape[0]
        if isinstance(captions, str):
            captions = [captions] * B
        elif isinstance(captions, list) and len(captions) == 1 and B > 1:
            captions = captions * B
        caption_embs, emb_masks = self._encode_text_raw(captions)
        caption_embs = caption_embs.to(**self.tensor_kwargs)

        # 3. Sample t and noise, then apply flow shift.
        # Under CP, t and the FM noise must be identical across CP ranks — we
        # pre-sample both and broadcast them, then pass `noise=` explicitly to
        # `fm_trainer.loss` to bypass its internal `randn_like`.
        t = self.fm_trainer.sample_t(x0.shape[0], device=x0.device)
        # Apply flow shift to match original PixelDiT training (SpacedDiffusion shift).
        # sigma' = shift * sigma / (1 + (shift-1) * sigma), with shift=4.0.
        # This biases noise distribution toward high noise: t=0.25→0.571, t=0.50→0.800.
        # The shifted t is used for BOTH noise mixing (x_t) and network timestep (t*1000).
        # Same pattern as ssdd_model.py train_time_shift.
        if _shift != 1.0:
            t = (_shift * t) / (1.0 + (_shift - 1.0) * t)
        noise = torch.randn_like(x0)
        if cp_group is not None and cp_group.size() > 1:
            t = self._broadcast_tensor_for_cp(t)
            noise = self._broadcast_tensor_for_cp(noise)

        # 4. Flow matching loss via FlowMatchingTrainer
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()

        def _net_fn(x_t, t, **kwargs):
            # velocity mode: net predicts v = noise - x0; FM trainer expects x0 - noise → negate.
            # x0 mode: net predicts x0 directly; FM trainer (prediction_type="x0") consumes it as-is.
            out = self.net(x_t, t, caption_embs)
            if self.config.prediction_type == "x0":
                return out
            elif self.config.prediction_type == "velocity":
                return -out
            else:
                raise ValueError(f"Invalid prediction type: {self.config.prediction_type}")

        with autocast_ctx:
            diff_loss, (x_t, noise, t, v_pred, x0_pred) = self.fm_trainer.loss(
                fn=_net_fn,
                x=x0,
                t=t,
                noise=noise,
            )

        loss_dict = {"diffusion_loss": diff_loss}
        total_loss = self.config.loss_weights.get("diffusion", 1.0) * diff_loss

        # 4. Optional REPA loss (hook was triggered during fm_trainer.loss -> _net_fn -> self.net())
        if self.repa_loss is not None:
            repa_loss = self.repa_loss(x0)
            loss_dict["repa_loss"] = repa_loss
            total_loss = total_loss + self.config.loss_weights.get("repa", 0.25) * repa_loss

        loss_dict["total_loss"] = total_loss

        output_batch = {"edm_loss": total_loss.detach(), "loss_dict": loss_dict}
        # Scale loss by cp_size before backward — every gradient flowing into
        # `self.net` is a 1/cp_size slice of the full-batch gradient (the
        # network gathers L tokens via `cat_outputs_cp_with_grad`, which
        # preserves grad only on the local slice). FSDP averages gradients over
        # `world_size`, so multiplying the loss by `cp_size` restores the
        # correct full-batch gradient. No-op when cp_size==1.
        total_loss = total_loss * self._cp_loss_scale
        return output_batch, total_loss

    def validation_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        return self.training_step(data_batch, iteration)

    def forward(self, x, t, y, **kwargs):
        """Direct network forward pass."""
        return self.net(x, t, y, **kwargs)

    # =========================================================================
    # Inference: generate_samples_from_batch
    # =========================================================================

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = None,
        cfg_scale: float = None,
        num_steps: int = None,
        seed: int = 0,
        image_size=None,
        shift: float = None,
        **kwargs,
    ) -> Tensor:
        """Generate images from text captions using DPM-Solver with CFG.

        Compatible with EveryNDrawSample callback interface which passes
        guidance, shift, num_steps.

        Args:
            data_batch: must contain self.config.input_caption_key (list[str])
            guidance: alias for cfg_scale (used by EveryNDrawSample callback)
            cfg_scale: classifier-free guidance scale (default: config.cfg_scale)
            num_steps: number of DPM-Solver steps (default: config.num_sample_steps)
            seed: random seed
            image_size: output resolution (default: config.image_size)
            shift: flow shift for DPM-Solver (default: config.shift)

        Returns:
            generated images [B, 3, H, W] in [-1, 1]
        """
        from pid._src.modules.dpmsolver import DPMS

        if guidance is not None:
            cfg_scale = guidance
        elif cfg_scale is None:
            cfg_scale = self.config.cfg_scale

        num_steps = num_steps or self.config.num_sample_steps
        _shift_override = shift  # None means "not explicitly passed"

        # Enable CP on the network and broadcast inputs from CP rank 0 so all
        # ranks step the same noise/timesteps and emit identical samples.
        self._maybe_enable_cp_on_nets([self.net])
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            if isinstance(data_batch.get(self.config.input_data_key), torch.Tensor):
                data_batch[self.config.input_data_key] = self._broadcast_tensor_for_cp(
                    data_batch[self.config.input_data_key]
                )
            data_batch[self.config.input_caption_key] = self._broadcast_object_for_cp(
                data_batch.get(self.config.input_caption_key)
            )

        # Infer generation size from batch if available, otherwise use config default
        x0_key = self.config.input_data_key
        if image_size is None and x0_key in data_batch:
            x0_shape = data_batch[x0_key].shape  # [B, C, H, W] or [B, C, 1, H, W]
            img_h, img_w = x0_shape[-2], x0_shape[-1]
        else:
            image_size = image_size or self.config.image_size
            # int -> square; subscriptable [H, W] (list / tuple / OmegaConf
            # ListConfig, which is not a `list` instance) -> rectangular.
            if isinstance(image_size, int):
                img_h = img_w = int(image_size)
            else:
                img_h, img_w = int(image_size[0]), int(image_size[1])

        # Determine shift: explicit arg > SD3 formula (if dynamic_shift) > config default
        if _shift_override is not None:
            shift = _shift_override
        elif self.config.dynamic_shift is not None:
            _ds = self.config.dynamic_shift
            shift = _ds["base_shift"] * math.sqrt(math.sqrt(img_h * img_w) / _ds["base_image_size_for_shift_calc"])
        else:
            shift = self.config.shift

        # When called from ema_scope, EMA weights are already in self.net.
        # Always use self.net — ema_scope handles the swap.
        net = self.net
        net.eval()

        # Encode captions (normalize to list[str] — webdataset collate returns str when batch_size=1)
        captions = data_batch[self.config.input_caption_key]
        if isinstance(captions, str):
            captions = [captions]
        B = len(captions)
        caption_embs, emb_masks = self._encode_text_raw(captions)  # [B, L, C]
        caption_embs = caption_embs.unsqueeze(1)  # [B, 1, L, C] for DPM-Solver wrapper

        # Null conditioning for CFG
        null_y = self._null_caption_embs.unsqueeze(1).repeat(B, 1, 1, 1)  # [B, 1, L, C]

        # Model wrapper that handles y squeeze (DPM-Solver passes y as [B, 1, L, C])
        model_dtype = next(net.parameters()).dtype

        def _forward_fn(x, timestep, y, mask=None, **kwargs):
            x = x.to(model_dtype)
            timestep = timestep.to(model_dtype)
            if y.dim() == 4:
                y = y.squeeze(1)
            y = y.to(model_dtype)
            return net(x, timestep, y)

        # Sample initial noise. Under CP, every rank must step the same z so
        # the gathered samples agree — broadcast from CP rank 0 after the draw.
        torch.manual_seed(seed)
        z = torch.randn(B, 3, img_h, img_w, device="cuda")
        if cp_group is not None and cp_group.size() > 1:
            z = self._broadcast_tensor_for_cp(z)

        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()

        with autocast_ctx:
            model_kwargs = dict(mask=emb_masks)
            # x0 mode: DPMS has a built-in "x_start" branch that converts x0 → noise internally
            # using the FLOW noise schedule's alpha_t=1-t, sigma_t=t; no wrapper change needed.
            if self.config.prediction_type == "x0":
                dpms_model_type = "x_start"
            elif self.config.prediction_type == "velocity":
                dpms_model_type = "flow"
            else:
                raise ValueError(f"Invalid prediction_type: {self.config.prediction_type}")
            dpm_solver = DPMS(
                _forward_fn,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=cfg_scale,
                model_type=dpms_model_type,
                guidance_type="classifier-free",
                model_kwargs=model_kwargs,
                schedule="FLOW",
                interval_guidance=[0, 1],
            )
            samples = dpm_solver.sample(
                z,
                steps=num_steps,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=shift,
            )

        # Add temporal dim [B, C, H, W] -> [B, C, 1, H, W] for callback compatibility
        return samples.clamp(-1, 1).unsqueeze(2)

    # =========================================================================
    # Optimizer / scheduler
    # =========================================================================

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        from pid._ext.imaginaire.utils.optim_instantiate import get_base_scheduler

        # Explicitly collect trainable modules: net + REPA projection MLP.
        # (Don't pass `self` — that would pull in the frozen text encoder params too,
        # and the HuggingFace tokenizer stored via object.__setattr__ is not relevant.)
        trainable_modules = [self.net]
        if self.repa_loss is not None:
            trainable_modules.append(self.repa_loss.repa_mlp)
        optim_model = torch.nn.ModuleList(trainable_modules)
        optimizer = lazy_instantiate(optimizer_config, model=optim_model)

        # get_base_scheduler stores the model ref in the scheduler for sample_counter
        # access, which makes LambdaLR unpicklable when the model holds HuggingFace
        # objects. Use a lightweight proxy that only exposes sample_counter.
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

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0, **kwargs) -> torch.Tensor:
        params = list(self.net.parameters())
        if self.repa_loss is not None:
            params += [p for p in self.repa_loss.repa_mlp.parameters() if p.requires_grad]
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type, **kwargs)

    def model_param_stats(self) -> dict[str, int]:
        return {"total_learnable_param_num": sum(p.numel() for p in self.net.parameters() if p.requires_grad)}

    def return_data_type(self, data_batch: dict) -> str:
        return "image"

    def is_image_batch(self, data_batch: dict) -> bool:
        return True

    # =========================================================================
    # Training hooks (EMA update)
    # =========================================================================

    def ema_beta(self, iteration: int) -> float:
        """Calculate EMA beta. Same formula as ssdd_model / wan_t2v_model."""
        iteration = iteration + self.config.ema.iteration_shift
        if iteration <= 0:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        """Temporarily swap net weights with EMA weights for inference."""
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled and hasattr(self, "net_ema"):
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def on_before_zero_grad(self, optimizer, scheduler, iteration):
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(src_model=self.net, tgt_model=self.net_ema, beta=ema_beta)

    # =========================================================================
    # Checkpoint save/load
    # =========================================================================

    def state_dict(self, *args, **kwargs):
        sd = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            sd.update(self.net_ema.state_dict(prefix="net_ema."))
        if self.repa_loss is not None:
            sd.update(self.repa_loss.state_dict(prefix="repa_loss."))
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        """Load checkpoint. Handles both our format and original PixelDiT format.

        Original PixelDiT checkpoint keys:
        - "core.*" -> maps to self.net (strip "core." prefix)
        - "_repa_projector.*" -> maps to self.repa_loss.repa_mlp (strip prefix)

        Our checkpoint keys:
        - "net.*" -> self.net
        - "net_ema.*" -> self.net_ema
        - "repa_loss.*" -> self.repa_loss
        """
        # Detect format: original PixelDiT vs our format
        has_core_keys = any(k.startswith("core.") for k in state_dict)
        has_net_keys = any(k.startswith("net.") for k in state_dict)

        if has_core_keys and not has_net_keys:
            # Original PixelDiT checkpoint (from PixDiTTrainer wrapper)
            logger.info("Loading original PixelDiT checkpoint (core.* prefix)")
            net_sd = {}
            repa_sd = {}
            for k, v in state_dict.items():
                if k == "pos_embed":
                    continue
                if k.startswith("core."):
                    net_sd[k[len("core.") :]] = v
                elif k.startswith("_repa_projector."):
                    # Map _repa_projector.N.weight -> repa_mlp.N.weight
                    new_key = k.replace("_repa_projector.", "repa_mlp.")
                    repa_sd[new_key] = v
            missing, unexpected = self.net.load_state_dict(net_sd, strict=False, assign=assign)
            if missing:
                logger.warning(f"Missing keys in net: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in net: {unexpected}")
            if self.repa_loss is not None and repa_sd:
                self.repa_loss.load_state_dict(repa_sd, strict=False, assign=assign)
            # Copy to EMA
            if self.config.ema.enabled and hasattr(self, "net_ema"):
                self.net_ema.load_state_dict(net_sd, strict=False, assign=assign)
        else:
            # Our checkpoint format
            _net_sd = {
                k[len("net.") :]: v
                for k, v in state_dict.items()
                if k.startswith("net.") and not k.startswith("net_ema.")
            }
            _ema_sd = {k[len("net_ema.") :]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
            _repa_sd = {k[len("repa_loss.") :]: v for k, v in state_dict.items() if k.startswith("repa_loss.")}

            if _net_sd:
                self.net.load_state_dict(_net_sd, strict=strict, assign=assign)
            if _ema_sd and self.config.ema.enabled and hasattr(self, "net_ema"):
                self.net_ema.load_state_dict(_ema_sd, strict=False, assign=assign)
            if _repa_sd and self.repa_loss is not None:
                self.repa_loss.load_state_dict(_repa_sd, strict=False, assign=assign)

    def enable_compile(self) -> None:
        """Arm torch.compile for the text encoder and `self.net`.

        The text encoder does not depend on the output resolution, so install its
        compile wrapper immediately. Its actual compilation remains lazy and happens
        on the first caption encode. The pixel net is wrapped later, once the output
        (H, W) is known (see `_maybe_compile_net`), and cached per resolution.

        Standard SR path only — context-parallel and the encoder-decoder path are not
        supported under compile.
        """
        assert not getattr(self.net, "is_context_parallel_enabled", False) and self.net._cp_group is None, (
            "--compile is incompatible with context parallel; disable CP first."
        )
        assert not getattr(self.net, "use_ed", False), "--compile does not support the encoder-decoder net path."
        if not hasattr(self, "_compiled_nets"):
            self._compiled_nets = {}
        if not hasattr(self, "_text_encoder_compiled"):
            self._text_encoder_compiled = False
        if not self._text_encoder_compiled:
            # text_encoder was installed via object.__setattr__ in PixelDiTModel.__init__
            # to bypass DCP state-dict registration; mirror that bypass after wrapping.
            object.__setattr__(
                self,
                "text_encoder",
                torch.compile(self.text_encoder, mode="default", dynamic=False),
            )
            self._text_encoder_compiled = True
            logger.info("PixelDiTModel: text_encoder wrapped for torch.compile (lazy, first caption encode).")
        self._compile_enabled = True
        logger.info("PixelDiTModel: torch.compile armed for net (lazy, per output resolution).")

    def _maybe_compile_net(self, image_h: int, image_w: int, text_len: int):
        """Return the net to run for this shape: a torch.compile-wrapped net when
        `--compile` is on (compiled once per (H, W) and cached), else the eager net."""
        del text_len  # The release PixelDiT net is compile-safe without positional cache prewarm.
        if not getattr(self, "_compile_enabled", False):
            return self.net
        compile_mode = "default"
        dynamic = False
        key = (int(image_h), int(image_w))
        compiled_nets = getattr(self, "_compiled_nets", None)
        if compiled_nets is None:
            compiled_nets = {}
            self._compiled_nets = compiled_nets
        compiled = compiled_nets.get(key)
        if compiled is None:
            logger.info(f"--compile: compiling net for {image_h}x{image_w}")
            # mode="default": fast compile, solid speedup. For more inference throughput
            # at the cost of a much slower first compile, switch to "max-autotune".
            compiled = torch.compile(self.net, mode=compile_mode, dynamic=dynamic)
            compiled_nets[key] = compiled
        return compiled
