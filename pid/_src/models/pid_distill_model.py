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
PixelDiT SR DMD distillation model — 1-step pixel-space super-resolution.

Implements DMD (Distribution Matching Distillation) to convert the multi-step
PixelDiTSRModel into a 1-step generator. Maintains three networks:
  - Student (self.net):       Trainable, becomes the 1-step generator.
  - Teacher (self.teacher):   Frozen pretrained multi-step SR model.
  - Fake Score (self.fake_score): Trainable, learns student's output distribution.

Training modes:
  - DMD2 alternating (default, joint_update=False):
      Student update (VSD) and fake_score update (DSM) alternate per student_update_freq.
Key differences from FlashVSRDistillModel:
  - Pixel space: x0 is [B, 3, H, W], not latent [B, C, T, H, W]
  - DDP (not FSDP): no distributed sharding logic
  - Net forward: net(x_t, t_scaled, caption_embs, lq_video_or_image=None, lq_latent=..., degrade_sigma=...)
  - Net output convention (per-net, controlled by prediction_type / teacher_prediction_type):
      "velocity" (default): net returns noise - x0; x0 = x_t - t * net_output.
      "x0" (JiT paradigm):  net returns x0 directly. Helpers: _net_output_to_x0 /
                            _net_output_to_velocity convert on demand. DSM loss
                            automatically switches to x0-space MSE when student prediction_type="x0".
  - Timestep range: t ∈ [0, 1]; student_timestep=1.0 (net sees 1000.0 after scaling)
  - GAN: config fields present (discriminator_lr=1e-5) but disabled by default
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Optional

import attrs
import torch
import torch.nn.functional as F

from pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from pid._ext.imaginaire.utils import misc
from pid._ext.imaginaire.utils.optim_instantiate import get_base_scheduler
from pid._src.losses.dmd_losses import (
    denoising_score_matching_loss_flow,
    gan_loss_discriminator,
    gan_loss_generator,
    variational_score_distillation_loss,
)
from pid._src.models.pid_model import PidModel, PidModelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@attrs.define(slots=False)
class PidDistillModelConfig(PidModelConfig):
    """Configuration for PixelDiT SR DMD distillation model."""

    # Path to pretrained teacher checkpoint (required).
    # Supports DCP directory format (iter_XXXXXX/model/) or regular state_dict.
    pretrained_teacher_path: str = ""

    # Training schedule: student updates every N iterations, fake_score on others (DMD2 alternating).
    student_update_freq: int = 5

    # Loss weights
    vsd_loss_weight: float = 1.0
    dsm_loss_weight: float = 1.0

    # Student timestep for 1-step generation (in [0,1] scale).
    # The network receives student_timestep * fm_timescale (e.g. 1.0 * 1000 = 1000.0).
    student_timestep: float = 1.0

    # Few-step distillation: student denoises in N steps instead of 1.
    # student_sample_steps=1 preserves the original 1-step behavior (pure noise → x0).
    # When > 1: training samples random intermediate t from t_list; inference runs N steps.
    student_sample_steps: int = 1

    # Inference sampling type for multi-step (student_sample_steps > 1):
    #   "sde": each step converts v→x0, then re-noises with fresh random eps (stochastic)
    #   "ode": Euler step x_{t_next} = x_t + (t_next - t_cur) * v_pred (deterministic, no x0 needed)
    student_sample_type: str = "sde"

    # Explicit t schedule for multi-step: list of length student_sample_steps+1, must end at 0.
    # If None, auto-generates uniform linspace from student_timestep down to 0.
    # Example for 4 steps: [0.999, 0.749, 0.499, 0.249, 0.0]
    student_t_list: Optional[list] = None

    # Controls how input_student is constructed when student_sample_steps > 1.
    # Has no effect for student_sample_steps == 1 (always pure noise → x0 in one pass).
    #   "teacher_forcing":  real data + noise at a random intermediate t from t_list. Efficient
    #                       (1 forward pass per step), but trains on real-data-derived states
    #                       while inference sees student-derived states — classic exposure bias.
    #   "pidself_rollout": DMD2-equivalent. Randomly pick grad_step_idx ∈ [0, K), do
    #                       that many no_grad rollout steps from pure noise, then a single with_grad
    #                       forward at t_list[grad_step_idx]. Broadcast grad_step_idx within each
    #                       context-parallel group so each CP group supervises at the same t
    #                       per iteration. Different
    #                       data-parallel groups may cover different t values. BOTH student-VSD
    #                       AND fake_score-DSM paths use the same
    #                       random-k partial-rollout x0 — critical for VSD correctness: fake_score
    #                       must be trained on the same x0 distribution VSD queries it on.
    #                       Inference-consistent input distribution AND t-uniform gradient signal.
    #                       Avg K/2 + 0.5 forwards per update.
    student_input_mode: str = "teacher_forcing"

    # Fake score optimizer settings
    fake_score_lr: float = 1e-5
    fake_score_weight_decay: float = 1e-3
    fake_score_betas: tuple = (0.9, 0.999)

    # DMD timestep clamping for perturbation step (in [0,1] scale).
    # Default (0.0, 1.0) means no clamping. E.g. (0.02, 0.98) avoids extremes.
    dmd_timestep_clamp_min: float = 0.02
    dmd_timestep_clamp_max: float = 0.98

    # Teacher CFG scale during distillation training (1.0 = no CFG, single conditional forward).
    # When > 1.0: two teacher forwards per step — v = v_uncond + scale*(v_cond - v_uncond).
    # This makes the VSD loss target the teacher's CFG-guided distribution rather than its
    # raw conditional output, so the student learns to replicate guidance-amplified results.
    teacher_cfg_scale: float = 1.0

    # GAN loss settings (disabled by default when gan_loss_weight_gen=0)
    gan_loss_weight_gen: float = 0.05
    gan_warmup_steps: int = 0  # linear warmup for generator loss weight
    gan_use_same_t_noise: bool = True  # reuse same t/noise for real and fake samples
    gan_r1_reg_weight: float = 0.0  # R1 regularization weight (0 = disabled)
    gan_r1_reg_alpha: float = 0.1  # noise scale for R1 approximation
    discriminator_lr: float = 1e-5
    discriminator_weight_decay: float = 0.0

    # Network configs for teacher and fake_score.
    # net_teacher: if None, uses same arch as config.net.
    # net_fake_score: if None, uses same arch as teacher.
    # net_discriminator: LazyDict for future GAN discriminator.
    net_teacher: Any = None
    net_fake_score: Any = None
    net_discriminator: Any = None

    # teacher_prediction_type: "velocity" (teacher's net outputs noise - x0; current default)
    #   or "x0" (teacher's net outputs x0 directly — use this after Experiment A produces an
    #   x0-reparam T2I teacher ckpt). `prediction_type` (inherited from PixelDiTSRModelConfig)
    #   applies to the student `self.net` + `self.fake_score` (same architecture, init from teacher).
    #   In x0 mode the DSM loss is computed in x0 space (MSE(x0_fake, x0_student)) — no 1/t divide.
    teacher_prediction_type: str = "velocity"


# =============================================================================
# Model
# =============================================================================


class PidDistillModel(PidModel):
    """PixelDiT SR DMD distillation model.

    Extends PixelDiTSRModel with DMD distillation for 1-step pixel-space SR.

    Three networks:
    - self.net (student): Trainable, will become 1-step generator
    - self.teacher: Frozen, pretrained multi-step SR model
    - self.fake_score: Trainable, learns student's output distribution
    """

    @staticmethod
    def _cfg_get_optional(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _cfg_has_key(config: Any, key: str) -> bool:
        if config is None:
            return False
        if isinstance(config, dict):
            return key in config
        return hasattr(config, key)

    @staticmethod
    def _cfg_set(config: Any, key: str, value: Any) -> None:
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)

    @classmethod
    def _assert_and_disable_distill_lq_aux_rgb_heads(cls, config: PidDistillModelConfig) -> None:
        # Distillation does not train the LQ latent-image alignment objective.
        # Keep this invariant explicit and force-disable the aux RGB head before
        # student/EMA/teacher/fake_score are built, otherwise the unused head can
        # enter optimizer param groups without Adam state and break strict DCP resume.
        align_cfg = config.lq_latent_image_align_config
        assert not bool(cls._cfg_get_optional(align_cfg, "enabled", False)), (
            "PidDistillModel requires lq_latent_image_align_config.enabled=False. "
            "The distill path never builds or trains lq_aux_rgb_head."
        )

        changed = []
        for cfg_name in ("net", "net_teacher", "net_fake_score"):
            net_cfg = getattr(config, cfg_name, None)
            if net_cfg is None or not cls._cfg_has_key(net_cfg, "lq_aux_rgb_head"):
                continue
            if bool(cls._cfg_get_optional(net_cfg, "lq_aux_rgb_head", False)):
                cls._cfg_set(net_cfg, "lq_aux_rgb_head", False)
                changed.append(cfg_name)

        if changed:
            logger.info(
                "Disabled lq_aux_rgb_head for %s because PidDistillModel does not build the aux RGB head.",
                ", ".join(changed),
            )

    def __init__(self, config: PidDistillModelConfig):
        # Initialize teacher/fake_score/discriminator BEFORE super().__init__() so
        # they exist if any parent code checks for them.
        self.teacher = None
        self.fake_score = None
        self.discriminator = None
        self._current_update_type = "student"

        self._assert_and_disable_distill_lq_aux_rgb_heads(config)

        # Parent builds self.net, self.vae_encoder, text encoder, etc.
        super().__init__(config)

        # Build teacher and fake_score after net is ready.
        self._build_teacher_and_fake_score()

        # Build discriminator if GAN is enabled.
        if self.config.gan_loss_weight_gen > 0 and self.config.net_discriminator:
            self._build_discriminator()

    # =========================================================================
    # Model construction
    # =========================================================================

    def _build_teacher_and_fake_score(self):
        """Build teacher (frozen) and fake_score (trainable) from the same architecture."""
        logger.info("Building teacher and fake_score networks for DMD distillation...")

        # Build teacher
        teacher_cfg = self.config.net_teacher if self.config.net_teacher else self.config.net
        with misc.timer("PidDistillModel: build teacher"):
            self.teacher = lazy_instantiate(teacher_cfg)
            self.teacher = self.teacher.to(device="cuda", dtype=torch.float32)
            if hasattr(self.teacher, "init_weights"):
                self.teacher.init_weights()

        # Load pretrained weights
        if self.config.pretrained_teacher_path:
            self._load_teacher_checkpoint()
        else:
            logger.warning("No pretrained_teacher_path specified! Teacher will use random weights.")

        # Freeze teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        logger.info(f"Teacher params: {teacher_params:,} (frozen)")

        # Build fake_score
        fake_score_cfg = self.config.net_fake_score if self.config.net_fake_score else teacher_cfg
        with misc.timer("PidDistillModel: build fake_score"):
            self.fake_score = lazy_instantiate(fake_score_cfg)
            self.fake_score = self.fake_score.to(device="cuda", dtype=torch.float32)
            if hasattr(self.fake_score, "init_weights"):
                self.fake_score.init_weights()

        # Copy teacher weights into fake_score (same architecture)
        if not self.config.net_fake_score:
            self.fake_score.load_state_dict(self.teacher.state_dict(), strict=False)
            logger.info("Copied teacher weights to fake_score (same architecture)")
        else:
            logger.info("Fake_score uses different architecture, starting from random weights")

        self.fake_score.requires_grad_(True)
        if getattr(self.fake_score, "patch_blocks", None):
            last_patch_block = self.fake_score.patch_blocks[-1]
            if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                last_patch_block.freeze_unused_text_output_branch()
        self.fake_score.train()

        fs_params = sum(p.numel() for p in self.fake_score.parameters())
        logger.info(f"Fake_score params: {fs_params:,} (trainable)")

    def _load_teacher_checkpoint(self):
        """Load pretrained checkpoint into teacher.

        Handles:
        - DCP directory format (iter_XXXXXX/ or iter_XXXXXX/model/)
        - Regular .pth state dict (our net.* prefix or original core.* prefix)
        """
        path = self.config.pretrained_teacher_path
        logger.info(f"Loading teacher checkpoint from: {path}")

        if not path.endswith(".pth") and not path.endswith(".pt"):
            # DCP directory format
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint import load as dcp_load
            from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

            ckpt_model_path = path if path.endswith("model") else f"{path.rstrip('/')}/model"

            # Build state dict with net. prefix to match DCP format
            teacher_state = self.teacher.state_dict()

            prefixed_state = {}
            for k, v in teacher_state.items():
                if "_extra_state" not in k:
                    prefixed_state[f"net.{k}"] = v
            # Also register net_ema. keys so EMA weights get loaded (preferred)
            for k in list(prefixed_state.keys()):
                prefixed_state[k.replace("net.", "net_ema.")] = prefixed_state[k]

            load_planner = DefaultLoadPlanner(allow_partial_load=True)
            storage_reader = FileSystemReader(ckpt_model_path)
            try:
                dcp_load(prefixed_state, storage_reader=storage_reader, planner=load_planner)
            except Exception as e:
                logger.warning(f"DCP load failed ({e}), trying without model/ suffix...")
                storage_reader = FileSystemReader(path.rstrip("/"))
                dcp_load(prefixed_state, storage_reader=storage_reader, planner=load_planner)

            # Prefer EMA: copy net_ema.* -> net.*
            for k in list(prefixed_state.keys()):
                if k.startswith("net.") and not k.startswith("net_ema."):
                    ema_k = k.replace("net.", "net_ema.")
                    if ema_k in prefixed_state:
                        prefixed_state[k] = prefixed_state[ema_k]

            # Strip net. prefix and load
            loaded = {
                k[len("net.") :]: v
                for k, v in prefixed_state.items()
                if k.startswith("net.") and not k.startswith("net_ema.")
            }
            missing, unexpected = self.teacher.load_state_dict(loaded, strict=False)
            if missing:
                logger.warning(f"Teacher missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                logger.warning(f"Teacher unexpected keys: {len(unexpected)}")
            logger.info("Teacher loaded from DCP checkpoint (preferring EMA weights)")

        else:
            # Regular .pth state dict
            state_dict = torch.load(path, map_location="cpu")
            # Handle both our format (net.*) and original PixelDiT format (core.*)
            if any(k.startswith("core.") for k in state_dict):
                # Original PixelDiT checkpoint
                loaded = {}
                for k, v in state_dict.items():
                    if k.startswith("core.") and k != "pos_embed":
                        loaded[k[len("core.") :]] = v
                missing, unexpected = self.teacher.load_state_dict(loaded, strict=False)
            elif any(k.startswith("net.") for k in state_dict):
                # Our format: prefer net_ema if available
                ema_sd = {k[len("net_ema.") :]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
                net_sd = {
                    k[len("net.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith("net.") and not k.startswith("net_ema.")
                }
                loaded = ema_sd if ema_sd else net_sd
                missing, unexpected = self.teacher.load_state_dict(loaded, strict=False)
            else:
                # Bare state dict
                missing, unexpected = self.teacher.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Teacher missing keys: {len(missing)}")
            logger.info("Teacher loaded from .pth checkpoint")

    # =========================================================================
    # Discriminator construction and GAN utilities
    # =========================================================================

    def _build_discriminator(self):
        """Build lightweight discriminator head on teacher intermediate features.

        No FSDP needed — PixelDiT uses DDP. Discriminator is kept in float32
        to match teacher features (also float32).
        """
        logger.info("Building discriminator for GAN loss...")
        self.discriminator = lazy_instantiate(self.config.net_discriminator)
        self.discriminator = self.discriminator.to(device="cuda", dtype=torch.float32)
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info(f"Discriminator params: {disc_params:,}, feature_indices={self.discriminator.feature_indices}")

    def _get_gan_weight(self, iteration: int) -> float:
        """GAN generator loss weight with optional linear warmup."""
        w = self.config.gan_loss_weight_gen
        if self.config.gan_warmup_steps > 0 and iteration < self.config.gan_warmup_steps:
            w = w * iteration / self.config.gan_warmup_steps
        return w

    def _compute_real_feat(
        self,
        x0_gt: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ):
        """Perturb GT image and extract teacher features for discriminator real samples.

        Args:
            x0_gt:   Ground truth HQ image, [B, 3, H, W] in [-1, 1].
            t:       Perturbation timestep from current fake update, shape [B], in [0, 1].
            eps:     Noise from current fake update, same shape as x0_gt.

        Returns:
            (real_features, t_real, perturbed_real)
            - real_features: list of [B, D, 1, Hs, Ws] tensors
            - t_real:        timestep used for real perturbation, shape [B]
            - perturbed_real: noised GT used (needed for optional R1 regularization)
        """
        if self.config.gan_use_same_t_noise:
            t_real = t
            eps_real = eps
        else:
            t_real = self.fm_trainer.sample_t(x0_gt.shape[0], device=x0_gt.device)
            if self.config.dmd_timestep_clamp_min > 0 or self.config.dmd_timestep_clamp_max < 1.0:
                t_real = t_real.clamp(self.config.dmd_timestep_clamp_min, self.config.dmd_timestep_clamp_max)
            eps_real = torch.randn_like(x0_gt)
            t_real = self._broadcast_tensor_for_cp(t_real)
            eps_real = self._broadcast_tensor_for_cp(eps_real)

        s = [x0_gt.shape[0]] + [1] * (x0_gt.ndim - 1)
        t_bcast = t_real.view(*s)
        perturbed_real = (1.0 - t_bcast) * x0_gt + t_bcast * eps_real
        t_scaled = t_real * self.fm_trainer.timescale

        real_feat = self.teacher(
            perturbed_real.to(**self.tensor_kwargs),
            t_scaled,
            caption_embs,
            lq_video_or_image=None,
            lq_latent=lq_latent,
            degrade_sigma=degrade_sigma,
            feature_indices=self.discriminator.feature_indices,
            return_features_early=True,
        )
        return real_feat, t_real, perturbed_real

    def _compute_r1_regularization(
        self,
        real_logits: torch.Tensor,
        perturbed_real: torch.Tensor,
        t_real: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Approximate R1 regularization in noised input space.

        Adds small noise to already-perturbed real, re-extracts teacher features, and
        penalizes discriminator logit change via MSE. Matches APT/FastGen implementation.
        """
        perturbed_alpha = perturbed_real + self.config.gan_r1_reg_alpha * torch.randn_like(perturbed_real)
        t_scaled = t_real * self.fm_trainer.timescale
        with torch.no_grad():
            real_feat_alpha = self.teacher(
                perturbed_alpha.to(**self.tensor_kwargs),
                t_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
                feature_indices=self.discriminator.feature_indices,
                return_features_early=True,
            )
        real_logits_alpha = self.discriminator(real_feat_alpha)
        return F.mse_loss(real_logits, real_logits_alpha, reduction="mean")

    # =========================================================================
    # Few-step sampling helpers
    # =========================================================================

    def _get_t_list(self, device, num_steps: Optional[int] = None) -> torch.Tensor:
        """Return the t schedule for multi-step student sampling.

        When num_steps is None, uses student_sample_steps from config (training schedule).
        When num_steps is given and differs from student_sample_steps, subsamples
        student_t_list evenly so that num_steps+1 points are taken (always keeping
        the first and last entries).  Example with [0.999, 0.866, 0.634, 0.342, 0.0]:
          num_steps=2 → indices [0,2,4] → [0.999, 0.634, 0.0]
          num_steps=1 → indices [0,4]   → [0.999, 0.0]
        """
        target_steps = num_steps if num_steps is not None else self.config.student_sample_steps

        if self.config.student_t_list is not None:
            full_t = torch.tensor(self.config.student_t_list, device=device, dtype=torch.float32)
            if target_steps != self.config.student_sample_steps:
                # Subsample evenly: pick target_steps+1 points from the full schedule
                indices = torch.linspace(0, len(full_t) - 1, target_steps + 1).round().long()
                t_list = full_t[indices]
            else:
                t_list = full_t
        else:
            t_list = torch.linspace(
                self.config.student_timestep,
                0.0,
                target_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        assert abs(t_list[-1].item()) < 1e-6, "t_list must end at 0"
        if num_steps is not None:
            logger.info(f"[distill inference] num_steps={num_steps}, t_list={t_list.tolist()}")
        return t_list

    def _student_sample_1step(
        self,
        noise: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """1-step student forward: pure noise → x0 in a single net call.

        The full pipeline — `t_student` construction, `timescale` multiply,
        autocast, net forward, and v→x0 conversion — lives inside this helper.
        """
        B = noise.shape[0]
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        with autocast_ctx:
            t_student = torch.full((B,), self.config.student_timestep, device=noise.device, dtype=torch.float32)
            t_student_scaled = t_student * self.fm_trainer.timescale
            v_student = self.net(
                noise,
                t_student_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            return self._velocity_to_x0(noise, v_student, t_student)

    def _student_sample_loop(
        self,
        noise: torch.Tensor,
        num_steps: int,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Iterative multi-step denoising for inference (num_steps > 1).

        Runs `num_steps` denoising steps: noise → x0. With N = num_steps, we run
        N-1 intermediate iterations then one final step that emits clean x0.

        SDE re-noise uses `torch.randn_like` (no explicit Generator) to stay
        compatible with the existing seeding path; see `generate_samples_from_batch`.
        """
        t_list = self._get_t_list(noise.device, num_steps=num_steps)
        B = noise.shape[0]
        timescale = self.fm_trainer.timescale
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        x = noise  # t_list[0] ≈ 1.0, so pure noise is a valid initialization
        net = self.net

        with autocast_ctx:
            # Intermediate steps: all pairs except the last (which terminates at t=0).
            # For t_list of length N+1 this is N-1 iterations.
            for i in range(t_list.shape[0] - 2):
                t_cur = t_list[i]
                t_next = t_list[i + 1]
                t_cur_batch = t_cur.expand(B)
                t_cur_scaled = t_cur_batch * timescale

                v_pred = net(
                    x,
                    t_cur_scaled,
                    caption_embs,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma_tensor,
                )

                if self.config.student_sample_type == "ode":
                    v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, self.config.prediction_type)
                    dt = t_next - t_cur
                    x = x + dt * v_for_step
                else:  # "sde"
                    x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn_like(x0_pred)
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer

            # Final step: t_next == 0 implicitly. Both "ode" and "sde" collapse to
            # the same v -> x0 conversion at t_next == 0.
            t_cur = t_list[-2]
            t_cur_batch = t_cur.expand(B)
            t_cur_scaled = t_cur_batch * timescale
            v_pred = net(
                x,
                t_cur_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            x = self._velocity_to_x0(x, v_pred, t_cur_batch)

        return x

    def _pidself_rollout_x0_student(
        self,
        noise: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
        with_grad: bool,
    ) -> torch.Tensor:
        """Self-rollout with t-supervision + fake_score/VSD distribution match (Method A / DMD2-equivalent).

        Both `with_grad=True` (student VSD) and `with_grad=False` (fake_score DSM) draw a
        random `grad_step_idx ∈ [0, K)` and broadcast it within each context-parallel group.
        Different data-parallel groups may draw different steps. `with_grad` only
        controls whether grad is enabled on the chosen step, not which step is chosen.
        Statistically the marginal x0 distribution for both paths becomes "uniform-k
        partial rollout output", matching what VSD queries fake_score on.

        Two issues this fixes (both observed in prior attempts):

        1. A pure self-rollout that always tracks grad at the final non-zero step
           `t_list[-2]` supervises VSD at only one t value — high-t behavior is never
           directly trained. Random-k restores t-uniform supervision.

        2. Forcing `grad_step_idx = K-1` for the fake_score path while the student path
           draws random k caused fake_score to be trained on the **full** K-step rollout x0
           while VSD queried it on **random-k partial** rollout x0. The distribution mismatch
           returned unreliable scores → wrong VSD gradient → student collapsed to black
           around iter ~3000. DMD2 official (`sd_unified_model.py` + `sd_guidance.py`) reuses
           the same `generated_image` from a single `sample_backward` for both VSD and DSM
           paths — both see the partial-rollout x0.

        BPTT through the prior no_grad rollout is blocked by `x.detach()` plus the
        `torch.no_grad()` context, so the gradient-tracked computation is exactly one
        net forward.
        """
        B = noise.shape[0]
        timescale = self.fm_trainer.timescale
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        t_list = self._get_t_list(noise.device)
        K = t_list.shape[0] - 1  # number of denoising steps (t_list has K+1 entries, last == 0)

        # Random k for BOTH student and fake_score paths — see docstring bug #2.
        # Broadcast inside each CP group: CP peers must execute the same number of
        # collectives, while different data-parallel groups can cover different k.
        grad_step_idx_tensor = torch.randint(0, K, (1,), device=noise.device, dtype=torch.long)
        grad_step_idx_tensor = self._broadcast_tensor_for_cp(grad_step_idx_tensor)
        grad_step_idx = int(grad_step_idx_tensor.item())

        x = noise
        with autocast_ctx:
            for step_idx in range(K):
                t_cur = t_list[step_idx]
                t_next = t_list[step_idx + 1]
                t_cur_batch = t_cur.expand(B)
                t_cur_scaled = t_cur_batch * timescale

                if step_idx == grad_step_idx:
                    if with_grad:
                        # The only forward that contributes gradient. x is detached so BPTT
                        # cannot leak back through the prior no_grad rollout.
                        v_pred = self.net(
                            x.detach(),
                            t_cur_scaled,
                            caption_embs,
                            lq_video_or_image=None,
                            lq_latent=lq_latent,
                            degrade_sigma=degrade_sigma,
                        )
                        return self._velocity_to_x0(x.detach(), v_pred, t_cur_batch)
                    # fake_score path: same step but no gradient anywhere.
                    with torch.no_grad():
                        v_pred = self.net(
                            x,
                            t_cur_scaled,
                            caption_embs,
                            lq_video_or_image=None,
                            lq_latent=lq_latent,
                            degrade_sigma=degrade_sigma,
                        )
                        return self._velocity_to_x0(x, v_pred, t_cur_batch)

                # Prior no_grad rollout step: advance x toward t_next via SDE re-noise (or ODE).
                with torch.no_grad():
                    v_pred = self.net(
                        x,
                        t_cur_scaled,
                        caption_embs,
                        lq_video_or_image=None,
                        lq_latent=lq_latent,
                        degrade_sigma=degrade_sigma,
                    )
                    x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.view(*s)
                    if self.config.student_sample_type == "sde":
                        eps_infer = torch.randn_like(x0_pred)
                        eps_infer = self._broadcast_tensor_for_cp(eps_infer)
                        x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
                    else:  # "ode"
                        v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, self.config.prediction_type)
                        dt = t_next - t_cur
                        x = x + dt * v_for_step

        raise RuntimeError("grad_step_idx out of range; should not reach here.")

    # =========================================================================
    # Net output ↔ (x0, velocity) conversion
    #
    # Velocity mode: net_output = noise - x0 (current PixelDiT convention).
    #   Flow matching: x_t = (1-t)*x0 + t*noise  →  x0 = x_t - t * net_output.
    # x0 mode (JiT paradigm): net_output = x0 directly; velocity derived as (x_t - x0)/t.
    # Each net (student, fake_score, teacher) may have its own prediction_type, so call
    # sites pass the explicit mode: self.config.prediction_type for student/fake_score,
    # self.config.teacher_prediction_type for the teacher.
    # =========================================================================

    def _net_output_to_x0(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "x0":
            return net_output.to(x_t.dtype)
        elif prediction_type == "velocity":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s)
            x0 = x_t.double() - t_shaped * net_output.double()
            return x0.to(original_dtype)
        else:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _net_output_to_velocity(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        """Return net output expressed in the 'noise - x0' (PixelDiT) velocity convention."""
        if prediction_type == "velocity":
            return net_output
        elif prediction_type == "x0":
            # v = (x_t - x0_pred) / t; clamp t at 5e-2 to avoid 1/t blowup (matches
            # FlowMatchingTrainer.loss in flow_matching.py:152).
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s).clamp(min=5e-2)
            return ((x_t.double() - net_output.double()) / t_shaped).to(original_dtype)
        else:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _velocity_to_x0(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Shim for student / fake_score call sites: routes through `_net_output_to_x0` with
        `self.config.prediction_type`. The teacher path calls `_net_output_to_x0` directly
        with `self.config.teacher_prediction_type`."""
        return self._net_output_to_x0(x_t, net_output, t, self.config.prediction_type)

    def _teacher_cfg_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_scaled: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
        v_cond_precomputed: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Return detached x0_teacher with optional CFG weighting.

        When teacher_cfg_scale == 1.0: single conditional forward (original behaviour).
        When teacher_cfg_scale > 1.0: double forward — conditional + unconditional (null text),
            then v = v_uncond + scale * (v_cond - v_uncond).

        Args:
            x_t:                 Perturbed input [B, C, H, W].
            t:                   Perturbation timestep in [0, 1], shape [B].
            t_scaled:            t * fm_timescale, shape [B].
            caption_embs:        Conditional text embeddings [B, L, C].
            v_cond_precomputed:  Pre-computed conditional velocity from an earlier teacher
                                 forward (GAN path). Skips the conditional forward when given.

        Returns:
            x0_teacher, shape [B, C, H, W], always detached.
        """
        cfg = self.config.teacher_cfg_scale if cfg_scale is None else float(cfg_scale)

        with torch.no_grad():
            if v_cond_precomputed is None:
                v_cond = self.teacher(
                    x_t,
                    t_scaled,
                    caption_embs,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
            else:
                v_cond = v_cond_precomputed.detach()

            if cfg == 1.0:
                v_teacher = v_cond
            else:
                B = x_t.shape[0]
                null_embs = self._null_caption_embs.expand(B, -1, -1).to(
                    device=caption_embs.device, dtype=caption_embs.dtype
                )
                v_uncond = self.teacher(
                    x_t,
                    t_scaled,
                    null_embs,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
                v_teacher = v_uncond + cfg * (v_cond - v_uncond)

        return self._net_output_to_x0(x_t, v_teacher, t, self.config.teacher_prediction_type).detach()

    def _compute_student_gan_targets_and_loss(
        self,
        x_t_perturbed: torch.Tensor,
        t: torch.Tensor,
        t_scaled: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-feature GAN path used by the default FastGen-style implementation.

        Subclasses can override this hook to keep the VSD teacher target while
        changing only the adversarial feature extractor.
        """
        # Teacher forward without no_grad: frozen weights act as differentiable
        # feature extractor for GAN generator loss.
        teacher_result = self.teacher(
            x_t_perturbed.to(**self.tensor_kwargs),
            t_scaled,
            caption_embs,
            lq_video_or_image=None,
            lq_latent=lq_latent,
            degrade_sigma=degrade_sigma,
            feature_indices=self.discriminator.feature_indices,
        )
        v_cond, teacher_features = teacher_result

        # CFG-weighted x0_teacher for VSD. v_cond is detached inside
        # _teacher_cfg_x0; the GAN gradient flows separately through
        # teacher_features -> discriminator -> student.
        x0_teacher = self._teacher_cfg_x0(
            x_t_perturbed.to(**self.tensor_kwargs),
            t,
            t_scaled,
            caption_embs,
            lq_latent,
            degrade_sigma,
            v_cond_precomputed=v_cond,
        )

        gan_loss_gen = gan_loss_generator(self.discriminator(teacher_features))

        with torch.no_grad():
            v_fake = self.fake_score(
                x_t_perturbed.to(**self.tensor_kwargs),
                t_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
            )
            x0_fake = self._velocity_to_x0(x_t_perturbed, v_fake, t)

        return x0_teacher, x0_fake, gan_loss_gen

    # =========================================================================
    # Optimizer / scheduler management (DistillationTrainer interface)
    # =========================================================================

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Create optimizer_dict and scheduler_dict for all trainable networks."""

        # Student optimizer (follows experiment config LR)
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        class _SchedulerProxy:
            def __init__(self, model):
                self._model = model

            @property
            def sample_counter(self):
                return getattr(self._model, "sample_counter", 0)

        net_scheduler = get_base_scheduler(net_optimizer, _SchedulerProxy(self), scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        # Fake_score optimizer (constant LR)
        fake_score_params = [p for p in self.fake_score.parameters() if p.requires_grad]
        if not fake_score_params:
            raise RuntimeError("fake_score has no trainable parameters after freezing unused branches.")
        fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=self.config.fake_score_lr,
            weight_decay=self.config.fake_score_weight_decay,
            betas=tuple(self.config.fake_score_betas),
        )
        fake_score_scheduler = torch.optim.lr_scheduler.LambdaLR(fake_score_optimizer, lr_lambda=lambda step: 1.0)
        self.optimizer_dict["fake_score"] = fake_score_optimizer
        self.scheduler_dict["fake_score"] = fake_score_scheduler

        # Discriminator optimizer (only when GAN is enabled)
        if self.config.gan_loss_weight_gen > 0 and self.discriminator is not None:
            disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.discriminator_lr,
                weight_decay=self.config.discriminator_weight_decay,
            )
            disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lambda step: 1.0)
            self.optimizer_dict["discriminator"] = disc_optimizer
            self.scheduler_dict["discriminator"] = disc_scheduler

        return net_optimizer, net_scheduler

    def get_optimizers(self, iteration: int) -> list:
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            optimizers = [self.optimizer_dict["fake_score"]]
            if "discriminator" in self.optimizer_dict:
                optimizers.append(self.optimizer_dict["discriminator"])
            return optimizers

    def get_lr_schedulers(self, iteration: int) -> list:
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            schedulers = [self.scheduler_dict["fake_score"]]
            if "discriminator" in self.scheduler_dict:
                schedulers.append(self.scheduler_dict["discriminator"])
            return schedulers

    def get_current_optimizer_scheduler(self, iteration: int):
        if self.is_student_phase(iteration):
            return self.optimizer_dict["net"], self.scheduler_dict["net"]
        else:
            return self.optimizer_dict["fake_score"], self.scheduler_dict["fake_score"]

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    def optimizers_schedulers_step(self, grad_scaler, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
        grad_scaler.update()
        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def is_student_phase(self, iteration: int) -> bool:
        return iteration % self.config.student_update_freq == 0

    def get_effective_iteration(self, iteration: int) -> int:
        return iteration // self.config.student_update_freq

    def _set_discriminator_trainable(self, trainable: bool) -> None:
        """Switch discriminator parameter grads for the current DMD2 phase.

        Student/G update still needs discriminator forward to be differentiable
        with respect to teacher features so GAN loss can flow back into the
        student. However, discriminator parameters must be frozen in that phase;
        otherwise their stale grads are not cleared by the student optimizer and
        can leak into the next discriminator update.
        """
        if self.discriminator is None:
            return
        self.discriminator.train(trainable)
        self.discriminator.requires_grad_(trainable)
        if not trainable:
            for param in self.discriminator.parameters():
                param.grad = None

    # =========================================================================
    # Training
    # =========================================================================

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """Main DMD training step. Routes to student or fake_score update."""
        self._current_iteration = iteration

        # CP setup: enable on student / teacher / fake_score (the discriminator
        # consumes already-gathered features so it stays CP-unaware) and align
        # the data batch across CP ranks. Random tensors that must be identical
        # across CP peers are explicitly broadcast at the sampling sites below;
        # the seed alignment remains a belt-and-suspenders guard for stochastic
        # helper code that is harder to thread through manually.
        self._maybe_enable_cp_on_nets([self.net, self.teacher, self.fake_score])
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            data_batch[self.config.input_data_key] = self._broadcast_tensor_for_cp(
                data_batch[self.config.input_data_key]
            )
            data_batch[self.config.input_caption_key] = self._broadcast_object_for_cp(
                data_batch.get(self.config.input_caption_key)
            )
            data_batch.pop("LQ_video_or_image", None)
            data_batch.pop("LQ_latent", None)
            data_batch.pop("degrade_sigma", None)
            # Align CUDA RNG across the CP group for the duration of this step.
            seed_tensor = torch.tensor([iteration * 1009 + 17], dtype=torch.long, device="cuda")
            seed_tensor = self._broadcast_tensor_for_cp(seed_tensor)
            _seed = int(seed_tensor.item())
            torch.manual_seed(_seed)
            torch.cuda.manual_seed(_seed)

        # 1. Normalize HQ image
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]  # [B, C, 1, H, W] -> [B, C, H, W]

        # 2. Prepare LQ conditions (degradation + VAE encode + degrade_sigma)
        data_batch = self.prepare_data_batch_for_training(data_batch, training_iteration=iteration)
        if cp_group is not None and cp_group.size() > 1:
            for _key in ("LQ_video_or_image", "LQ_latent", "degrade_sigma"):
                if isinstance(data_batch.get(_key), torch.Tensor):
                    data_batch[_key] = self._broadcast_tensor_for_cp(data_batch[_key])

        # 3. Build conditioning
        condition = self.conditioner(data_batch)
        captions = condition.caption
        caption_embs, _ = self._encode_text_raw(captions)
        caption_embs = caption_embs.to(**self.tensor_kwargs)

        lq_latent = condition.lq_latent
        if lq_latent is None:
            raise ValueError("PiD conditioner did not produce lq_latent")
        lq_latent = lq_latent.to(**self.tensor_kwargs)

        degrade_sigma = data_batch.get("degrade_sigma")

        # 4. Route to update step. `self.fake_score` may be None in subclasses
        # that drop the score network (e.g. PixelDiTSRFDLossDistillModel) — guard
        # all train/eval/requires_grad calls on it.
        if self.is_student_phase(iteration):
            if self.fake_score is not None:
                self.fake_score.eval()
                self.fake_score.requires_grad_(False)
            self._set_discriminator_trainable(False)
            self.net.train()
            self.net.requires_grad_(True)
            if getattr(self.net, "patch_blocks", None):
                last_patch_block = self.net.patch_blocks[-1]
                if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                    last_patch_block.freeze_unused_text_output_branch()
            self._current_update_type = "student"
            return self._student_update_step(x0, caption_embs, lq_latent, degrade_sigma)
        else:
            self.net.eval()
            self.net.requires_grad_(False)
            if self.fake_score is not None:
                self.fake_score.train()
                self.fake_score.requires_grad_(True)
                if getattr(self.fake_score, "patch_blocks", None):
                    last_patch_block = self.fake_score.patch_blocks[-1]
                    if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                        last_patch_block.freeze_unused_text_output_branch()
            self._set_discriminator_trainable(True)
            self._current_update_type = "fake_score"
            return self._fake_score_update_step(x0, caption_embs, lq_latent, degrade_sigma)

    def _generate_x0_student(
        self,
        x0: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
        with_grad: bool,
    ) -> torch.Tensor:
        """Generate the student prediction used by both VSD and DSM updates.

        `with_grad=True` is the student/VSD path: the selected student forward
        must keep gradients for `self.net`. `with_grad=False` is the fake_score
        DSM path: x0_student is only a target, so every student operation is
        under `torch.no_grad()`.
        """
        B = x0.shape[0]
        timescale = self.fm_trainer.timescale
        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        mode = self.config.student_input_mode
        if mode not in ("teacher_forcing", "pidself_rollout"):
            raise ValueError(f"Invalid student_input_mode: {self.config.student_input_mode!r}")

        with grad_ctx:
            noise = torch.randn_like(x0)
            noise = self._broadcast_tensor_for_cp(noise)
            if self.config.student_sample_steps == 1:
                # 1-step: pure noise -> x0 in one forward pass. Same input
                # distribution for student/VSD and fake_score/DSM; only grad differs.
                t_student = torch.full((B,), self.config.student_timestep, device=x0.device, dtype=torch.float32)
                input_student = noise
                t_student_scaled = t_student * timescale
                v_student = self.net(
                    input_student,
                    t_student_scaled,
                    caption_embs,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
                x0_student = self._velocity_to_x0(input_student, v_student, t_student)
                if not with_grad:
                    x0_student = self._broadcast_tensor_for_cp(x0_student)
                return x0_student

            if mode == "pidself_rollout":
                x0_student = self._pidself_rollout_x0_student(
                    noise,
                    caption_embs,
                    lq_latent,
                    degrade_sigma,
                    with_grad=with_grad,
                )
                if not with_grad:
                    x0_student = self._broadcast_tensor_for_cp(x0_student)
                return x0_student

            if mode == "teacher_forcing":
                # Real data + noise at random intermediate t. Detach x0 in both
                # phases: the target image is only used to construct the state.
                t_list = self._get_t_list(x0.device)
                ids = torch.randint(0, len(t_list) - 1, (B,), device=x0.device)
                ids = self._broadcast_tensor_for_cp(ids)
                t_student = t_list[ids]
                s = [B] + [1] * (x0.ndim - 1)
                t_bcast = t_student.view(*s)
                input_student = (1.0 - t_bcast) * x0.detach() + t_bcast * noise
                t_student_scaled = t_student * timescale
                v_student = self.net(
                    input_student,
                    t_student_scaled,
                    caption_embs,
                    lq_video_or_image=None,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
                x0_student = self._velocity_to_x0(input_student, v_student, t_student)
                if not with_grad:
                    x0_student = self._broadcast_tensor_for_cp(x0_student)
                return x0_student

    def _student_update_step(
        self,
        x0: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ) -> tuple[dict, torch.Tensor]:
        """VSD loss update for student.

        Flow:
        1. Student: noise → x0_student via 1-step prediction
        2. Perturb x0_student at random t
        3. Teacher + fake_score predict x0 from perturbed (no_grad)
        4. VSD loss: distill student toward teacher, guided by fake_score

        Net output convention is handled by `_net_output_to_x0` based on
        `self.config.prediction_type` ("velocity": x0 = x_t - t * net_output;
        "x0": net_output IS x0).
        """
        B = x0.shape[0]
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        timescale = self.fm_trainer.timescale

        with autocast_ctx:
            # Step 1: Student generation — obtain x0_student
            x0_student = self._generate_x0_student(
                x0,
                caption_embs,
                lq_latent,
                degrade_sigma,
                with_grad=True,
            )

            # Step 2: Sample random perturbation t and perturb
            t = self.fm_trainer.sample_t(B, device=x0.device)
            if self.config.dmd_timestep_clamp_min > 0 or self.config.dmd_timestep_clamp_max < 1.0:
                t = t.clamp(self.config.dmd_timestep_clamp_min, self.config.dmd_timestep_clamp_max)
            t = self._broadcast_tensor_for_cp(t)
            t_scaled = t * timescale

            eps = torch.randn_like(x0_student)
            eps = self._broadcast_tensor_for_cp(eps)
            s = [B] + [1] * (x0_student.ndim - 1)
            t_bcast = t.view(*s)

            gan_enabled = self.config.gan_loss_weight_gen > 0 and self.discriminator is not None

            if gan_enabled:
                # GAN path: do NOT detach x0_student so gradient flows through
                # x_t_perturbed -> feature extractor -> discriminator -> student.
                x_t_perturbed = (1.0 - t_bcast) * x0_student + t_bcast * eps
                x0_teacher, x0_fake, gan_loss_gen = self._compute_student_gan_targets_and_loss(
                    x_t_perturbed=x_t_perturbed,
                    t=t,
                    t_scaled=t_scaled,
                    caption_embs=caption_embs,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
            else:
                # Original path: detach x0_student (no GAN gradient)
                x_t_perturbed = (1.0 - t_bcast) * x0_student.detach() + t_bcast * eps

                # Step 3: Teacher and fake_score x0 predictions (no grad)
                x0_teacher = self._teacher_cfg_x0(
                    x_t_perturbed.to(**self.tensor_kwargs),
                    t,
                    t_scaled,
                    caption_embs,
                    lq_latent,
                    degrade_sigma,
                )
                with torch.no_grad():
                    v_fake = self.fake_score(
                        x_t_perturbed.to(**self.tensor_kwargs),
                        t_scaled,
                        caption_embs,
                        lq_video_or_image=None,
                        lq_latent=lq_latent,
                        degrade_sigma=degrade_sigma,
                    )
                    x0_fake = self._velocity_to_x0(x_t_perturbed, v_fake, t)

            # Step 4: VSD loss
            vsd_loss = variational_score_distillation_loss(x0_student, x0_teacher, x0_fake)

        output = {"vsd_loss": vsd_loss, "update_type": "student"}
        total_loss = vsd_loss * self.config.vsd_loss_weight

        if gan_enabled:
            output["gan_loss_gen"] = gan_loss_gen
            gan_w = self._get_gan_weight(self._current_iteration)
            total_loss = total_loss + gan_loss_gen * gan_w

        # Every loss term in the student step (VSD, GAN gen) flows
        # gradient back through `self.net`'s gather-at-end output, so each rank
        # only contributes its local L slice. Scale by cp_size to compensate
        # for FSDP's averaging across CP ranks.
        total_loss = total_loss * self._cp_loss_scale
        return output, total_loss

    def _fake_score_update_step(
        self,
        x0: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ) -> tuple[dict, torch.Tensor]:
        """DSM loss update for fake_score.

        Flow:
        1. Generate x0_student (no grad)
        2. Perturb at random t
        3. fake_score forward (with grad) — output is velocity or x0 per prediction_type
        4. DSM loss: velocity mode MSE(v_fake, eps - x0_student); x0 mode MSE(x0_fake, x0_student)
        """
        B = x0.shape[0]
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        timescale = self.fm_trainer.timescale

        with autocast_ctx:
            # Step 1: Student generation (no grad — x0_student is used only as DSM target)
            x0_student = self._generate_x0_student(
                x0,
                caption_embs,
                lq_latent,
                degrade_sigma,
                with_grad=False,
            )

            # Step 2: Sample perturbation t and perturb
            t = self.fm_trainer.sample_t(B, device=x0.device)
            if self.config.dmd_timestep_clamp_min > 0 or self.config.dmd_timestep_clamp_max < 1.0:
                t = t.clamp(self.config.dmd_timestep_clamp_min, self.config.dmd_timestep_clamp_max)
            t = self._broadcast_tensor_for_cp(t)
            t_scaled = t * timescale

            eps = torch.randn_like(x0_student)
            eps = self._broadcast_tensor_for_cp(eps)
            s = [B] + [1] * (x0_student.ndim - 1)
            t_bcast = t.view(*s)
            x_t_perturbed = (1.0 - t_bcast) * x0_student + t_bcast * eps

            # Step 3: fake_score forward (with grad)
            v_fake = self.fake_score(
                x_t_perturbed.to(**self.tensor_kwargs),
                t_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
            )

            # Step 4: DSM loss — match fake_score's output to the student distribution in
            # whichever space the fake_score net is parameterized in.
            #   velocity mode: v_fake = noise - x0; target = eps - x0_student.
            #   x0 mode:       v_fake = x0 directly; target = x0_student (plain x0-space MSE,
            #                  avoids 1/t division entirely).
            if self.config.prediction_type == "x0":
                dsm_loss = denoising_score_matching_loss_flow(v_fake, x0_student)
            elif self.config.prediction_type == "velocity":
                target_v = eps - x0_student
                dsm_loss = denoising_score_matching_loss_flow(v_fake, target_v)
            else:
                raise ValueError(f"Invalid prediction_type: {self.config.prediction_type}")

        output = {"dsm_loss": dsm_loss, "update_type": "fake_score"}
        # DSM gradient flows through `self.fake_score`'s gather-at-end output
        # (CP-split per rank) → scale by cp_size. The discriminator and R1
        # terms below operate on already-gathered teacher features, so their
        # gradients are CP-replicated and do NOT receive the same scaling.
        total_loss = dsm_loss * self.config.dsm_loss_weight * self._cp_loss_scale

        # Step 5: GAN discriminator update (shared with FD-loss subclass).
        disc_out, disc_loss = self._discriminator_update_step(
            x_t_perturbed=x_t_perturbed,
            t=t,
            t_scaled=t_scaled,
            eps=eps,
            x0=x0,
            caption_embs=caption_embs,
            lq_latent=lq_latent,
            degrade_sigma=degrade_sigma,
        )
        output.update(disc_out)
        total_loss = total_loss + disc_loss

        return output, total_loss

    def _discriminator_update_step(
        self,
        x_t_perturbed: torch.Tensor,
        t: torch.Tensor,
        t_scaled: torch.Tensor,
        eps: torch.Tensor,
        x0: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma: Optional[torch.Tensor],
    ) -> tuple[dict, torch.Tensor]:
        """Compute the GAN discriminator update on already-perturbed inputs.

        Returns `(output_dict, additional_loss)` so callers can fold into their
        own running totals. When GAN is disabled this is a no-op returning
        `({}, zero_tensor)`. Extracted so the FD-loss subclass can reuse the
        same disc-training path without bringing in the DSM/fake_score pieces.
        """
        if self.config.gan_loss_weight_gen <= 0 or self.discriminator is None:
            return {}, x_t_perturbed.new_zeros(())

        with torch.no_grad():
            # Fake features: teacher processes student's noised output (early exit).
            fake_feat = self.teacher(
                x_t_perturbed.to(**self.tensor_kwargs),
                t_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
                feature_indices=self.discriminator.feature_indices,
                return_features_early=True,
            )
            # Real features: teacher processes noised GT with (optionally different) t/noise.
            real_feat, t_real, perturbed_real = self._compute_real_feat(
                x0, t, eps, caption_embs, lq_latent, degrade_sigma
            )

        # Discriminator forward with gradient (training discriminator).
        real_logits = self.discriminator(real_feat)
        fake_logits = self.discriminator(fake_feat)
        gan_disc_loss = gan_loss_discriminator(real_logits, fake_logits)
        output = {"gan_loss_disc": gan_disc_loss}
        total_loss = gan_disc_loss

        if self.config.gan_r1_reg_weight > 0:
            r1_loss = self._compute_r1_regularization(
                real_logits, perturbed_real, t_real, caption_embs, lq_latent, degrade_sigma
            )
            output["gan_loss_r1"] = r1_loss
            total_loss = total_loss + self.config.gan_r1_reg_weight * r1_loss

        return output, total_loss

    # =========================================================================
    # Inference (1-step)
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
        is_negative_prompt: bool = False,
        **kwargs,
    ):
        """1-step SR generation using distilled student.

        Generates SR images by running the student network at student_timestep=1.0
        (pure noise → x0 in a single forward pass).

        Returns:
            SR images [B, 3, 1, H, W] in [-1, 1] for callback compatibility.
        """
        self._validate_inference_data_batch(data_batch)

        # Enable CP on the student network for inference and align inputs across CP ranks.
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

        # Encode captions using the latent as the batch-size source of truth.
        captions = data_batch[self.config.input_caption_key]
        if isinstance(captions, str):
            captions = [captions] * lq_latent.shape[0]
        elif isinstance(captions, tuple):
            captions = list(captions)
        B = lq_latent.shape[0]
        if isinstance(captions, list) and len(captions) == 1 and B > 1:
            captions = captions * B
        if not isinstance(captions, list) or len(captions) != B:
            raise ValueError(f"Expected {B} captions for LQ_latent batch, got {captions!r}")
        caption_embs, _ = self._encode_text_raw(captions)
        caption_embs = caption_embs.to(**self.tensor_kwargs)

        # Degradation sigma conditioning. Source of truth is data_batch["degrade_sigma"];
        # accepts float / list / [B] tensor. Scalar values broadcast to the full batch.
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

        # Student generation (1-step or multi-step). Seed the global CUDA RNG so
        # both the initial noise and every intermediate SDE re-noise step
        # (`torch.randn_like` inside `_student_sample_loop`) consume from the
        # same seeded stream.
        torch.cuda.manual_seed(int(seed))
        noise = torch.randn(B, 3, img_h, img_w, device="cuda")

        self.net.eval()

        # num_steps overrides config at inference time (subsamples student_t_list)
        effective_steps = num_steps if num_steps is not None else self.config.student_sample_steps

        if effective_steps == 1:
            x0_student = self._student_sample_1step(
                noise,
                caption_embs,
                lq_latent,
                degrade_sigma_tensor,
            )
        else:
            x0_student = self._student_sample_loop(
                noise,
                effective_steps,
                caption_embs,
                lq_latent,
                degrade_sigma_tensor,
            )

        return x0_student.clamp(-1, 1).unsqueeze(2)  # [B, 3, 1, H, W]

    @torch.no_grad()
    def _dpms_sample_loop(
        self,
        net,
        prediction_type: str,
        data_batch: dict,
        num_steps: int,
        seed: int,
        guidance: Optional[float],
        image_size=None,
        shift: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the same SR DPM-Solver path used by PixelDiTSRModel inference.

        This is intentionally separate from `_multistep_sample_loop`: the latter is
        a simple diagnostic Euler/SDE probe, while production PixelDiT-SR inference
        uses DPMS with `time_uniform_flow` and flow shift. The difference is small
        at many steps but very visible at 4 steps.
        """
        from pid._src.modules.dpmsolver import DPMS

        data_batch = dict(data_batch)
        self._validate_inference_data_batch(data_batch)

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

        if shift is None:
            if self.config.dynamic_shift is not None:
                _ds = self.config.dynamic_shift
                shift = _ds["base_shift"] * math.sqrt(math.sqrt(img_h * img_w) / _ds["base_image_size_for_shift_calc"])
            else:
                shift = self.config.shift

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
        null_y = self._null_caption_embs.unsqueeze(1).repeat(B, 1, 1, 1)

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

        model_dtype = next(net.parameters()).dtype
        cfg_scale = float(guidance) if guidance is not None else 1.0
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        z = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)
        was_training = net.training
        net.eval()
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()

        try:
            with autocast_ctx:

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

                def _cfg_model_fn(x, timestep, y, mask=None, **kw):
                    half_B = x.shape[0] // 2
                    if y.dim() == 4:
                        y = y.squeeze(1)
                    out_uncond = net(
                        x[:half_B].to(model_dtype),
                        timestep[:half_B].to(model_dtype),
                        y[:half_B].to(model_dtype),
                        lq_video_or_image=None,
                        lq_latent=lq_latent,
                        degrade_sigma=degrade_sigma_tensor,
                    )
                    out_cond = net(
                        x[half_B:].to(model_dtype),
                        timestep[half_B:].to(model_dtype),
                        y[half_B:].to(model_dtype),
                        lq_video_or_image=None,
                        lq_latent=lq_latent,
                        degrade_sigma=degrade_sigma_tensor,
                    )
                    return torch.cat([out_uncond, out_cond], dim=0)

                if prediction_type == "x0":
                    dpms_model_type = "x_start"
                elif prediction_type == "velocity":
                    dpms_model_type = "flow"
                else:
                    raise ValueError(f"Invalid prediction_type: {prediction_type}")

                dpm_solver = DPMS(
                    _forward_fn if cfg_scale == 1.0 else _cfg_model_fn,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_type=dpms_model_type,
                    guidance_type="classifier-free",
                    model_kwargs=dict(mask=emb_masks),
                    schedule="FLOW",
                    interval_guidance=[0, 1],
                )
                return dpm_solver.sample(
                    z,
                    steps=int(num_steps),
                    order=min(int(num_steps), 2),
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=float(shift),
                )
        finally:
            if was_training:
                net.train()

    @torch.no_grad()
    def _multistep_sample_loop(
        self,
        net,
        num_steps: int,
        prediction_type: str,
        sample_type: str,
        cfg_scale: float,
        caption_embs: torch.Tensor,
        lq_latent: Optional[torch.Tensor],
        degrade_sigma_tensor: Optional[torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Generalized multi-step denoising loop usable with any net.

        Mirrors `_student_sample_loop` (line 578) with two extensions:
        - `prediction_type` is explicit (teacher may differ from student/fake_score).
        - When `cfg_scale != 1.0`, each step runs cond + null forwards and combines
          them with `v = v_uncond + cfg_scale * (v_cond - v_uncond)` (matches
          `_teacher_cfg_x0`, line 825).

        Builds a fresh `linspace(student_timestep, 0, num_steps + 1)` schedule.
        Diagnostic samplers are not bound to the training-time schedule, so a
        clean uniform linspace is the right schedule for a multi-step probe.
        """
        t_list = torch.linspace(
            float(self.config.student_timestep),
            0.0,
            num_steps + 1,
            device=noise.device,
            dtype=torch.float32,
        )
        B = noise.shape[0]
        timescale = self.fm_trainer.timescale
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        x = noise

        null_embs = None
        if cfg_scale != 1.0:
            null_embs = self._null_caption_embs.expand(B, -1, -1).to(
                device=caption_embs.device, dtype=caption_embs.dtype
            )

        def _net_forward(x_t, t_scaled):
            v_cond = net(
                x_t,
                t_scaled,
                caption_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            if null_embs is None:
                return v_cond
            v_uncond = net(
                x_t,
                t_scaled,
                null_embs,
                lq_video_or_image=None,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        with autocast_ctx:
            for i in range(t_list.shape[0] - 2):
                t_cur = t_list[i]
                t_next = t_list[i + 1]
                t_cur_batch = t_cur.expand(B)
                t_cur_scaled = t_cur_batch * timescale

                v_pred = _net_forward(x, t_cur_scaled)

                if sample_type == "ode":
                    v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, prediction_type)
                    dt = t_next - t_cur
                    x = x + dt * v_for_step
                else:  # "sde"
                    x0_pred = self._net_output_to_x0(x, v_pred, t_cur_batch, prediction_type)
                    eps_infer = torch.randn_like(x0_pred)
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer

            # Final step terminates at t_next == 0; both ode/sde collapse to v -> x0.
            t_cur = t_list[-2]
            t_cur_batch = t_cur.expand(B)
            t_cur_scaled = t_cur_batch * timescale
            v_pred = _net_forward(x, t_cur_scaled)
            x = self._net_output_to_x0(x, v_pred, t_cur_batch, prediction_type)

        return x

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def model_dict(self) -> dict:
        d = {"net": self.net}
        if self.fake_score is not None:
            d["fake_score"] = self.fake_score
        if self.discriminator is not None:
            d["discriminator"] = self.discriminator
        return d

    def state_dict(self, *args, **kwargs):
        """Save student + fake_score. Teacher is not saved (loaded from pretrained)."""
        sd = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            sd.update(self.net_ema.state_dict(prefix="net_ema."))
        if self.fake_score is not None:
            sd.update(self.fake_score.state_dict(prefix="fake_score."))
        if self.discriminator is not None:
            sd.update(self.discriminator.state_dict(prefix="discriminator."))
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        """Load checkpoint with prefix routing."""
        pretrain_copy = kwargs.get("pretrain_copy", False)
        _net_sd = OrderedDict()
        _ema_sd = OrderedDict()
        _fs_sd = OrderedDict()
        _disc_sd = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("net.") and not k.startswith("net_ema."):
                _net_sd[k[len("net.") :]] = v
            elif k.startswith("net_ema."):
                _ema_sd[k[len("net_ema.") :]] = v
            elif k.startswith("fake_score."):
                _fs_sd[k[len("fake_score.") :]] = v
            elif k.startswith("discriminator."):
                _disc_sd[k[len("discriminator.") :]] = v
            else:
                # Could be a bare state dict from the SR model checkpoint (no prefix)
                _net_sd[k] = v

        if _net_sd:
            missing, unexpected = self.net.load_state_dict(_net_sd, strict=False, assign=assign)
            if missing:
                lq_missing = [k for k in missing if "lq_proj" in k]
                other_missing = [k for k in missing if "lq_proj" not in k]
                if lq_missing:
                    logger.info(f"Expected missing LQ keys ({len(lq_missing)} keys)")
                if other_missing and strict:
                    logger.warning(f"Missing keys in net: {other_missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in net: {unexpected}")

        if _ema_sd and self.config.ema.enabled and hasattr(self, "net_ema"):
            self.net_ema.load_state_dict(_ema_sd, strict=False, assign=assign)
        elif pretrain_copy and _net_sd and self.config.ema.enabled and hasattr(self, "net_ema"):
            # EMA-only consolidated exports are intentionally renamed from
            # net_ema.* to net.*.  During pretrained initialization, seed both
            # student copies from those weights instead of leaving net_ema at
            # its random construction-time value.
            self.net_ema.load_state_dict(_net_sd, strict=False, assign=assign)

        if _fs_sd and self.fake_score is not None:
            self.fake_score.load_state_dict(_fs_sd, strict=False, assign=assign)

        if _disc_sd and self.discriminator is not None:
            self.discriminator.load_state_dict(_disc_sd, strict=False, assign=assign)

    # =========================================================================
    # Training hooks
    # =========================================================================

    def on_train_start(self, memory_format=torch.preserve_format) -> None:
        """Ensure teacher is frozen, fake_score is trainable."""
        super().on_train_start(memory_format)

        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            self.teacher = self.teacher.to(memory_format=memory_format)

        if self.fake_score is not None:
            self.fake_score.train()
            self.fake_score = self.fake_score.to(memory_format=memory_format)

    def on_before_zero_grad(self, optimizer, scheduler, iteration: int) -> None:
        """EMA update and joint-mode fake_score optimizer stepping."""
        from pid._src.utils.misc import update_master_weights

        del optimizer, scheduler

        if not self.is_student_phase(iteration):
            # fake_score phase: update master weights
            if self.fake_score is not None:
                update_master_weights(self.optimizer_dict["fake_score"])
            return
        # Student phase: EMA update
        if self.config.ema.enabled and hasattr(self, "net_ema"):
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_after_backward(self):
        """Pass-through for trainer compatibility."""
        pass

    def clip_grad_norm_(self, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        """Clip gradients for student + fake_score."""
        from pid._src.utils.torch_future import clip_grad_norm_ as clip_grad_norm_impl_

        if self.fake_score is not None:
            for p in self.fake_score.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=0, neginf=0, out=p.grad)
            clip_grad_norm_impl_(
                self.fake_score.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        return clip_grad_norm_impl_(
            self.net.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
