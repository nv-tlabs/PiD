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

from typing import List

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.callbacks.every_n_draw_sample import EveryNDrawSample
from pid._src.callbacks.every_n_evaluate import EveryNEvaluate

FIXED_BATCH_N_SAMPLE_TO_SAVE = 32


def get_every_n_callbacks_train(
    guidance_draw_sample: List[float],
    num_sampling_step: int,
    every_n_sample: int = 5000,
) -> dict:
    """Callbacks on the training-set batch."""
    _s = num_sampling_step
    return {
        f"every_n_sample_train_infer_{_s}step_reg": L(EveryNDrawSample)(
            every_n=every_n_sample,
            is_ema=False,
            guidance=guidance_draw_sample,
            num_sampling_step=num_sampling_step,
            resize_wandb_image=False,
            name=f"train_infer_{_s}step",
        ),
        f"every_n_sample_train_infer_{_s}step_ema": L(EveryNDrawSample)(
            every_n=every_n_sample,
            is_ema=True,
            guidance=guidance_draw_sample,
            num_sampling_step=num_sampling_step,
            resize_wandb_image=False,
            name=f"train_infer_{_s}step",
        ),
    }


def get_every_n_callbacks_fullstep(
    fix_batch_fp: str,
    fix_batch_dir: str,
    guidance_draw_sample: List[float],
    guidance_evaluate: float,
    num_sampling_step: int,
    every_n_sample: int = 5000,
    every_n_evaluate: int = 5000,
    name: str = "",
    is_ema: bool = True,
    n_sample_to_save: int = FIXED_BATCH_N_SAMPLE_TO_SAVE,
) -> dict:
    """Callbacks on fully-denoised (clean) latents from the diffusion backbone.

    fix_batch_fp  : path template, e.g.
        "assets/pixel_diffusion_flux_xt/full_step/2048/fix_batch_{:04d}.pt"
    fix_batch_dir : directory for EveryNEvaluate, e.g.
        "assets/pixel_diffusion_flux_xt/full_step/2048"

    No degrade_sigma — these latents have σ=0 (fully denoised).
    n_sample_to_save must match the number of prepared fix_batch_*.pt files.
    """
    _s = num_sampling_step
    name_suffix = f"_{name}" if name else ""
    ema_or_reg_name = "ema" if is_ema else "reg"
    return {
        f"every_n_sample_generated_fullstep{name_suffix}_infer_{_s}step_{ema_or_reg_name}": L(EveryNDrawSample)(
            every_n=every_n_sample,
            is_ema=is_ema,
            fix_batch_fp=fix_batch_fp,
            guidance=guidance_draw_sample,
            num_sampling_step=num_sampling_step,
            n_sample_to_save=n_sample_to_save,
            resize_wandb_image=False,
            name=f"generated_fullstep{name_suffix}_infer_{_s}step",
        ),
        f"every_n_evaluate_generated_fullstep{name_suffix}_infer_{_s}step_{ema_or_reg_name}": L(EveryNEvaluate)(
            every_n=every_n_evaluate,
            is_ema=is_ema,
            fix_batch_dir=fix_batch_dir,
            metrics=[
                "lq_color_de2000",
                # "psnr",
                # "ssim",
                # "lpips",
                # "musiq",
                # "musiq_paq2piq",
                # "musiq_spaq",
                # "clipiqa_plus",
                # "qalign_native",
                # "visualquality_r1",
            ],
            batch_size_in_evaluation=8,
            guidance=guidance_evaluate,
            num_sampling_step=num_sampling_step,
            name=f"generated_fullstep{name_suffix}_infer_{_s}step",
        ),
    }


def get_every_n_callbacks_at_step(
    step_name: str,
    fix_batch_fp: str,
    fix_batch_dir: str,
    guidance_draw_sample: List[float],
    guidance_evaluate: float,
    num_sampling_step: int,
    every_n_sample: int = 5000,
    every_n_evaluate: int = 5000,
    is_ema: bool = True,
    n_sample_to_save: int = FIXED_BATCH_N_SAMPLE_TO_SAVE,
) -> dict:
    """Callbacks on latents captured at a specific intermediate diffusion step.

    Args:
        step_name    : Label used in callback key names, e.g. "16step", "28step".
        fix_batch_fp : Path template for EveryNDrawSample, e.g.
            "assets/pixel_diffusion_flux_xt/16step/2048/fix_batch_{:04d}.pt"
        fix_batch_dir: Directory for EveryNEvaluate, e.g.
            "assets/pixel_diffusion_flux_xt/16step/2048"
        n_sample_to_save: Number of prepared fix_batch_*.pt files.
    """
    _s = num_sampling_step
    ema_or_reg_name = "ema" if is_ema else "reg"
    return {
        f"every_n_sample_generated_{step_name}_infer_{_s}step_{ema_or_reg_name}": L(EveryNDrawSample)(
            every_n=every_n_sample,
            is_ema=is_ema,
            fix_batch_fp=fix_batch_fp,
            guidance=guidance_draw_sample,
            num_sampling_step=num_sampling_step,
            n_sample_to_save=n_sample_to_save,
            resize_wandb_image=False,
            name=f"generated_{step_name}_infer_{_s}step",
        ),
        f"every_n_evaluate_generated_{step_name}_infer_{_s}step_{ema_or_reg_name}": L(EveryNEvaluate)(
            every_n=every_n_evaluate,
            is_ema=is_ema,
            fix_batch_dir=fix_batch_dir,
            metrics=["musiq", "musiq_paq2piq", "musiq_spaq"],  # more metrics can be added here
            guidance=guidance_evaluate,
            num_sampling_step=num_sampling_step,
            name=f"generated_{step_name}_infer_{_s}step",
        ),
    }
