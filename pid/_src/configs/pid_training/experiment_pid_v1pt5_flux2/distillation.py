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

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid_training.shared_utils import (
    get_every_n_callbacks_fullstep,
)
from pid._src.networks.discriminators import Discriminator_ImageDiT

# Default CHI prompt from the original PixelDiT config (used during training)
_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

PID_TEACHER_CKPT = "checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux2_undistilled/model_ema_bf16.pth"


def _build_debug_run(job):
    callback_overrides = dict(
        every_n_sample_generated_fullstep_res2048_infer_4step_ema=dict(every_n=10),
    )

    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_W_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
            callbacks=callback_overrides,
        ),
        upload_reproducible_setup=False,
    )


"""
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12350 \
    -m scripts.train --config=pid/_src/configs/pid_training/config.py \
    -- experiment="pid_v1pt5_student_flux2_h1024_d4_res_2048_distill_debug"
"""
PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
            {"override /model": "ddp_pid_distillation"},
            {"override /net": "pid_sr4x_v1pt5_for_flux2"},
            {"override /conditioner": "pid_caption_lq"},
            {"override /ckpt_type": "dcp_distill"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb_distill"]},
            {"override /checkpoint": "local"},
            {"override /tokenizer": "flux2_vae_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="pid_training_v1pt5",
            name="pid_v1pt5_student_flux2_h1024_d4_res_2048_distill",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1e-6],
            warm_up_steps=[100],
            cycle_lengths=[10_000_000],
        ),
        model=dict(
            config=dict(
                precision="bfloat16",
                input_data_key="image",
                input_caption_key="caption",
                chi_prompt=_CHI_PROMPT,
                shift=6.0,
                cfg_scale=3,
                image_size=2048,
                repa_config=None,
                ema=dict(
                    enabled=True,
                ),
                conditioner=dict(
                    lq_latent=dict(
                        dropout_rate=0,
                    ),
                ),
                state_ch=128,
                train_degradation_config=dict(
                    downscale=4.0,
                ),
                latent_noising=dict(
                    enabled=True,
                    backbone="flow_matching",
                    add_sigma_min=0.0,
                    add_sigma_max=0.8,
                ),
                dynamic_shift=dict(
                    base_shift=6.0,
                    base_image_size_for_shift_calc=2048,
                ),
                lq_latent_image_align_config=dict(
                    enabled=False,
                ),
                net=dict(
                    train_lq_proj_only=False,  # distill trains the full student
                    lq_hidden_dim=1024,
                    lq_num_res_blocks=4,
                ),
                pretrained_teacher_path=PID_TEACHER_CKPT,
                student_update_freq=6,
                vsd_loss_weight=1.0,
                dsm_loss_weight=1.0,
                student_sample_steps=4,
                student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0],
                student_sample_type="sde",
                fake_score_lr=1e-5,
                fake_score_weight_decay=1e-3,
                fake_score_betas=(0.9, 0.999),
                dmd_timestep_clamp_min=0.02,
                dmd_timestep_clamp_max=0.99,
                teacher_cfg_scale=3,
                gan_loss_weight_gen=0.05,
                gan_warmup_steps=100,
                discriminator_lr=1e-5,
                net_discriminator=L(Discriminator_ImageDiT)(
                    feature_indices={7},
                    num_blocks=14,
                    inner_dim=1536,
                ),
                gan_r1_reg_weight=200.0,
                gan_r1_reg_alpha=0.1,
                gan_use_same_t_noise=True,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        checkpoint=dict(
            save_iter=100,
            replicate_ema_to_reg_in_training=False,
            load_training_state=False,
            strict_resume=False,
            load_path=PID_TEACHER_CKPT,
        ),
        trainer=dict(
            max_iter=3_000,
            logging_iter=50,
            ddp=dict(
                static_graph=False,
                find_unused_parameters=True,
            ),
            callbacks=dict(
                grad_clip=dict(clip_norm=0.1),
                **get_every_n_callbacks_fullstep(
                    fix_batch_fp="assets/pid_callback_assets/flux2/full_step/2048/fix_batch_{:04d}.pt",
                    fix_batch_dir="assets/pid_callback_assets/flux2/full_step/2048",
                    guidance_draw_sample=[1],
                    guidance_evaluate=1,
                    num_sampling_step=4,
                    every_n_sample=100,
                    every_n_evaluate=100,
                    name="res2048",
                ),
                **get_every_n_callbacks_fullstep(
                    fix_batch_fp="assets/pid_callback_assets/flux2/46step/2048/fix_batch_{:04d}.pt",
                    fix_batch_dir="assets/pid_callback_assets/flux2/46step/2048",
                    guidance_draw_sample=[1],
                    guidance_evaluate=1,
                    num_sampling_step=4,
                    every_n_sample=100,
                    every_n_evaluate=100,
                    name="46step_res2048",
                ),
            ),
        ),
    ),
)

"""
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12350 \
    -m scripts.train --config=pid/_src/configs/pid_training/config.py \
    -- experiment="pid_v1pt5_student_flux2_h1024_d4_res_2048_to_3840_distill_debug"
"""
PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_TO_3840_DISTILL: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/pid_v1pt5_student_flux2_h1024_d4_res_2048_distill",
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840"},
            "_self_",
        ],
        job=dict(
            group="pid_training_v1pt5",
            name="pid_v1pt5_student_flux2_h1024_d4_res_2048_to_3840_distill",
        ),
        model_parallel=dict(
            context_parallel_size=4,
        ),
        trainer=dict(
            callbacks=dict(
                **get_every_n_callbacks_fullstep(
                    fix_batch_fp="assets/pid_callback_assets/flux2/full_step/4096/fix_batch_{:04d}.pt",
                    fix_batch_dir="assets/pid_callback_assets/flux2/full_step/4096",
                    guidance_draw_sample=[1],
                    guidance_evaluate=1,
                    num_sampling_step=4,
                    every_n_sample=100,
                    every_n_evaluate=100,
                    name="res4096",
                ),
                **get_every_n_callbacks_fullstep(
                    fix_batch_fp="assets/pid_callback_assets/flux2/46step/4096/fix_batch_{:04d}.pt",
                    fix_batch_dir="assets/pid_callback_assets/flux2/46step/4096",
                    guidance_draw_sample=[1],
                    guidance_evaluate=1,
                    num_sampling_step=4,
                    every_n_sample=100,
                    every_n_evaluate=100,
                    name="46step_res4096",
                ),
            ),
        ),
    ),
)

cs = ConfigStore.instance()

for _item, _item_debug in [
    [
        PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_DISTILL,
        _build_debug_run(PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_DISTILL),
    ],
    [
        PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_TO_3840_DISTILL,
        _build_debug_run(PID_V1PT5_STUDENT_FLUX2_H1024_D4_RES_2048_TO_3840_DISTILL),
    ],
]:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
    if _item_debug is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_debug",
            node=_item_debug,
        )
