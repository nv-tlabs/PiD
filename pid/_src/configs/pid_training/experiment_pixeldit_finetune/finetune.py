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
from pid._src.callbacks.every_n_draw_sample import EveryNDrawSample, EveryNDrawSampleMultiResolution

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

RES2K_TO_4K_CKPT = "checkpoints/PixelDiT_finetune_2kto4k/model_ema_bf16.pth"


def _build_debug_run(job):
    _TRAINER_DEBUG_CONFIG = dict(
        max_iter=25,
        logging_iter=2,
        callbacks=dict(
            every_n_sample=dict(
                every_n=10,
            ),
            every_n_sample_ema=dict(
                every_n=10,
            ),
            every_n_sample_ema_normal=dict(
                every_n=5,
            ),
            every_n_sample_ema_small_face=dict(
                every_n=5,
            ),
        ),
    )
    w_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_debug" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            wandb_mode="disabled",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        upload_reproducible_setup=False,
    )
    return [w_resume]


# =============================================================================
# Multi-resolution finetune via dataloader-side resolution sampling
# =============================================================================

"""
# debug run
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048_debug"

# full run
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048"
"""
PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
            {"override /model": "ddp_pixeldit"},
            {"override /net": "pixeldit_h1536_d14p2"},
            {"override /conditioner": "pixeldit_caption"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb"]},
            {"override /checkpoint": "local"},
            {"override /tokenizer": None},  # no VAE tokenizer needed
            "_self_",
        ],
        job=dict(
            group="pixeldit_finetune",
            name="pixeldit_text_to_image_finetune_res_2048",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.0,
        ),
        # Original uses lr_schedule=constant with 2000 warmup steps.
        # f_max=f_min=1.0 gives constant LR after warmup (no decay).
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1e-6],
            warm_up_steps=[2000],
            cycle_lengths=[10_000_000],  # effectively infinite (no cycle)
        ),
        model=dict(
            config=dict(
                precision="bfloat16",
                # CHI prompt: enabled during training (matches original)
                chi_prompt=_CHI_PROMPT,
                # Inference defaults (per-step training shift comes from dynamic_shift)
                shift=6.0,
                cfg_scale=2.75,
                image_size=2048,
                negative_prompt="low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, watermark",
                num_sample_steps=50,
                # REPA disabled for high resolution (matches repa_loss_weight=0.0 in original)
                repa_config=None,
                loss_weights={
                    "diffusion": 1.0,
                    "repa": 0.5,
                },
                ema=dict(
                    enabled=True,
                    rate=0.1,
                    iteration_shift=0,
                ),
                net=dict(
                    rope_mode="original",
                ),
            ),
        ),
        checkpoint=dict(
            save_iter=5000,
            load_path=RES2K_TO_4K_CKPT,
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=10_000_000,
            logging_iter=50,
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                every_n_sample=L(EveryNDrawSample)(
                    every_n=5000,
                    is_ema=False,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    resize_wandb_image=False,
                ),
                every_n_sample_ema=L(EveryNDrawSample)(
                    every_n=5000,
                    is_ema=True,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    resize_wandb_image=False,
                ),
                every_n_sample_ema_small_face=L(EveryNDrawSample)(
                    every_n=5000,
                    is_ema=True,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    resize_wandb_image=False,
                    fix_batch_fp="pid/_src/dataprep/prompts/prompts_example.txt",
                    n_sample_to_save=128,
                    name="example",
                ),
                every_n_sample_ema_small_text=L(EveryNDrawSample)(
                    every_n=5000,
                    is_ema=True,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    resize_wandb_image=False,
                    fix_batch_fp="pid/_src/dataprep/prompts/prompts_harder_cases.txt",
                    n_sample_to_save=128,
                    name="harder_cases",
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
    ),
)


"""
# debug run
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048_to_3840_debug"

Note: we use rope_ref_h & rope_ref_w = 2048 for PiD v1.5 checkpoint
For early checkpoints, we use rope_ref_h & rope_ref_w = 1024
"""
PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048_TO_3840: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/pixeldit_text_to_image_finetune_res_2048",
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840"},
            "_self_",
        ],
        job=dict(
            group="pixeldit_finetune",
            name="pixeldit_text_to_image_finetune_res_2048_to_3840",
        ),
        model=dict(
            config=dict(
                image_size=4096,
                dynamic_shift=dict(
                    base_shift=6.0,
                    base_image_size_for_shift_calc=2048,
                ),
                net=dict(
                    rope_mode="ntk_aware",
                    rope_ref_h=2048,
                    rope_ref_w=2048,
                ),
            ),
        ),
        checkpoint=dict(
            load_path=RES2K_TO_4K_CKPT,
            replicate_ema_to_reg_in_training=True,
        ),
        trainer=dict(
            grad_accum_iter=2,
            callbacks=dict(
                every_n_sample_ema_small_face=L(EveryNDrawSampleMultiResolution)(
                    every_n=5000,
                    is_ema=True,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    image_sizes=[2048, 3072, 4096],
                    prompt_only_sample=True,
                    resize_wandb_image=False,
                    fix_batch_fp="pid/_src/dataprep/prompts/prompts_example.txt",
                    n_sample_to_save=128,
                    name="example",
                ),
                every_n_sample_ema_small_text=L(EveryNDrawSampleMultiResolution)(
                    every_n=5000,
                    is_ema=True,
                    guidance=[2.75, 5.0],
                    num_sampling_step=50,
                    image_sizes=[2048, 3072, 4096],
                    prompt_only_sample=True,
                    resize_wandb_image=False,
                    fix_batch_fp="pid/_src/dataprep/prompts/prompts_harder_cases.txt",
                    n_sample_to_save=128,
                    name="harder_cases",
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
    ),
)

cs = ConfigStore.instance()

for _item, _item_debug in [
    [
        PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048,
        *_build_debug_run(PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048),
    ],
    [
        PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048_TO_3840,
        *_build_debug_run(PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048_TO_3840),
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
