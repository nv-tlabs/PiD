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

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment_2k.shared_config import (
    _common_model_overrides,
)

# SigLIP-2 So400M + Scale-RAE ViT-XL tokenizer (state_ch=1152, virtual 16x
# compression). 8x SR (LQ=256 → HQ=2048): the default PID_SR4X net has
# ``sr_scale=4`` so we override it to 8 here.
_SIGLIP_MODEL_CONFIG = _common_model_overrides(state_ch=1152)
_SIGLIP_MODEL_CONFIG["net"] = {
    "lq_latent_channels": 1152,
    "latent_spatial_down_factor": 16,
    "sr_scale": 8,
}

PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP = LazyDict(
    dict(
        defaults=[
            {"override /model": "ddp_inference_pid"},
            {"override /net": "pid_sr4x"},
            {"override /conditioner": "pid_caption_lq"},
            {"override /ckpt_type": "dcp"},
            {"override /ema": None},
            {"override /checkpoint": "local"},
            {"override /tokenizer": "scale_rae_tokenizer"},
            "_self_",
        ],
        job=dict(group="pid_official", name="PiD_res2k_sr8x_official_siglip_distill_4step"),
        model=dict(config=_SIGLIP_MODEL_CONFIG),
    ),
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP["job"]["name"],
    node=PID_RES2K_SR8X_OFFICIAL_SIGLIP_DISTILL_4STEP,
)
