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

PID_RES2K_SR4X_OFFICIAL_SD3_DISTILL_4STEP = LazyDict(
    dict(
        defaults=[
            {"override /model": "ddp_inference_pid"},
            {"override /net": "pid_sr4x"},
            {"override /conditioner": "pid_caption_lq"},
            {"override /ckpt_type": "dcp"},
            {"override /ema": None},
            {"override /checkpoint": "local"},
            {"override /tokenizer": "sd3_vae_tokenizer"},
            "_self_",
        ],
        job=dict(group="pid_official", name="PiD_res2k_sr4x_official_sd3_distill_4step"),
        model=dict(config=_common_model_overrides(state_ch=16)),
    ),
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2K_SR4X_OFFICIAL_SD3_DISTILL_4STEP["job"]["name"],
    node=PID_RES2K_SR4X_OFFICIAL_SD3_DISTILL_4STEP,
)
