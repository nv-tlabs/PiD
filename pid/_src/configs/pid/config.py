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

from typing import Any, List

import attrs

from pid._ext.imaginaire import config
from pid._ext.imaginaire.trainer import ImaginaireTrainer as Trainer
from pid._ext.imaginaire.utils.config_helper import import_all_modules_from_package
from pid._src.configs.common.defaults.checkpoint import register_checkpoint
from pid._src.configs.common.defaults.ckpt_type import register_ckpt_type
from pid._src.configs.common.defaults.conditioner_pid import register_conditioner_pid
from pid._src.configs.common.defaults.conditioner_pixeldit import register_conditioner_pixeldit
from pid._src.configs.common.defaults.ema import register_ema
from pid._src.configs.common.defaults.net import (
    register_pid_net,
)
from pid._src.configs.common.defaults.tokenizer import register_tokenizer
from pid._src.configs.pid.defaults.model_pid_inference import (
    register_model_pid_inference,
)


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "ddp_inference_pid"},
            {"net": None},
            {"conditioner": None},
            {"ema": "power"},
            {"tokenizer": None},
            {"checkpoint": "local"},
            {"ckpt_type": "dummy"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "pid"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.callbacks = None

    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    register_conditioner_pixeldit()
    register_model_pid_inference()
    register_pid_net()
    register_conditioner_pid()

    import_all_modules_from_package("pid._src.configs.pid.experiment_2k", reload=True)
    import_all_modules_from_package("pid._src.configs.pid.experiment_2kto4k", reload=True)
    import_all_modules_from_package("pid._src.configs.pid.experiment_2kto4k_v1pt5", reload=True)
    return c
