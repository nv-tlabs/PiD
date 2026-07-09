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
from pid._src.configs.common.defaults.dataloader import register_training_and_val_data
from pid._src.configs.common.defaults.ema import register_ema
from pid._src.configs.common.defaults.net import register_pid_net
from pid._src.configs.common.defaults.optimizer import register_optimizer
from pid._src.configs.common.defaults.scheduler import register_scheduler
from pid._src.configs.common.defaults.tokenizer import register_tokenizer
from pid._src.configs.pid_training.defaults.callbacks import register_pid_training_callbacks
from pid._src.configs.pid_training.defaults.dataloader_pixeldit import (
    register_text_to_image_data,
    register_text_to_image_multi_resolution_data,
)
from pid._src.configs.pid_training.defaults.model_pid import (
    register_model_pid,
)
from pid._src.configs.pid_training.defaults.model_pixeldit import (
    register_model_pixeldit,
    register_pixeldit_net,
)


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock_image"},
            {"data_val": "mock_image"},
            {"optimizer": "adamw"},
            {"scheduler": "lambdalinear"},
            {"model": "ddp_pixeldit"},
            {"callbacks": "basic"},
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

    c.job.project = "pid_training"
    c.job.group = "placeholder"
    c.job.name = "placeholder_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    # Common registrations
    register_training_and_val_data()
    register_optimizer()
    register_scheduler()
    register_ema()
    register_tokenizer()
    register_checkpoint()
    register_ckpt_type()

    # PixelDiT models + nets + data + conditioner
    register_model_pixeldit()
    register_pixeldit_net()
    register_text_to_image_data()
    register_text_to_image_multi_resolution_data()
    register_conditioner_pixeldit()

    # PixelDiT SR models (PiD) + nets + conditioner
    register_model_pid()
    register_pid_net()
    register_conditioner_pid()

    # PiD training callbacks
    register_pid_training_callbacks()

    # PixelDiT finetune experiments
    import_all_modules_from_package("pid._src.configs.pid_training.experiment_pixeldit_finetune", reload=True)

    # PiD v1.5 experiments
    import_all_modules_from_package("pid._src.configs.pid_training.experiment_pid_v1pt5_flux", reload=True)
    import_all_modules_from_package("pid._src.configs.pid_training.experiment_pid_v1pt5_flux2", reload=True)
    import_all_modules_from_package("pid._src.configs.pid_training.experiment_pid_v1pt5_qwenimage", reload=True)

    return c
