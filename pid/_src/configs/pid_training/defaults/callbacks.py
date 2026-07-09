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

import copy

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.callbacks.wandb_distill_log import WandbDistillCallback
from pid._src.callbacks.wandb_log import WandbCallback
from pid._src.configs.common.defaults.callbacks import BASIC_CALLBACKS

_basic_callback = copy.deepcopy(BASIC_CALLBACKS)

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(WandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
)

# Distillation-specific wandb callback for DMD losses (vsd/dsm/recon/gan breakdown).
WANDB_DISTILL_CALLBACK = dict(
    wandb_distill=L(WandbDistillCallback)(
        save_s3="${upload_reproducible_setup}",
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_distill_10x=L(WandbDistillCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
        save_s3="${upload_reproducible_setup}",
    ),
)


def register_pid_training_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=_basic_callback)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb_distill", node=WANDB_DISTILL_CALLBACK)
