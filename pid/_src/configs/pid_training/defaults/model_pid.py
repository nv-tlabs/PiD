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
from pid._src.models.pid_distill_model import PidDistillModel, PidDistillModelConfig
from pid._src.models.pid_model import PidModel, PidModelConfig
from pid._src.trainer.trainer_distillation import DistillationTrainer

# =============================================================================
# Model config
# =============================================================================

DDP_PID_TEACHER_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(PidModel)(
        config=PidModelConfig(
            precision="bfloat16",
        ),
        _recursive_=False,
    ),
)

DDP_PID_DISTILLATION_CONFIG = LazyDict(
    dict(
        trainer=dict(
            distributed_parallelism="ddp",
            type=DistillationTrainer,
        ),
        model=L(PidDistillModel)(
            config=PidDistillModelConfig(
                precision="bfloat16",
            ),
            _recursive_=False,
        ),
    ),
    flags={"allow_objects": True},
)


def register_model_pid():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_pid_teacher", node=DDP_PID_TEACHER_CONFIG)
    cs.store(group="model", package="_global_", name="ddp_pid_distillation", node=DDP_PID_DISTILLATION_CONFIG)
