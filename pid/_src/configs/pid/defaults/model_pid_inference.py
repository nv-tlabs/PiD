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
from pid._src.models.pid_distill_model_infer import PidInferenceConfig, PidInferenceModel

DDP_INFERENCE_PID_CONFIG = LazyDict(
    dict(
        model=L(PidInferenceModel)(
            config=PidInferenceConfig(
                precision="bfloat16",
            ),
            _recursive_=False,
        ),
    ),
    flags={"allow_objects": True},
)


def register_model_pid_inference():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="ddp_inference_pid",
        node=DDP_INFERENCE_PID_CONFIG,
    )
