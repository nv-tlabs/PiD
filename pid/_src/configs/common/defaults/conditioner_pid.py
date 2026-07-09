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

# Registers:
# - pid_caption_lq: caption (10% drop) + lq_latent (10% drop)
# - pid_lq_only: caption (0% drop) + lq_latent (10% drop)
#   When caption dropout=0, uncondition also keeps caption — CFG only applies to LQ.

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.modules.conditioner import (
    CaptionStringDrop,
    LQTensorDrop,
    PidConditioner,
)

# Caption + LQ latent with 10% dropout each (full dual CFG)
Pid_CaptionLQ_Config = L(PidConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.1,
    ),
    lq_latent=L(LQTensorDrop)(
        input_key="LQ_latent",
        output_key="lq_latent",
        dropout_rate=0.1,
    ),
)

# LQ-only CFG: caption never dropped, only the LQ latent is dropped for CFG
Pid_LQOnly_Config = L(PidConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.0,  # Never dropped -> uncondition also keeps caption
    ),
    lq_latent=L(LQTensorDrop)(
        input_key="LQ_latent",
        output_key="lq_latent",
        dropout_rate=0.1,
    ),
)


def register_conditioner_pid():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pid_caption_lq",
        node=Pid_CaptionLQ_Config,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pid_lq_only",
        node=Pid_LQOnly_Config,
    )
