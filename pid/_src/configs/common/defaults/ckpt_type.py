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


from typing import Dict

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.checkpointer.dcp import DistributedCheckpointer
from pid._ext.imaginaire.checkpointer.dummy import Checkpointer as DummyCheckpointer
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.checkpointer.dcp_distill import DistillationCheckpointer

DUMMY_CHECKPOINTER: Dict[str, str] = L(DummyCheckpointer)()
DISTRIBUTED_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()
DISTILLATION_CHECKPOINTER: Dict[str, str] = L(DistillationCheckpointer)()


def register_ckpt_type():
    cs = ConfigStore.instance()
    cs.store(group="ckpt_type", package="checkpoint.type", name="dummy", node=DUMMY_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp", node=DISTRIBUTED_CHECKPOINTER)
    cs.store(group="ckpt_type", package="checkpoint.type", name="dcp_distill", node=DISTILLATION_CHECKPOINTER)
