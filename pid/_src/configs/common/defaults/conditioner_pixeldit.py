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
# - pixeldit_caption: caption-only conditioner with 10% dropout

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.modules.conditioner import (
    CaptionStringDrop,
    PixelDiTConditioner,
)

# Caption-only conditioner with 10% dropout (matches original class_dropout_prob=0.1)
PixelDiTCaptionConfig = L(PixelDiTConditioner)(
    caption=L(CaptionStringDrop)(
        input_key="caption",
        output_key="caption",
        dropout_rate=0.1,
    ),
)


def register_conditioner_pixeldit():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pixeldit_caption",
        node=PixelDiTCaptionConfig,
    )
