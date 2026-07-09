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

from pid._src.tokenizers.dinov2_vae import DINOv2RAEConfig
from pid._src.tokenizers.flux2_vae import Flux2VAEConfig
from pid._src.tokenizers.flux_vae import FluxVAEConfig, SD3VAEConfig
from pid._src.tokenizers.qwenimage_vae import QwenImageVAEConfig
from pid._src.tokenizers.scale_rae_vae import ScaleRAEConfig
from pid._src.tokenizers.sdxl_vae import SDXLVAEConfig


def register_tokenizer():
    cs = ConfigStore.instance()
    cs.store(group="tokenizer", package="model.config.tokenizer", name="flux_vae_tokenizer", node=FluxVAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="sd3_vae_tokenizer", node=SD3VAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="flux2_vae_tokenizer", node=Flux2VAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="sdxl_vae_tokenizer", node=SDXLVAEConfig)
    cs.store(
        group="tokenizer", package="model.config.tokenizer", name="qwenimage_vae_tokenizer", node=QwenImageVAEConfig
    )
    cs.store(group="tokenizer", package="model.config.tokenizer", name="dinov2_rae_tokenizer", node=DINOv2RAEConfig)
    cs.store(group="tokenizer", package="model.config.tokenizer", name="scale_rae_tokenizer", node=ScaleRAEConfig)
