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
from pid._src.models.pixeldit_model import PixelDiTModel, PixelDiTModelConfig
from pid._src.networks.pixeldit_official import PixDiT_T2I

# =============================================================================
# Model config
# =============================================================================

DDP_PIXELDIT_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(PixelDiTModel)(
        config=PixelDiTModelConfig(
            precision="bfloat16",
        ),
        _recursive_=False,
    ),
)

# =============================================================================
# Network configs
# =============================================================================

PIXELDIT_H1536_D14P2 = L(PixDiT_T2I)(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    repa_encoder_index=6,
    rope_mode="ntk_aware",
    rope_ref_h=2048,
    rope_ref_w=2048,
)

# =============================================================================
# Registration functions
# =============================================================================


def register_model_pixeldit():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_pixeldit", node=DDP_PIXELDIT_CONFIG)


def register_pixeldit_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="pixeldit_h1536_d14p2", node=PIXELDIT_H1536_D14P2)
