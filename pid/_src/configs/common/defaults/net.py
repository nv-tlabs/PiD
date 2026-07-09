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

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.networks.pid_net import PidNet

# =============================================================================
# Network config — single base, override in experiments
# =============================================================================

# Base PidNet network (controlnet injection — the only mode supported here).
# Experiments override: lq_in_channels, lq_latent_channels, lq_gate_type,
#                       lq_interval, train_lq_proj_only, etc.
PID_SR4X = L(PidNet)(
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
    # SR-specific defaults (controlnet + latent-only)
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_latent_channels=16,
    lq_hidden_dim=512,
    lq_latent_unpatchify_factor=1,
    lq_conv_padding_mode="zeros",
    lq_aux_rgb_head=False,
    lq_aux_rgb_head_latent_block_idx=-1,
    lq_gate_type="sigma_aware_per_token_per_dim",
    lq_interval=2,
    zero_init_lq=True,
    train_lq_proj_only=False,
    sr_scale=4,
    pit_lq_inject=False,
)

# PiD v1.5 changes:
PID_SR4X_V1PT5 = deepcopy(PID_SR4X)
PID_SR4X_V1PT5["lq_conv_padding_mode"] = "replicate"
PID_SR4X_V1PT5["rope_mode"] = "ntk_aware"
PID_SR4X_V1PT5["rope_ref_h"] = 2048
PID_SR4X_V1PT5["rope_ref_w"] = 2048
PID_SR4X_V1PT5["lq_aux_rgb_head"] = True
PID_SR4X_V1PT5["pit_lq_inject"] = True
PID_SR4X_V1PT5["lq_gate_type"] = "sigma_aware_per_token"
PID_SR4X_V1PT5["train_lq_proj_only"] = True
PID_SR4X_V1PT5["lq_interval"] = 2
PID_SR4X_V1PT5["lq_hidden_dim"] = 1024
PID_SR4X_V1PT5["lq_num_res_blocks"] = 4


# PiD v1.5 + Flux2 normalized latent through the generic LQProjection2D.
# Keeps the BN-normalized latent values, only unpatchifies 2x2 channel packing
# back to the raw Flux2 latent grid: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8].
PID_SR4X_V1PT5_FOR_FLUX2 = deepcopy(PID_SR4X_V1PT5)
PID_SR4X_V1PT5_FOR_FLUX2["lq_latent_channels"] = 128
PID_SR4X_V1PT5_FOR_FLUX2["latent_spatial_down_factor"] = 16
PID_SR4X_V1PT5_FOR_FLUX2["lq_latent_unpatchify_factor"] = 2


def register_pid_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="pid_sr4x",
        node=PID_SR4X,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="pid_sr4x_v1pt5",
        node=PID_SR4X_V1PT5,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="pid_sr4x_v1pt5_for_flux2",
        node=PID_SR4X_V1PT5_FOR_FLUX2,
    )
