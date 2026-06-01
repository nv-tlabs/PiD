# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
