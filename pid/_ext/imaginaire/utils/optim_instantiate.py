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

import hydra
import torch
from omegaconf import ListConfig
from torch import nn

from pid._ext.imaginaire.utils import log


def get_regular_param_group(net: nn.Module):
    """
    seperate the parameters of the network into two groups: decay and no_decay.
    based on nano_gpt codebase.
    """
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def get_base_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    net_decay_param, net_nodecay_param = get_regular_param_group(model)

    num_decay_params = sum(p.numel() for p in net_decay_param)
    num_nodecay_params = sum(p.numel() for p in net_nodecay_param)
    net_param_total = num_decay_params + num_nodecay_params
    log.info(f"total num parameters : {net_param_total:,}")

    param_group = [
        {
            "params": net_decay_param + net_nodecay_param,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    for k, v in kwargs.items():
        if isinstance(v, ListConfig):
            kwargs[k] = list(v)

    # When parameters are bfloat16/float16, use FusedAdam which stores optimizer
    # states (exp_avg, exp_avg_sq) in float32 — critical for convergence.
    # torch.optim.AdamW stores states in parameter dtype, causing precision loss.
    all_params = net_decay_param + net_nodecay_param
    has_low_precision = any(p.dtype in (torch.float16, torch.bfloat16) for p in all_params)

    if optim_type == "adamw" and has_low_precision:
        from pid._ext.imaginaire.utils.fused_adam import FusedAdam

        # FusedAdam accepts: betas, eps, weight_decay, adam_w_mode, capturable, master_weights
        # Filter out kwargs not accepted by FusedAdam (e.g. 'fused' from torch.optim.AdamW)
        fused_adam_keys = {"betas", "eps", "adam_w_mode", "capturable", "master_weights", "bias_correction"}
        fused_kwargs = {k: v for k, v in kwargs.items() if k in fused_adam_keys}
        if "betas" in fused_kwargs and isinstance(fused_kwargs["betas"], list):
            fused_kwargs["betas"] = tuple(fused_kwargs["betas"])
        low_dtypes = {p.dtype for p in all_params if p.dtype in (torch.float16, torch.bfloat16)}
        log.info(f"Using FusedAdam (float32 optimizer states) for {low_dtypes} parameters")
        return FusedAdam(param_group, **fused_kwargs)

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    return opt_cls(param_group, **kwargs)


def get_base_scheduler(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scheduler_config: dict,
):
    net_scheduler = hydra.utils.instantiate(scheduler_config)
    net_scheduler.model = model
    num_param_groups = len(optimizer.param_groups)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            net_scheduler.schedule,
        ]
        * num_param_groups,
    )
