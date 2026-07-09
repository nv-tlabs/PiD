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
import os
from functools import wraps

import torch

from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.misc import get_local_tensor_if_DTensor

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]


def update_master_weights(optimizer: torch.optim.Optimizer):
    if getattr(optimizer, "master_weights", False) and optimizer.param_groups_master is not None:
        params, master_params = [], []
        for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
            for p, p_master in zip(group["params"], group_master["params"]):
                params.append(get_local_tensor_if_DTensor(p.data))
                master_params.append(p_master.data)
        torch._foreach_copy_(params, master_params)


class sync_timer:
    """
    Synchronized timer to count the inference time of `nn.Module.forward` or else.
    set env var SYNC_TIMER=1 to enable logging!

    Example as context manager:
    ```python
    with timer('name'):
        run()
    ```

    Example as decorator:
    ```python
    @timer('name')
    def run():
        pass
    ```
    """

    def __init__(self, name=None, flag_env="SYNC_TIMER"):
        self.name = name
        self.flag_env = flag_env

    def __enter__(self):
        if os.environ.get(self.flag_env, "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.environ.get(self.flag_env, "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                log.info(f"{self.name} takes {self.time / 1000:.4f}s", rank0_only=False)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper
