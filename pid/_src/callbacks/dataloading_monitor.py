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

import time

import numpy as np
import torch
import wandb

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import distributed
from pid._ext.imaginaire.utils.callback import Callback
from pid._ext.imaginaire.utils.easy_io import easy_io


class DetailedDataLoadingSpeedMonitor(Callback):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        save_s3: bool = False,
    ):
        self.every_n = every_n
        self.step_size = step_size
        self.should_run = False
        self.start_dataloading_time = None
        self.dataloading_time = None
        self.name = self.__class__.__name__
        self.save_s3 = save_s3
        self.time_delta_list = []

    def on_before_dataloading(self, iteration: int = 0) -> None:
        # We want to run it one iteration before on_training_step_start should_run is set to True.
        global_step = iteration // self.step_size
        self.should_run = (global_step + 1) % self.every_n == 0
        self.start_dataloading_time = time.time()

    def on_after_dataloading(self, iteration: int = 0) -> None:
        self.time_delta_list.append(time.time() - self.start_dataloading_time)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.should_run:
            self.should_run = False
            cur_rank_mean, cur_rank_max = np.mean(self.time_delta_list), np.max(self.time_delta_list)
            self.time_delta_list = []  # Reset the list

            dataloading_time_gather_list = distributed.all_gather_tensor(
                torch.tensor([cur_rank_mean, cur_rank_max]).cuda()
            )
            wandb_info = {
                f"{self.name}_mean/dataloading_{k:03d}": v[0].item() for k, v in enumerate(dataloading_time_gather_list)
            }
            wandb_info.update(
                {
                    f"{self.name}_max/dataloading_{k:03d}": v[1].item()
                    for k, v in enumerate(dataloading_time_gather_list)
                }
            )
            mean_times = torch.stack(dataloading_time_gather_list)[:, 0]
            slowest_dataloading_rank_id = torch.argmax(mean_times)
            max_dataloading = torch.max(mean_times)
            wandb_info.update(
                {
                    "slowest_rank/slowest_dataloading_rank": slowest_dataloading_rank_id.item(),
                    "slowest_rank/slowest_dataloading_time": max_dataloading.item(),
                }
            )

            if wandb.run:
                wandb.log(wandb_info, step=iteration)

            if self.save_s3 and distributed.is_rank0():
                easy_io.dump(
                    wandb_info,
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )
