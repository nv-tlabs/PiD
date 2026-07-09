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

import random
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Optional

import numpy as np
import torch

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.trainer import ImaginaireTrainer
from pid._ext.imaginaire.utils import distributed, log
from pid._ext.imaginaire.utils.callback import Callback


@contextmanager
def preserve_rng_state():
    """Keep periodic callbacks from perturbing the next training iteration."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.random.get_rng_state()
    torch_cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    )
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch.cuda.set_rng_state_all(torch_cuda_states)


class EveryN(Callback):
    def __init__(
        self,
        every_n: Optional[int] = None,
        step_size: int = 1,
        barrier_after_run: bool = True,
        run_at_start: bool = False,
        preserve_rng_state: bool = True,
    ) -> None:
        """Constructor for `EveryN`.

        Args:
            every_n (int): Frequency with which callback is run during training.
            step_size (int): Size of iteration step count. Default 1.
            barrier_after_run (bool): Whether to have a distributed barrier after each execution. Default True, to avoid timeouts.
            run_at_start (bool): Whether to run at the beginning of training. Default False.
            preserve_rng_state (bool): Whether to restore Python/NumPy/Torch RNG state after callback execution.
        """
        self.every_n = every_n
        if self.every_n == 0:
            log.warning(
                f"every_n is set to 0. Callback {self.__class__.__name__} will be invoked only once in the beginning of the training. Calls happens on_training_step_end will be skipped."
            )

        self.step_size = step_size
        self.barrier_after_run = barrier_after_run
        self.run_at_start = run_at_start
        self.preserve_rng_state = preserve_rng_state

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # every_n = 0 is a special case which means every_n_impl will be called only once in the beginning of the training
        if self.every_n != 0:
            trainer = self.trainer
            global_step = iteration // self.step_size
            should_run = (iteration == 1 and self.run_at_start) or (
                global_step % self.every_n == 0
            )  # (self.every_n - 1)
            if should_run:
                log.debug(f"Callback {self.__class__.__name__} fired on train_batch_end step {global_step}")
                was_training = model.training
                rng_context = preserve_rng_state if self.preserve_rng_state else nullcontext
                model.eval()
                try:
                    with rng_context():
                        self.every_n_impl(trainer, model, data_batch, output_batch, loss, iteration)
                finally:
                    model.train(was_training)
                log.debug(f"Callback {self.__class__.__name__} finished on train_batch_end step {global_step}")
                # add necessary barrier to avoid timeout
                if self.barrier_after_run:
                    distributed.barrier()

    @abstractmethod
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None: ...
