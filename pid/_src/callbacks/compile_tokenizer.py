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

import torch

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.callback import Callback


class CompileTokenizer(Callback):
    def __init__(self, enabled: bool = False, compile_after_iterations: int = 10, dynamic: bool = False):
        super().__init__()
        self.enabled = enabled
        self.compiled = False
        self.compile_after_iterations = compile_after_iterations
        self.skip_counter = 0
        self.dynamic = (
            dynamic  # If there are issues with constant recompilations you may set this value to None or True
        )

    def on_training_step_start(
        self, model: ImaginaireModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if not self.enabled or self.compiled:
            return

        if isinstance(model.tokenizer, torch.jit.ScriptModule):
            log.critical(
                f"The Tokenizer model {type(model.tokenizer)} is a JIT model, which is not compilable. The Tokenizer will not be compiled."
            )

        if self.skip_counter == self.compile_after_iterations:
            try:
                # PyTorch >= 2.7
                torch._dynamo.config.recompile_limit = 32
            except AttributeError:
                try:
                    torch._dynamo.config.cache_size_limit = 32
                except AttributeError:
                    log.warning(
                        "Tokenizer compilation requested, but Torch Dynamo is unavailable – skipping compilation."
                    )
                    self.enabled = False
                    return

            model.tokenizer.encode = torch.compile(model.tokenizer.encode, dynamic=self.dynamic)
            self.compiled = True
        self.skip_counter += 1
