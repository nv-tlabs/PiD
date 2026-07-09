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


import sys

import torch

from pid._ext.imaginaire.config import Config
from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.trainer import ImaginaireTrainer
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.callback import LowPrecisionCallback as BaseCallback


class LowPrecisionCallback(BaseCallback):
    """
    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    It also auto disabled when using fp32.
    """

    def __init__(self, config: Config, trainer: ImaginaireTrainer, update_iter: int):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if model.precision == torch.float32:
            log.critical("Using fp32, should disable master weights.")
            self.update_iter = sys.maxsize
        else:
            assert model.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision
