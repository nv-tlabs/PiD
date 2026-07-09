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
from typing import Optional

import numpy as np
import torch

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


class torch_CNHW_uint8_to_NCHW_float32(Augmentor):
    """
    Convert Torch Tensor [C, N, H, W] uint8 -> Torch float [N, C, H, W] in [0,1].
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.output_keys = output_keys or input_keys

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            tensor = data_dict[in_k].clone()
            # permute returns a new view, can't be inplace, unavoidable
            tensor = tensor.permute(1, 0, 2, 3).float() / 255.0  # [C,N,H,W] -> [N,C,H,W]
            data_dict[out_k] = tensor.contiguous()  # contiguous() allocates if needed, but better for downstream ops
        return data_dict


class torch_NCHW_float32_to_CNHW_uint8(Augmentor):
    """
    Convert Torch float [N, C, H, W] in [0,1] -> Torch uint8 [C, N, H, W].
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.output_keys = output_keys or input_keys

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            tensor = data_dict.pop(in_k)
            tensor = (
                tensor.permute(1, 0, 2, 3).clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            )  # [N,C,H,W] -> [C,N,H,W]
            data_dict[out_k] = tensor.contiguous()

        return data_dict


class np_CNHW_uint8_to_NHWC_uint8(Augmentor):
    """
    Convert NumPy array [C,N,H,W] uint8 -> NumPy array [N,H,W,C] uint8.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.output_keys = output_keys or input_keys

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            array = data_dict[in_k].clone()
            data_dict[out_k] = np.transpose(array, (1, 2, 3, 0))  # [C,N,H,W] -> [N,H,W,C]

        return data_dict


class np_NHWC_uint8_to_CHWN_uint8(Augmentor):
    """
    Convert NumPy array [N,H,W,C] uint8 -> NumPy array [C,H,W,N] uint8.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.output_keys = output_keys or input_keys

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            array = data_dict[in_k]
            data_dict[out_k] = np.transpose(array, (3, 0, 1, 2))  # [N,H,W,C] -> [C,H,W,N]

        return data_dict


class NumpyToTensor(Augmentor):
    """
    Convert NumPy array to Torch tensor.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.output_keys = output_keys or input_keys

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            data_dict[out_k] = torch.from_numpy(data_dict[in_k])

        return data_dict


class Copy(Augmentor):
    """
    Copy the input to the output.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        assert len(input_keys) == len(output_keys), "Input and output keys must have the same length"
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            data_dict[out_k] = deepcopy(data_dict[in_k])
        return data_dict


class PopAndClamp(Augmentor):
    """
    Pop the input and clamp the output to the range [0, 1].
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        assert len(input_keys) == len(output_keys), "Input and output keys must have the same length"
        super().__init__(input_keys, output_keys, args)
        self.clamp_min_max = args["clamp_min_max"]

    def __call__(self, data_dict: dict) -> dict:
        for in_k, out_k in zip(self.input_keys, self.output_keys):
            data_dict[out_k] = data_dict.pop(in_k).clamp(self.clamp_min_max[0], self.clamp_min_max[1])

        return data_dict
