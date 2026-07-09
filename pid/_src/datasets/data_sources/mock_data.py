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


from functools import partial

import torch

from pid._ext.imaginaire.datasets.mock_dataset import CombinedDictDataset, LambdaDataset
from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO


def get_image_dataset(
    resolution: str = "1024",
    **kwargs,
):
    h, w = IMAGE_RES_SIZE_INFO[resolution]["9,16"]
    del kwargs
    return CombinedDictDataset(
        **{
            "images": LambdaDataset(partial(torch.randn, size=(3, h, w))),
            "image_size": LambdaDataset(partial(torch.tensor, [h, w, h, w], dtype=torch.float32)),
            "dataset_name": LambdaDataset(lambda: "image_data"),
            "caption": LambdaDataset(lambda: "placeholder"),
            "__url__": LambdaDataset(lambda: "placeholder"),
            "__key__": LambdaDataset(lambda: "placeholder"),
        }
    )
