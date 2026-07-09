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

from typing import Dict, List, Union

import torch
import webdataset

from pid._ext.imaginaire.lazy_config import instantiate


class CoupledDataLoader(webdataset.WebLoader):
    r"""
    A coupled dataloader that samples from each registered dataloader multiple times per iteration.
    Each __iter__ call will access each registered dataloader sample_num times and return results as a dict.

    Returns:
        A dictionary containing:
        - "samples": List of all samples from all dataloaders
        - "dataset_names": List of dataset names corresponding to each sample
        - "counts": Dictionary mapping dataset names to their sample counts

        Example: {
            "samples": [sample1, sample2, sample3, sample4, sample5],
            "dataset_names": ["image_data", "image_data", "image_data", "image_data", "video_data"],
            "counts": {"image_data": 4, "video_data": 1}
        }
    """

    def __init__(
        self, dataloaders: Dict[str, Dict[str, Union[torch.utils.data.DataLoader, webdataset.WebLoader, int]]]
    ):
        """
        Initialize the CoupledDataLoader with multiple datasets.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": dataloader, "sample_num": sample_num}

        Example:
            coupled_loader = CoupledDataLoader(
                dataloaders={
                    "image_data": {
                        "dataloader": webdataset.WebLoader(...),
                    },
                    "video_data": {
                        "dataloader": torch.utils.data.DataLoader(...),
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list = [], []

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader"}, f"Invalid config: {dataloader_data}"
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"]))

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        for data in self.dataloader_list:
            self.data_len += len(data)

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            all_samples = {}

            # 遍历每个注册的 dataloader
            for idx, (curr_dataloader, dataset_name) in enumerate(zip(self.dataloaders, self.dataset_name_list)):
                output = next(curr_dataloader)
                all_samples[dataset_name] = output

            yield all_samples

    def merge_samples(self, samples: List[Dict]) -> Dict:
        pass
