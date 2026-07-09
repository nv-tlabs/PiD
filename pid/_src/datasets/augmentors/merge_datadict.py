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

from typing import Optional

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.utils import log


class DataDictMerger(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Merge the dictionary associated with the input keys into data_dict. Only keys in output_keys are merged.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with dictionary associated with the input keys merged.
        """
        for key in self.input_keys:
            if key not in data_dict:
                log.warning(
                    f"DataDictMerger dataloader error: missing {key}, {data_dict['__url__']}, {data_dict['__key__']}",
                    rank0_only=False,
                )
                return None
            key_dict = data_dict.pop(key)
            for sub_key in key_dict:
                if sub_key in self.output_keys and sub_key not in data_dict:
                    data_dict[sub_key] = key_dict[sub_key]
            del key_dict
        return data_dict


class RenameKeys(Augmentor):
    """Rename the keys of the data_dict.

    It does the following: data_dict[new_key] = data_dict[old_key] for
    (old_key, new_key) in zip(input_keys, output_keys).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, data_dict: dict) -> dict:
        """Augmentor function.

        Args:
            data_dict (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with keys added/modified.
        """

        for old_key, new_key in zip(self.input_keys, self.output_keys):
            if old_key in data_dict:
                data_dict[new_key] = data_dict.pop(old_key)

        return data_dict


class AddIsProcessedFlag(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        data_dict["is_preprocessed"] = True
        return data_dict
