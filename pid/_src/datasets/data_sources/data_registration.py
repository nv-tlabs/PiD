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

"""
Dataset registration for cosmos datasets with support for different caption types.
"""

import glob
import os
from typing import List, Union

from pid._ext.imaginaire import config
from pid._ext.imaginaire.datasets.webdataset.config.schema import DatasetInfo
from pid._ext.imaginaire.utils import log
from pid._src.datasets.data_sources.data_source_local import IMAGES_DATASET_SOURCES, VIDEO_DATASET_SOURCES
from pid._src.datasets.data_sources.dataset_definition import IMAGES_DATASETS, VIDEO_DATASETS

DATASET_OPTIONS = {}


def dataset_register(key):
    log.info(f"registering dataset {key}")

    def decorator(func):
        DATASET_OPTIONS[key] = func
        return func

    return decorator


def create_dataset_infos_local(
    dataset_name: str,
    data_type: str,
    input_keys: List[str] = None,
    embedding_type: Union[str, None] = None,
    aspect_ratio_choices: List[str] = None,
) -> list[DatasetInfo]:
    """Create dataset infos for webdatasets.

    Args:
        dataset_name: Name of the dataset
        data_type: Type of the data (video or image)
        embedding_type: Type of the embedding, None or umt5
        input_keys: List of keys for webdataset, None for default keys
            default keys are "video" and "caption" for video datasets, "image" and "caption" for image datasets
    """
    assert data_type in ["video", "image"], "Invalid data type"

    if data_type == "video":
        dataset_sources = VIDEO_DATASETS[dataset_name]
        if input_keys is None:
            input_keys = ["video", "caption"]
    else:
        dataset_sources = IMAGES_DATASETS[dataset_name]
        if input_keys is None:
            input_keys = ["image", "caption"]

    if embedding_type is not None:
        input_keys.append(embedding_type)

    dataset_infos = []
    for dataset_source_name in dataset_sources:
        dataset_source_path = (
            VIDEO_DATASET_SOURCES[dataset_source_name]
            if data_type == "video"
            else IMAGES_DATASET_SOURCES[dataset_source_name]
        )

        wdinfo_paths = glob.glob(os.path.join(dataset_source_path, "**", "wdinfo.json"), recursive=True)

        # filter by aspect_ratio_choices
        # first examine if "aspect_ratio_*" exists in wdinfo_paths, currently only cosmos4k does!
        aspect_ratio_in_wdinfo_paths = all(["aspect_ratio_" in wdinfo_path for wdinfo_path in wdinfo_paths])
        if aspect_ratio_in_wdinfo_paths and aspect_ratio_choices is not None:
            original_wdinfo_path_num = len(wdinfo_paths)
            available_sub_paths = [
                f"aspect_ratio_{width_comma_height.replace(',', '_')}" for width_comma_height in aspect_ratio_choices
            ]
            wdinfo_paths = [
                wdinfo_path
                for wdinfo_path in wdinfo_paths
                if any(available_sub_path in wdinfo_path for available_sub_path in available_sub_paths)
            ]
            filtered_wdinfo_path_num = len(wdinfo_paths)
            log.info(
                f"Filtered wdinfo_paths by aspect_ratio_choices {aspect_ratio_choices}: {original_wdinfo_path_num} -> {filtered_wdinfo_path_num}"
            )
            if filtered_wdinfo_path_num == 0:
                raise ValueError(f"No wdinfo_paths found for aspect_ratio_choices: {aspect_ratio_choices}")

        dataset_infos.extend(
            [
                DatasetInfo(
                    object_store_config=config.ObjectStoreConfig(
                        enabled=False,
                    ),
                    wdinfo=wdinfo_paths,
                    opts={},
                    per_dataset_keys=input_keys,
                    source=dataset_source_name,
                )
            ]
        )

    return dataset_infos
