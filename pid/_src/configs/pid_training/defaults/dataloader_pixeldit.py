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


from typing import List, Union

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.dataloaders.aspect_ratio_dataloader import get_aspect_ratio_dataloader
from pid._ext.imaginaire.dataloaders.cached_replay_dataloader import get_cached_replay_dataloader
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.datasets.augmentor_provider import AugmentationConfig
from pid._src.datasets.data_sources.dataset_definition import IMAGES_DATASETS
from pid._src.datasets.dataset_provider import get_image_dataset

_IMAGE_DATASETS = list(IMAGES_DATASETS.keys())
_IMAGE_RESOLUTIONS = ["1024", "2048", "3072", "3840", "4096"]
_IMAGE_MULTI_RESOLUTIONS_UPPERBOUND = ["3072", "3840", "4096"]
_IMAGE_BATCH_SIZES = [1, 2, 4, 8, 12, 16, 32, 64]


def get_image_loader(
    dataset_name: str,
    resolution: str,
    batch_size: int = 8,
    is_train: bool = True,
    augmentor_name: str = "image_caption_augmentor",
    aspect_ratio_choices: list[str] = ["16,9", "3,2", "4,3", "1,1", "3,4", "2,3", "9,16"],
    resolution_downsample_factor: Union[int, str] = "adaptive",
    exclude_resolution_tags: List[str] | None = None,
    include_resolution_tags: List[str] | None = None,
    num_workers: int = 4,
    total_max_samples: int | None = None,
):
    if len(aspect_ratio_choices) > 1:
        dataloader_fn = L(get_aspect_ratio_dataloader)
        dataloader_kwargs = {"aspect_ratio_dataloader_name": "image_ar_dataloader"}
    else:
        dataloader_fn = L(get_cached_replay_dataloader)
        dataloader_kwargs = {"cache_replay_name": "image_dataloader"}

    input_keys = ["image", "caption"]

    _IMAGE_LOADER = dataloader_fn(
        dataset=L(get_image_dataset)(
            dataset_name=dataset_name,
            resolution=resolution,
            is_train=is_train,
            augmentor_config=L(AugmentationConfig)(
                aspect_ratio_choices=aspect_ratio_choices,
                resolution_downsample_factor=resolution_downsample_factor,
                tolerate_smaller_shape_threshold=0.9,
            ),
            augmentor_name=augmentor_name,
            input_keys=input_keys,
            exclude_resolution_tags=exclude_resolution_tags,
            include_resolution_tags=include_resolution_tags,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        webdataset=True,
        total_max_samples=total_max_samples,
        **dataloader_kwargs,
    )
    return _IMAGE_LOADER


def register_text_to_image_data():
    cs = ConfigStore.instance()

    for dataset_name in _IMAGE_DATASETS:
        for res in _IMAGE_RESOLUTIONS:
            for batch_size in _IMAGE_BATCH_SIZES:
                _LOADER = get_image_loader(
                    dataset_name=dataset_name,
                    resolution=res,
                    batch_size=batch_size,
                    is_train=True,
                    augmentor_name="image_caption_augmentor",
                )
                cs.store(
                    group="data_train",
                    package="dataloader_train",
                    name=f"pixeldit_{dataset_name}_{batch_size}bs_{res}",
                    node=_LOADER,
                )


def register_text_to_image_multi_resolution_data():
    cs = ConfigStore.instance()

    for dataset_name in _IMAGE_DATASETS:
        for res in _IMAGE_MULTI_RESOLUTIONS_UPPERBOUND:
            for batch_size in _IMAGE_BATCH_SIZES:
                _LOADER = get_image_loader(
                    dataset_name=dataset_name,
                    resolution=res,
                    batch_size=batch_size,
                    is_train=True,
                    augmentor_name="image_caption_multi_resolution_augmentor",
                )
                cs.store(
                    group="data_train",
                    package="dataloader_train",
                    name=f"pixeldit_{dataset_name}_{batch_size}bs_multires_2048_{res}",
                    node=_LOADER,
                )
