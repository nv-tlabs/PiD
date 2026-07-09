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


try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
from typing import List, Optional

import attrs
from webdataset.handlers import warn_and_continue

import pid._ext.imaginaire.datasets.webdataset.decoders.image as image_decoders
import pid._ext.imaginaire.datasets.webdataset.decoders.json_decoder as json_decoder
import pid._ext.imaginaire.datasets.webdataset.decoders.pickle as pickle_decoders
import pid._ext.imaginaire.datasets.webdataset.decoders.video_decoder as video_decoder
import pid._ext.imaginaire.datasets.webdataset.distributors as distributors
import pid._ext.imaginaire.datasets.webdataset.webdataset as webdataset
from pid._ext.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from pid._ext.imaginaire.utils import log
from pid._src.datasets.augmentor_provider import (
    AUGMENTOR_OPTIONS,
    AugmentationConfig,
)
from pid._src.datasets.data_sources.data_registration import (
    create_dataset_infos_local,
)
from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO


def filter_dataset_infos_by_key_words(
    dataset_infos: list,
    exclude: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
) -> list:
    """Filter wdinfo paths inside each DatasetInfo by resolution tags embedded in the path.

    Args:
        exclude: Drop paths containing any of these substrings (e.g. ["resolution_lt_720"]).
        include: Keep only paths containing at least one of these substrings.
                 Applied after exclude. If None, no include filtering is applied.

    Image resolution tags:  resolution_gt_1080 / resolution_lt_1080 / resolution_lt_720
    Video resolution tags:  resolution_480 / resolution_720 / resolution_1024 / resolution_2048 / resolution_gt_2048
    """
    if exclude is None and include is None:
        return dataset_infos
    result = []
    for info in dataset_infos:
        paths = info.wdinfo
        if exclude:
            paths = [p for p in paths if not any(tag in p for tag in exclude)]
        if include:
            paths = [p for p in paths if any(tag in p for tag in include)]
        log.info(f"[{info.source}] resolution filter: {len(info.wdinfo)} -> {len(paths)} wdinfo paths")
        result.append(attrs.evolve(info, wdinfo=paths))
    return result


def get_video_dataset(
    dataset_name: str,
    resolution: str,
    video_decoder_name: str = "video_naive_bytes",
    is_train: bool = True,
    embedding_type: Optional[str] = None,
    num_video_frames: int = 121,
    augmentor_config: AugmentationConfig = AugmentationConfig(),
    augmentor_name: str = "video_vsr_augmentor",
    detshuffle: bool = False,
    input_keys: List[str] = None,
    exclude_resolution_tags: Optional[List[str]] = None,
    include_resolution_tags: Optional[List[str]] = None,
) -> webdataset.Dataset:
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    aspect_ratio_choices = augmentor_config.aspect_ratio_choices
    dataset_infos = create_dataset_infos_local(dataset_name, "video", input_keys, embedding_type, aspect_ratio_choices)
    dataset_infos = filter_dataset_infos_by_key_words(
        dataset_infos,
        exclude=exclude_resolution_tags,
        include=include_resolution_tags,
    )
    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        augmentor_config=augmentor_config,
        num_video_frames=num_video_frames,
        embedding_type=embedding_type,
    )

    if (
        USE_MEGATRON
        and parallel_state.is_initialized()
        and (
            parallel_state.get_context_parallel_world_size() > 1
            or parallel_state.get_tensor_model_parallel_world_size() > 1
        )
    ):
        log.critical(
            f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = distributors.ShardlistBasicParallelSync(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=True,
            is_infinite_loader=is_train,
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        log.critical(f"We use naive ShardlistBasic distributor for video dataset.")
        distributor = distributors.ShardlistBasic(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    if not is_train:
        detshuffle = True  # overwrite detshuffle if validation mode

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=100,
        streaming_download=True,
        dataset_info=dataset_infos,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
            ),
            pickle_decoders.pkl_decoder,
            json_decoder.json_decoder,
        ],
        augmentation=augmentor,
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )

    dataset_kwargs = {"decoder_handler": warn_and_continue, "detshuffle": detshuffle}
    return webdataset.Dataset(config=video_data_config, **dataset_kwargs)


def get_image_dataset(
    dataset_name: str,
    resolution: str,
    is_train: bool = True,
    embedding_type: Optional[str] = None,
    augmentor_config: AugmentationConfig = AugmentationConfig(),
    augmentor_name: str = "image_vsr_augmentor",
    detshuffle: bool = False,
    input_keys: List[str] = None,
    exclude_resolution_tags: Optional[List[str]] = None,
    include_resolution_tags: Optional[List[str]] = None,
) -> webdataset.Dataset:
    assert resolution in IMAGE_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in IMAGE_RES_SIZE_INFO."
    aspect_ratio_choices = augmentor_config.aspect_ratio_choices
    dataset_infos = create_dataset_infos_local(dataset_name, "image", input_keys, embedding_type, aspect_ratio_choices)
    dataset_infos = filter_dataset_infos_by_key_words(
        dataset_infos,
        exclude=exclude_resolution_tags,
        include=include_resolution_tags,
    )
    augmentation = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        augmentor_config=augmentor_config,
        embedding_type=embedding_type,
    )

    if (
        USE_MEGATRON
        and parallel_state.is_initialized()
        and (
            parallel_state.get_context_parallel_world_size() > 1
            or parallel_state.get_tensor_model_parallel_world_size() > 1
        )
    ):
        log.critical(
            f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, TP :{parallel_state.get_tensor_model_parallel_world_size()} for image dataset, switch to ShardlistBasicParallelSync distributor"
        )
        distributor = distributors.ShardlistBasicParallelSync(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=True,
            is_infinite_loader=is_train,
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        log.critical(f"We use naive ShardlistBasic distributor for image dataset.")
        distributor = distributors.ShardlistBasic(
            shuffle=is_train,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    if not is_train:
        detshuffle = True  # overwrite detshuffle if validation mode

    image_data_config = DatasetConfig(
        keys=[],
        buffer_size=25,
        streaming_download=True,
        dataset_info=dataset_infos,
        distributor=distributor,
        decoders=[
            image_decoders.pil_loader,
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentation,
    )

    dataset_kwargs = {"decoder_handler": warn_and_continue, "detshuffle": detshuffle}
    return webdataset.Dataset(config=image_data_config, **dataset_kwargs)
