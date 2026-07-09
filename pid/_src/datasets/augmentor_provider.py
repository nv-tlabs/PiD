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

from typing import Optional, Union

import attrs
import torchvision.transforms.functional as transforms_F

import pid._ext.imaginaire.datasets.webdataset.augmentors.image.cropping as cropping
import pid._ext.imaginaire.datasets.webdataset.augmentors.image.normalize as normalize
import pid._ext.imaginaire.datasets.webdataset.augmentors.image.resize as resize
import pid._src.datasets.augmentors.add_aspect_ratio as add_aspect_ratio
import pid._src.datasets.augmentors.merge_datadict as merge_datadict
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.utils import log
from pid._src.datasets.augmentors.caption_augmentors import CaptionExtractor
from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO

AUGMENTOR_OPTIONS = {}


@attrs.define(slots=False)
class AugmentationConfig:
    """Configuration for video / image vsr augmentation."""

    aspect_ratio_choices: list[str] = attrs.field(default=None)
    resolution_downsample_factor: Union[int, str] = 1  # int (1,2,4,...) or "adaptive"
    use_center_crop: bool = True
    # Near-miss rescue threshold for ResizeScale. When the downsampled image falls
    # below the final crop size but min(resized_h/final_h, resized_w/final_w) is
    # still above this value, Lanczos-upscale to the final crop size instead of
    # dropping the sample. None disables the rescue path (i.e. drop as before).
    tolerate_smaller_shape_threshold: Optional[float] = None


def augmentor_register(key):
    log.info(f"registering {key}...")

    def decorator(func):
        AUGMENTOR_OPTIONS[key] = func
        return func

    return decorator


@augmentor_register("image_caption_augmentor")
def get_image_caption_augmentor(
    resolution: str,
    augmentor_config: AugmentationConfig = AugmentationConfig(),
    embedding_type: Optional[str] = None,
):
    """Multi-aspect-ratio image augmentor with caption extraction for T2I training.

    Like image_vsr_augmentor but without VSR-specific steps (embedding_transform,
    add_predefined_embedding, append_fps_frames). Adds CaptionExtractor to parse
    caption JSON {"prompt": "...", "file_name": "..."} into raw prompt string.
    """

    del embedding_type  # unused for image+caption training

    allow_aspect_ratios = augmentor_config.aspect_ratio_choices
    if allow_aspect_ratios is None:
        allow_aspect_ratios = IMAGE_RES_SIZE_INFO[resolution].keys()

    augmentation = {
        "rename_keys": L(merge_datadict.RenameKeys)(
            input_keys=["images"],
            output_keys=["image"],
        ),
        "infer_aspect_ratio": L(add_aspect_ratio.InferAspectRatio)(
            input_keys=["image"],
            args={"aspect_ratio_choices": allow_aspect_ratios},
        ),
        "resize_scale": L(resize.ResizeScale)(
            input_keys=["image"],
            args={
                "scale_factor": augmentor_config.resolution_downsample_factor,
                "interpolation": transforms_F.InterpolationMode.LANCZOS,
                "larger_than_final_crop_size": True,
                "tolerate_smaller_shape_threshold": augmentor_config.tolerate_smaller_shape_threshold,
                "size": IMAGE_RES_SIZE_INFO[resolution],
            },
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["image"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "center_crop": L(cropping.CenterCrop)(
            input_keys=["image"],
            args={"size": IMAGE_RES_SIZE_INFO[resolution]},
        ),
        "caption_extractor": L(CaptionExtractor)(
            input_keys=["caption", "captions_qwen2p5_7b_v4"],  # either or
            args={"output_key": "caption"},
        ),
    }
    return augmentation


@augmentor_register("image_caption_multi_resolution_augmentor")
def get_image_caption_multi_resolution_augmentor(
    resolution: str,
    augmentor_config: AugmentationConfig = AugmentationConfig(),
    embedding_type: Optional[str] = None,
):
    """Image+caption augmentor with per-sample resolution sampling at the
    dataloader level (no model-side multi_resolution resize needed).

    Each sample's native (W, H) determines the largest grid level that fits;
    the AspectRatioDataLoader then buckets samples by the composite
    f"L{level}_{ar}" key written into data_dict["aspect_ratio"] so every
    batch is shape-uniform.

    The `resolution` arg selects the highest grid preset. Use "3072" for 2K..3K,
    "3840" to avoid the 4096x4096 square bucket, or "4096" for the full legacy
    grid.
    """
    from pid._src.datasets.augmentors.multi_resolution_aspect_ratio import (
        CenterCropByTargetSize,
        InferMultiResolutionAspectRatio,
        ResizeScaleByTargetSize,
    )

    del embedding_type  # unused for image+caption multi-resolution training

    augmentation = {
        "rename_keys": L(merge_datadict.RenameKeys)(
            input_keys=["images"],
            output_keys=["image"],
        ),
        "infer_multi_resolution_aspect_ratio": L(InferMultiResolutionAspectRatio)(
            input_keys=["image"],
            args={"max_resolution": resolution},
        ),
        "resize_scale": L(ResizeScaleByTargetSize)(
            input_keys=["image"],
            args={
                "scale_factor": augmentor_config.resolution_downsample_factor,
                "interpolation": transforms_F.InterpolationMode.LANCZOS,
            },
        ),
        "normalize": L(normalize.Normalize)(
            input_keys=["image"],
            args={"mean": 0.5, "std": 0.5},
        ),
        "center_crop": L(CenterCropByTargetSize)(
            input_keys=["image"],
        ),
        "caption_extractor": L(CaptionExtractor)(
            input_keys=["caption", "captions_qwen2p5_7b_v4"],  # either or
            args={"output_key": "caption"},
        ),
    }
    return augmentation
