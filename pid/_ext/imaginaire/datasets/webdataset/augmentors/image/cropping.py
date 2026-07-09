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

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from loguru import logger as logging

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.datasets.webdataset.augmentors.image.misc import (
    obtain_augmentation_size,
    obtain_image_size,
)


def _crop_tensor_or_ndarray(
    data: Union[torch.Tensor, np.ndarray], top: int, left: int, height: int, width: int
) -> Union[torch.Tensor, np.ndarray]:
    """Crop a tensor or numpy array.

    Args:
        data: Input data with shape (..., H, W) - last two dims are height and width.
        top: Top coordinate of the crop box.
        left: Left coordinate of the crop box.
        height: Height of the crop box.
        width: Width of the crop box.

    Returns:
        Cropped data with shape (..., height, width).
    """
    if isinstance(data, torch.Tensor):
        return transforms_F.crop(data, top, left, height, width)
    elif isinstance(data, np.ndarray):
        # Use ellipsis to handle any number of leading dimensions
        # data shape: (..., H, W), crop the last two dimensions
        return data[..., top : top + height, left : left + width]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _center_crop_tensor_or_ndarray(
    data: Union[torch.Tensor, np.ndarray], height: int, width: int
) -> Union[torch.Tensor, np.ndarray]:
    """Center crop a tensor or numpy array.

    Args:
        data: Input data with shape (..., H, W) - last two dims are height and width.
        height: Height of the crop box.
        width: Width of the crop box.

    Returns:
        Center cropped data with shape (..., height, width).
    """
    if isinstance(data, torch.Tensor):
        return transforms_F.center_crop(data, [height, width])
    elif isinstance(data, np.ndarray):
        orig_h, orig_w = data.shape[-2:]
        top = (orig_h - height) // 2
        left = (orig_w - width) // 2
        return data[..., top : top + height, left : left + width]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.skip_if_smaller = args.get("skip_if_smaller", True)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("size" in self.args), "Please specify size in args"

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = _center_crop_tensor_or_ndarray(data_dict[key], height, width)

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - width) // 2
        crop_y0 = (orig_h - height) // 2

        if crop_x0 < 0 or crop_y0 < 0:
            if self.skip_if_smaller:
                logging.warning(
                    f"Cropping failed. Skip this sample, please check your data sources. original_size(wxh): {orig_w}x{orig_h}, random_size(wxh): {width}x{height}."
                    + f"data_url: {data_dict['__url__']}, data_key: {data_dict['__key__']}"
                )
                return None
            else:
                return data_dict

        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))
        return data_dict


class RandomCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.skip_if_smaller = args.get("skip_if_smaller", True)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are randomly cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("size" in self.args), "Please specify size in args"

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

        # Check if image is smaller than crop size
        if orig_w < width or orig_h < height:
            if self.skip_if_smaller:
                logging.warning(
                    f"Random crop failed: image smaller than crop size. Skip this sample. original_size(wxh): {orig_w}x{orig_h}, crop_size(wxh): {width}x{height}."
                    + f"data_url: {data_dict['__url__']}, data_key: {data_dict['__key__']}"
                )
                return None
            else:
                return data_dict

        # Obtaining random crop coords
        crop_x0 = int(torch.randint(0, orig_w - width + 1, size=(1,)).item())
        crop_y0 = int(torch.randint(0, orig_h - height + 1, size=(1,)).item())

        # We also add the aug params we use. This will be useful for other transforms
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))

        # We must perform same random cropping for all input keys
        for key in self.input_keys:
            data_dict[key] = _crop_tensor_or_ndarray(data_dict[key], crop_y0, crop_x0, height, width)

        return data_dict
