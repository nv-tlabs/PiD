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

import logging
import math
from typing import Optional

import omegaconf
import torchvision.transforms.functional as transforms_F

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.datasets.webdataset.augmentors.image.misc import (
    obtain_augmentation_size,
    obtain_image_size,
)


class ResizeSmallestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to smaller side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=out_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to larger side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

            scaling_ratio = min(out_size / orig_w, out_size / orig_h)
            target_size = [int(scaling_ratio * orig_h), int(scaling_ratio * orig_w)]

            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        )
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=(
                    self.args["interpolation"]
                    if "interpolation" in self.args
                    else transforms_F.InterpolationMode.BICUBIC
                ),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the larger ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the larger of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(img_size, (tuple, omegaconf.listconfig.ListConfig)), (
            f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        )
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] <= img_h and target_size[1] <= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"
        )

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeScale(Augmentor):
    """Fixed-ratio or adaptive downsampler used before the final crop stage.

    Args (via `args`):
        scale_factor: number, or the string "adaptive".
            - number: image is resized to (orig_w / scale_factor, orig_h / scale_factor).
              scale_factor == 1 is a no-op (short-circuited).
            - "adaptive": resize to the smallest aspect-ratio-preserving
              resolution that still covers the final crop size. Requires
              `larger_than_final_crop_size=True` and `aspect_ratio` in data_dict.
        larger_than_final_crop_size (bool): governs interaction with the downstream crop.
            - True:  read final crop size from `data_dict["aspect_ratio"]`, and if the
                     resized image is smaller than the crop target in either dim, return
                     None so webdataset drops the sample. Also required for "adaptive".
            - False: pure ratio-based downsample. final crop size is NOT consulted,
                     no sample is dropped, and "adaptive" mode is forbidden (asserts).
        tolerate_smaller_shape_threshold (float, optional): rescue knob for near-miss
            samples. Only consulted when `larger_than_final_crop_size=True` and the
            downsampled image fell below the final crop size. Let
                r = min(resized_h / final_h, resized_w / final_w)   # < 1 means too small
            If `threshold < r < 1`, instead of dropping, Lanczos-upscale so the
            bottleneck dim exactly hits the final crop size (aspect ratio preserved).
            If `r <= threshold`, fall back to the original drop behavior.
            Default None = disabled (original drop-on-too-small behavior).
        interpolation (optional): torchvision InterpolationMode, default LANCZOS.
            Used for the normal downsample path. The rescue path always uses LANCZOS.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        # If True, enforce resized >= final crop size (drop sample otherwise) and
        # enable scale_factor="adaptive". If False, behaves as a plain fixed-ratio resize.
        self.larger_than_final_crop_size = args["larger_than_final_crop_size"]
        # Near-miss rescue threshold; None disables the rescue path.
        self.tolerate_threshold = args.get("tolerate_smaller_shape_threshold", None)

    def _compute_adaptive_scale_factor(self, orig_w, orig_h, final_width, final_height):
        """Compute the continuous downsample factor for adaptive resize."""
        return min(orig_w / final_width, orig_h / final_height)

    def _compute_adaptive_resize_size(self, orig_w, orig_h, final_width, final_height):
        """Compute the smallest aspect-ratio-preserving resize that still covers
        the final crop size in both dimensions.

        This mirrors the video decode-time adaptive resize path: the binding
        dimension lands exactly on the crop target, and the other dimension
        remains >= the target. It is intentionally not restricted to integer or
        power-of-2 downsample factors.
        """
        scale_factor = self._compute_adaptive_scale_factor(orig_w, orig_h, final_width, final_height)
        if scale_factor <= 1.0:
            return orig_w, orig_h
        resized_w = max(final_width, int(math.ceil(orig_w / scale_factor)))
        resized_h = max(final_height, int(math.ceil(orig_h / scale_factor)))
        return resized_w, resized_h

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs fixed-scale or adaptive downsampling on images.

        When scale_factor is a number, the image is resized by dividing its
        dimensions by that factor.

        When scale_factor == "adaptive", resizes to the smallest
        aspect-ratio-preserving resolution that still covers the final crop size.
        Requires larger_than_final_crop_size=True and aspect_ratio in data_dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are downsampled
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"
        assert "scale_factor" in self.args, "Please specify scale_factor in args"

        scale_factor = self.args["scale_factor"]
        is_adaptive = isinstance(scale_factor, str) and scale_factor == "adaptive"

        if not is_adaptive:
            assert scale_factor > 0, f"scale_factor must be positive, got {scale_factor}"
            # Short-circuit: skip resize entirely when scale_factor == 1 (no-op)
            if scale_factor == 1:
                return data_dict

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            orig_w, orig_h = obtain_image_size(data_dict, [inp_key])

            if self.larger_than_final_crop_size:
                assert "aspect_ratio" in data_dict, "aspect_ratio is required when larger_than_final_crop_size is True"
                final_target_size = obtain_augmentation_size(data_dict, self.args)
                final_width, final_height = final_target_size

            if is_adaptive:
                assert self.larger_than_final_crop_size, (
                    "adaptive scale_factor requires larger_than_final_crop_size=True"
                )
                resized_w, resized_h = self._compute_adaptive_resize_size(orig_w, orig_h, final_width, final_height)
                if resized_w == orig_w and resized_h == orig_h and orig_w >= final_width and orig_h >= final_height:
                    # Image already large enough, no resize needed
                    if out_key != inp_key:
                        del data_dict[inp_key]
                    continue
                # If the image is too small, fall through to the rescue/drop logic below.
            else:
                # Calculate target size after fixed-factor downsampling.
                resized_w = int(orig_w / scale_factor)
                resized_h = int(orig_h / scale_factor)

            # Rescue near-miss samples via Lanczos upscale (when configured).
            # If the downsampled image is slightly below the final crop size and
            # tolerate_smaller_shape_threshold is set, scale just enough so the
            # bottleneck dim exactly hits the final crop size (aspect ratio preserved).
            use_lanczos_upscale = False
            if self.larger_than_final_crop_size and (resized_w < final_width or resized_h < final_height):
                shape_ratio = min(resized_h / final_height, resized_w / final_width)
                if self.tolerate_threshold is not None and shape_ratio > self.tolerate_threshold:
                    pre_w, pre_h = resized_w, resized_h
                    upscale = 1.0 / shape_ratio
                    resized_w = max(int(resized_w * upscale + 0.5), final_width)
                    resized_h = max(int(resized_h * upscale + 0.5), final_height)
                    use_lanczos_upscale = True
                    logging.warning(
                        f"ResizeScale: image {orig_w}x{orig_h} downsampled to "
                        f"{pre_w}x{pre_h} (< crop target {final_width}x{final_height}, "
                        f"shape_ratio={shape_ratio:.3f} > threshold={self.tolerate_threshold}), "
                        f"Lanczos-upscaling to {resized_w}x{resized_h}."
                    )
                else:
                    logging.warning(
                        f"ResizeScale: image {orig_w}x{orig_h} smaller than crop target "
                        f"{final_width}x{final_height}, skipping."
                    )
                    return None

            interp = (
                transforms_F.InterpolationMode.LANCZOS
                if use_lanczos_upscale
                else self.args.get("interpolation", transforms_F.InterpolationMode.LANCZOS)
            )
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=[resized_h, resized_w],
                interpolation=interp,
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]

        return data_dict
