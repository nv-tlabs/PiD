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
Multi-resolution aspect-ratio augmentor for PixelDiT T2I training.

Goal: pick the per-sample (resolution, AR) bucket at the dataloader level so
that a single training run can mix 2K + 4K source images without upsampling.
Each sample's native (W, H) determines the largest grid resolution it can fit;
the AspectRatioDataLoader buckets samples by a composite key f"L{level}_{ar}"
so every batch is shape-uniform.

Pipeline:
  1. InferMultiResolutionAspectRatio  -- classify AR, find largest fitting
     level, write composite bucket key + target_crop_size into data_dict.
  2. ResizeScaleByTargetSize          -- adaptive continuous downsample to
     the smallest size that still covers target_crop_size. Reads target from
     data_dict directly,
     bypassing obtain_augmentation_size (which prefers URL-meta AR and
     cannot encode our composite key).
  3. CenterCropByTargetSize           -- center crop to target_crop_size.

Grid levels are filtered by the requested max-resolution preset. Use
max_resolution selects the largest enabled area level and per-AR crop ceiling.
Use 3072 for 2K..3K, 3840 to avoid the 4096x4096 square bucket, or 4096 for
the full legacy grid.
"""

import logging as _logging
import math
from typing import List, Optional, Tuple

import torchvision.transforms.functional as transforms_F
from PIL import Image

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.datasets.webdataset.augmentors.image.cropping import _center_crop_tensor_or_ndarray
from pid._ext.imaginaire.datasets.webdataset.augmentors.image.misc import obtain_image_size
from pid._ext.imaginaire.utils import log
from pid._src.datasets.utils import IMAGE_RES_SIZE_INFO

# -----------------------------------------------------------------------------
# Grid construction
# -----------------------------------------------------------------------------

# Resolution levels keyed off the 1:1 max-side. For non-1:1 ARs, (W, H) is
# derived via area-scale (W*H ≈ L^2) with W/H matching the AR exactly, then
# snapped to multiples of 64 so HQ // sr_scale stays a multiple of 16 (the
# `simple_downsample` division_factor) for sr_scale up to 4 — keeps HQ:LQ at
# exact 4× across every (level, AR) bucket. After the formula, (W, H) is
# clipped per-AR to the requested crop ceiling from IMAGE_RES_SIZE_INFO.

_MULTI_RES_LEVELS: List[int] = [2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
_MULTI_RES_MAX_RESOLUTIONS: tuple[str, ...] = ("3072", "3840", "4096")
_ALIGN: int = 64

# (ar_string, w_ratio, h_ratio). Ordering kept consistent with InferAspectRatio.
_AR_RATIOS: List[Tuple[str, int, int]] = [
    ("1,1", 1, 1),
    ("4,3", 4, 3),
    ("3,4", 3, 4),
    ("16,9", 16, 9),
    ("9,16", 9, 16),
]


def _round_to_multiple(x: float, m: int) -> int:
    return int(round(x / m)) * m


def _normalize_max_resolution(max_resolution: str) -> str:
    max_resolution = str(max_resolution)
    if max_resolution not in _MULTI_RES_MAX_RESOLUTIONS:
        raise ValueError(
            f"Unsupported multi-resolution max_resolution={max_resolution}. "
            f"Expected one of {_MULTI_RES_MAX_RESOLUTIONS}."
        )
    return max_resolution


def _compute_target(
    level: int,
    ar: str,
    w_r: int,
    h_r: int,
    ar_max_wh: dict[str, Tuple[int, int]],
) -> Tuple[int, int]:
    """Compute (W, H) for a given level and AR via area-scale formula, snap-64,
    then clip independently by the selected per-AR OOM ceiling.
    """
    r = w_r / h_r
    w = level * math.sqrt(r)
    h = level / math.sqrt(r)
    w = _round_to_multiple(w, _ALIGN)
    h = _round_to_multiple(h, _ALIGN)
    w_max, h_max = ar_max_wh[ar]
    w = min(w, w_max)
    h = min(h, h_max)
    return (w, h)


def _build_grid(max_resolution: str) -> dict:
    """Returns dict[ar_string -> list[(level, W, H)]] sorted descending by level
    (so largest-fit search walks from biggest to smallest).
    """
    max_resolution = _normalize_max_resolution(max_resolution)
    max_level = int(max_resolution)
    levels = [level for level in _MULTI_RES_LEVELS if level <= max_level]
    ar_max_wh = IMAGE_RES_SIZE_INFO[max_resolution]

    grid = {}
    for ar, w_r, h_r in _AR_RATIOS:
        entries = []
        for L in levels:
            w, h = _compute_target(L, ar, w_r, h_r, ar_max_wh)
            entries.append((L, w, h))
        entries.sort(key=lambda e: e[0], reverse=True)
        grid[ar] = entries
    return grid


# Module-level constants. Each grid is dict[ar -> list[(level, W, H)]], sorted
# descending by level. Keep MULTI_RES_GRID as the legacy 4096-grid alias.
MULTI_RES_GRID_BY_MAX_RESOLUTION: dict[str, dict] = {
    max_resolution: _build_grid(max_resolution) for max_resolution in _MULTI_RES_MAX_RESOLUTIONS
}
MULTI_RES_GRID: dict = MULTI_RES_GRID_BY_MAX_RESOLUTION["4096"]


# -----------------------------------------------------------------------------
# AR classification (mirrors InferAspectRatio at add_aspect_ratio.py:80)
# -----------------------------------------------------------------------------

_PREDEFINED_AR_TO_RATIO: List[Tuple[str, float]] = [
    ("16,9", 16 / 9),
    ("4,3", 4 / 3),
    ("1,1", 1.0),
    ("3,4", 3 / 4),
    ("9,16", 9 / 16),
]


def _classify_aspect_ratio(w: int, h: int) -> str:
    """Classify (w, h) into the closest predefined AR via log-space distance."""
    log_ratio = math.log(w / h)
    best_ar = _PREDEFINED_AR_TO_RATIO[0][0]
    best_dist = float("inf")
    for ar, ref_ratio in _PREDEFINED_AR_TO_RATIO:
        dist = abs(log_ratio - math.log(ref_ratio))
        if dist < best_dist:
            best_dist = dist
            best_ar = ar
    return best_ar


# -----------------------------------------------------------------------------
# Augmentors
# -----------------------------------------------------------------------------


class InferMultiResolutionAspectRatio(Augmentor):
    """Classify the sample's AR and pick the largest grid level that fits.

    Writes:
      data_dict["aspect_ratio"]    -> composite bucket key f"L{level}_{ar}"
      data_dict["target_crop_size"] -> (W, H) for downstream resize/crop

    Returns None (drops the sample) if even the smallest grid level (L=2048)
    is larger than the native image.

    args:
      max_resolution: "3072" for 2K..3K, "3840" avoids the 4096x4096 bucket,
        and "4096" keeps the legacy full grid.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 1, "InferMultiResolutionAspectRatio expects a single input key"
        assert input_keys[0] == "image", "Only image input is supported"
        self.max_resolution = _normalize_max_resolution(
            (args or {}).get("max_resolution", "4096"),
        )
        self.multi_res_grid = MULTI_RES_GRID_BY_MAX_RESOLUTION[self.max_resolution]

    def __call__(self, data_dict: dict) -> Optional[dict]:
        key = self.input_keys[0]
        if key not in data_dict:
            log.warning(
                f"[InferMultiResolutionAspectRatio] Missing key '{key}' in sample "
                f"{data_dict.get('__key__', '?')}. Skipping."
            )
            return None

        img = data_dict[key]
        if not isinstance(img, Image.Image):
            log.warning(f"[InferMultiResolutionAspectRatio] Expected PIL.Image, got {type(img)}. Skipping.")
            return None

        w_n, h_n = img.size
        ar = _classify_aspect_ratio(w_n, h_n)
        entries = self.multi_res_grid[ar]  # sorted desc by level

        chosen = None
        for level, w_t, h_t in entries:
            if w_n >= w_t and h_n >= h_t:
                chosen = (level, w_t, h_t)
                break

        if chosen is None:
            # Native image smaller than the smallest grid bucket -> skip.
            return None

        level, w_t, h_t = chosen
        data_dict["aspect_ratio"] = f"L{level}_{ar}"
        data_dict["target_crop_size"] = (w_t, h_t)
        return data_dict


class ResizeScaleByTargetSize(Augmentor):
    """Adaptive continuous downsample so the image is just above the target.

    Cloned from imaginaire ResizeScale (resize.py:194) but reads
    data_dict["target_crop_size"] directly. We bypass obtain_augmentation_size
    because that helper preferentially reads URL-meta AR, which would lose our
    composite "L{level}_{ar}" key.

    args:
        scale_factor: int or "adaptive". For multi-res training use "adaptive".
        interpolation: PIL/torchvision interpolation mode (default LANCZOS for
            PIL inputs).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert args is not None and "scale_factor" in args, "scale_factor required"

    @staticmethod
    def _compute_adaptive_scale_factor(orig_w: int, orig_h: int, final_w: int, final_h: int) -> float:
        return min(orig_w / final_w, orig_h / final_h)

    def _compute_adaptive_resize_size(self, orig_w: int, orig_h: int, final_w: int, final_h: int) -> Tuple[int, int]:
        """Mirror ResizeScale/video adaptive resize: the limiting dimension lands
        on the crop target and the other dimension remains >= target.
        """
        scale_factor = self._compute_adaptive_scale_factor(orig_w, orig_h, final_w, final_h)
        if scale_factor <= 1.0:
            return orig_w, orig_h
        resized_w = max(final_w, int(math.ceil(orig_w / scale_factor)))
        resized_h = max(final_h, int(math.ceil(orig_h / scale_factor)))
        return resized_w, resized_h

    def __call__(self, data_dict: dict) -> Optional[dict]:
        if self.output_keys is None:
            self.output_keys = self.input_keys

        if "target_crop_size" not in data_dict:
            log.warning(
                f"[ResizeScaleByTargetSize] Missing target_crop_size in sample "
                f"{data_dict.get('__key__', '?')}. Skipping."
            )
            return None

        final_w, final_h = data_dict["target_crop_size"]
        scale_factor = self.args["scale_factor"]
        is_adaptive = isinstance(scale_factor, str) and scale_factor == "adaptive"
        if not is_adaptive:
            assert scale_factor > 0, f"scale_factor must be positive, got {scale_factor}"
            if scale_factor == 1:
                return data_dict

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            orig_w, orig_h = obtain_image_size(data_dict, [inp_key])

            if is_adaptive:
                resized_w, resized_h = self._compute_adaptive_resize_size(orig_w, orig_h, final_w, final_h)
                if resized_w == orig_w and resized_h == orig_h and orig_w >= final_w and orig_h >= final_h:
                    if out_key != inp_key:
                        del data_dict[inp_key]
                    continue
            else:
                sf = scale_factor
                resized_w = int(orig_w / sf)
                resized_h = int(orig_h / sf)

            if resized_w < final_w or resized_h < final_h:
                _logging.warning(
                    f"ResizeScaleByTargetSize: image {orig_w}x{orig_h} smaller than crop target "
                    f"{final_w}x{final_h}, skipping."
                )
                return None

            interp = self.args.get("interpolation", transforms_F.InterpolationMode.BICUBIC)
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=[resized_h, resized_w],
                interpolation=interp,
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]

        return data_dict


class CenterCropByTargetSize(Augmentor):
    """Center crop to data_dict["target_crop_size"] (W, H).

    Cloned from imaginaire CenterCrop (cropping.py:79) but reads target directly
    from data_dict, bypassing obtain_augmentation_size (URL-meta-AR override).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.skip_if_smaller = (args or {}).get("skip_if_smaller", True)

    def __call__(self, data_dict: dict) -> Optional[dict]:
        if "target_crop_size" not in data_dict:
            log.warning(
                f"[CenterCropByTargetSize] Missing target_crop_size in sample "
                f"{data_dict.get('__key__', '?')}. Skipping."
            )
            return None

        width, height = data_dict["target_crop_size"]
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

        if orig_w < width or orig_h < height:
            if self.skip_if_smaller:
                _logging.warning(
                    f"CenterCropByTargetSize: image {orig_w}x{orig_h} smaller than crop "
                    f"{width}x{height}. Skipping sample {data_dict.get('__key__', '?')}."
                )
                return None

        for key in self.input_keys:
            data_dict[key] = _center_crop_tensor_or_ndarray(data_dict[key], height, width)

        crop_x0 = (orig_w - width) // 2
        crop_y0 = (orig_h - height) // 2
        if "aug_params" not in data_dict:
            data_dict["aug_params"] = {}
        data_dict["aug_params"]["cropping"] = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }
        return data_dict
