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

import math
import random
from typing import Optional, Tuple

from PIL import Image

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.utils import log


class AddRandomAspectRatio(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        aspect_ratio_choice = random.choice(list(self.args["aspect_ratio_choices"]))
        data_dict["aspect_ratio"] = aspect_ratio_choice
        return data_dict


class InferAspectRatio(Augmentor):
    PREDEFINED_ASPECT_RATIOS = [
        ("16_9", 16 / 9),
        ("3_2", 3 / 2),
        ("4_3", 4 / 3),
        ("1_1", 1.0),
        ("3_4", 3 / 4),
        ("2_3", 2 / 3),
        ("9_16", 9 / 16),
    ]

    RETURN_VALUES = {
        "aspect_ratio_16_9": "16,9",
        "aspect_ratio_3_2": "3,2",
        "aspect_ratio_4_3": "4,3",
        "aspect_ratio_1_1": "1,1",
        "aspect_ratio_3_4": "3,4",
        "aspect_ratio_2_3": "2,3",
        "aspect_ratio_9_16": "9,16",
    }

    @staticmethod
    def _get_video_metadata(video: bytes) -> Tuple[int, int, float]:
        """Read video resolution and duration using decord (lazy import).

        Returns (width, height, duration_seconds).
        """
        import io

        from decord import VideoReader

        video_buffer = io.BytesIO(video)
        vr = VideoReader(video_buffer, num_threads=4)

        w, h = vr[0].shape[1], vr[0].shape[0]  # decord frame shape is (H, W, C)
        fps = vr.get_avg_fps()
        duration = len(vr) / fps if fps > 0 else 0.0
        return w, h, duration

    @staticmethod
    def _get_image_metadata(image: Image.Image) -> Tuple[int, int]:
        """Read image dimensions using PIL (header only, lazy import).

        Returns (width, height).
        """
        width, height = image.size
        return width, height

    @staticmethod
    def _classify_aspect_ratio(w: int, h: int) -> str:
        """Classify width/height into the nearest predefined aspect ratio using log-space distance.

        Returns a string like "aspect_ratio_16_9".
        """
        ratio = w / h
        log_ratio = math.log(ratio)
        best_name = InferAspectRatio.PREDEFINED_ASPECT_RATIOS[0][0]
        best_dist = float("inf")
        for name, ref_ratio in InferAspectRatio.PREDEFINED_ASPECT_RATIOS:
            dist = abs(log_ratio - math.log(ref_ratio))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return f"aspect_ratio_{best_name}"

    def __call__(self, data_dict: dict) -> dict | None:
        assert len(self.input_keys) == 1, "InferAspectRatio only supports one input key"
        required_key = self.input_keys[0]
        if required_key not in data_dict:
            # Sample is missing the required key (e.g. corrupted image skipped by decoder_handler).
            log.warning(
                f"[InferAspectRatio] Missing key '{required_key}' in sample {data_dict.get('__key__', '?')}. Skipping."
            )
            return None
        if "video" in self.input_keys:
            # Corrupt WebDataset samples can still have a "video" key whose bytes
            # are empty or not decodable. Return None so the wrapper skips it.
            try:
                w, h, _ = self._get_video_metadata(data_dict["video"])
            except Exception as e:
                log.warning(
                    "[InferAspectRatio] Failed to read video metadata. "
                    f"Skipping sample {data_dict.get('__key__', '?')} from {data_dict.get('__url__', '?')}. "
                    f"error={type(e).__name__}: {e}",
                    rank0_only=False,
                )
                return None
        elif "image" in self.input_keys:
            w, h = self._get_image_metadata(data_dict["image"])
        else:
            raise ValueError(f"Unsupported file type: {data_dict['__key__']}")

        data_dict["aspect_ratio"] = self.RETURN_VALUES[self._classify_aspect_ratio(w, h)]

        if data_dict["aspect_ratio"] not in self.args["aspect_ratio_choices"]:
            log.warning(
                f"[InferAspectRatio] Aspect ratio {data_dict['aspect_ratio']} not in {self.args['aspect_ratio_choices']}. Skipping sample. Consider bucketing your data!"
            )
            return None

        return data_dict
