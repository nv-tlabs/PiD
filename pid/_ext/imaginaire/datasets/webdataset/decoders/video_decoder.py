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

import io
import re

# decord is intentionally NOT imported here at module level.
# Importing decord before torchvision corrupts the glibc heap allocator
# (decord 0.6.0 bug), so it is lazy-imported inside each decoder function.
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()

VIDEO_DECODER_OPTIONS = {}


def video_decoder_register(key):
    def decorator(func):
        VIDEO_DECODER_OPTIONS[key] = func
        return func

    return decorator


@video_decoder_register("video_decoder_metadata")
def video_decoder_metadata(num_threads, **kwargs):
    """
    Video decoder using the video's native fps
    """

    def video_decoder(key: str, data: bytes):
        import decord  # lazy import: must come after torchvision is loaded

        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None
        video_buffer = io.BytesIO(data)
        reader = decord.VideoReader(video_buffer, num_threads=num_threads)
        num_frames = len(reader)
        video_fps = int(np.round(reader.get_avg_fps()))
        length_in_s = float(num_frames) / float(video_fps)
        bitrate = video_buffer.getbuffer().nbytes * 8 / length_in_s
        video_frames = reader.get_batch([0]).asnumpy()
        video_frames = torch.from_numpy(video_frames).permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_frames, {"fps": video_fps, "num_frames": num_frames, "bitrate": bitrate}

    return video_decoder


@video_decoder_register("video_naive_bytes")
def video_naive_bytes(*args, **kwargs):
    """
    do nothing, just return the video bytes
    """
    del args, kwargs

    def video_decoder(
        key: str,
        data: bytes,
    ):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in _VIDEO_EXTENSIONS:
            return None

        return data

    return video_decoder


def construct_video_decoder(
    video_decoder_name: str = "video_naive_bytes",
    sequence_length: int = 34,
    chunk_size: int = 0,
    use_fps_control: bool = False,
    min_fps_thres: int = 4,
    max_fps_thres: int = 24,
    sampling_reweighting: bool = False,
    sampling_reweighting_factor: int = 1,
    num_threads=4,
    limit_fps_range: bool = False,
    # if true, video decoder will additionally save the raw video (alongside with processed frames) to the data_dict
    # set to true for inference/debugging
    save_raw: bool = False,
):
    return VIDEO_DECODER_OPTIONS[video_decoder_name](
        sequence_length=sequence_length,
        chunk_size=chunk_size,
        use_fps_control=use_fps_control,
        min_fps_thres=min_fps_thres,
        max_fps_thres=max_fps_thres,
        sampling_reweighting=sampling_reweighting,
        sampling_reweighting_factor=sampling_reweighting_factor,
        num_threads=num_threads,
        limit_fps_range=limit_fps_range,
        save_raw=save_raw,
    )


def construct_video_decoder_metadata(
    num_threads=4,
):
    return VIDEO_DECODER_OPTIONS["video_decoder_metadata"](num_threads=num_threads)
