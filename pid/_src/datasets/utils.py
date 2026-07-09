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


import re
from typing import List, Tuple

IMAGE_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "1024": {
        "1,1": (1024, 1024),
        "4,3": (1152, 896),
        "3,4": (896, 1152),
        "16,9": (1344, 768),
        "9,16": (768, 1344),
        "3,2": (1344, 896),
        "2,3": (896, 1344),
    },
    "2048": {
        "1,1": (2048, 2048),
        "4,3": (2304, 1728),
        "3,4": (1728, 2304),
        "16,9": (2688, 1536),
        "9,16": (1536, 2688),
        "3,2": (2688, 1792),
        "2,3": (1792, 2688),
    },
    "3072": {
        "1,1": (3072, 3072),
        "4,3": (3520, 2688),
        "3,4": (2688, 3520),
        "16,9": (4096, 2304),
        "9,16": (2304, 4096),
        "3,2": (3776, 2496),
        "2,3": (2496, 3776),
    },
    "3840": {
        "1,1": (3584, 3584),
        "4,3": (4096, 2816),
        "3,4": (2816, 4096),
        "16,9": (4096, 2176),
        "9,16": (2176, 4096),
        "3,2": (4096, 2688),
        "2,3": (2688, 4096),
    },
    "4096": {
        "1,1": (4096, 4096),
        "4,3": (4096, 3072),
        "3,4": (3072, 4096),
        "16,9": (4096, 2304),
        "9,16": (2304, 4096),
        "3,2": (4096, 2688),
        "2,3": (2688, 4096),
    },
}

VIDEO_RES_SIZE_INFO = IMAGE_RES_SIZE_INFO


def get_aspect_ratios_from_wdinfos(wdinfos: list[str]) -> list[str]:
    aspect_ratios = []
    for wdinfo in wdinfos:
        aspect_ratio_match = re.search(r"aspect_ratio_(\d+_\d+)", wdinfo)
        aspect_ratios.append(aspect_ratio_match.group(1))

    return aspect_ratios


def get_wdinfos_w_aspect_ratio(wdinfos: list[str]) -> List[Tuple[str, str]]:
    aspect_ratios = get_aspect_ratios_from_wdinfos(wdinfos)

    # return a list of (wdinfo_path, aspect_ratio) pairs
    return [(wdinfo, aspect_ratio.replace("_", ",")) for wdinfo, aspect_ratio in zip(wdinfos, aspect_ratios)]
