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

"""Metrics module for evaluation."""

from .base import BaseMetric, MetricRegistry
from .image_metrics import (
    CLIPIQA,
    LPIPS,
    MANIQA,
    MUSIQ,
    NIQE,
    PSNR,
    SSIM,
    CLIPIQAPlus,
    LQColorDE2000,
    QAlign,
    QAlignQualityNative,
    UniPerceptIAA,
    UniPerceptIQA,
    UniPerceptISTA,
)
from .video_metrics import DOVER

__all__ = [
    "BaseMetric",
    "MetricRegistry",
    "PSNR",
    "SSIM",
    "LPIPS",
    "LQColorDE2000",
    "NIQE",
    "MUSIQ",
    "CLIPIQA",
    "CLIPIQAPlus",
    "MANIQA",
    "QAlign",
    "QAlignQualityNative",
    "UniPerceptIAA",
    "UniPerceptIQA",
    "UniPerceptISTA",
    "DOVER",
]
