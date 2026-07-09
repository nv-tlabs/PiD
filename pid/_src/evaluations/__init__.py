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
Evaluation framework for LinearVSR video super-resolution models.

This module provides tools for evaluating VSR models on various datasets
with multiple metrics (PSNR, SSIM, LPIPS, NIQE, MUSIQ, CLIPIQA, DOVER).
"""

from pid._src.evaluations.metrics import CLIPIQA, DOVER, LPIPS, MUSIQ, NIQE, PSNR, SSIM, MetricRegistry

__all__ = ["PSNR", "SSIM", "LPIPS", "NIQE", "MUSIQ", "CLIPIQA", "DOVER", "MetricRegistry"]
