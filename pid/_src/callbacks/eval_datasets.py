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
Image evaluation datasets for PixelDiT SR training callbacks.

This module provides dataset classes for loading and managing evaluation data.
All data is pre-loaded to CPU memory during initialization and transferred to GPU only when needed.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch

from pid._ext.imaginaire.utils import log


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def _list_pt_files(fix_batch_dir: str) -> list[str]:
    return sorted([f for f in os.listdir(fix_batch_dir) if f.endswith(".pt")], key=_natural_sort_key)


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.reshape(-1)[0].item()) if value.numel() else default
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else default
    return float(value)


def _image_tensor_to_uint8_bhwc(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got shape={tuple(tensor.shape)}")
    return ((tensor.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


class EvalDataset(ABC):
    """Base class for evaluation datasets.

    Subclasses should implement load_sample() to define how to load LQ and HQ data.
    All data is stored on CPU and transferred to GPU only during evaluation.
    """

    def __init__(self, name: str):
        """Initialize the evaluation dataset.

        Args:
            name: Name of the dataset (e.g., "YouHQ40")
        """
        self.name = name
        self.samples = []  # List of (name, lq_data, hq_data, caption) tuples

    @abstractmethod
    def load_all_samples(self, dp_rank: int, dp_world_size: int) -> None:
        """Load all samples for this DP rank to CPU memory.

        Args:
            dp_rank: Data parallel rank ID
            dp_world_size: Total number of DP ranks
        """
        pass

    def get_sample(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray, any, any, float]:
        """Get a pre-loaded sample by index.

        Returns:
            Tuple (name, lq_data, hq_data, caption, lq_latent, degrade_sigma).
            degrade_sigma is the per-sample noise level of the LQ latent
            (0.0 for clean samples, >0 for xt-step samples).
        """
        return self.samples[idx]

    def __len__(self) -> int:
        """Get number of samples assigned to this DP rank."""
        return len(self.samples)

    @staticmethod
    def natural_sort_key(s: str):
        """Natural sort key for filenames."""
        return _natural_sort_key(s)


class FixBatchImageDataset(EvalDataset):
    """Evaluation dataset from fix_batch .pt files (PixelDiT SR).

    Each .pt file contains a dict with:
    - "HQ_video_or_image": [1, 3, H, W] float32 in [-1, 1] (HQ ground truth)
    - "LQ_video_or_image": [1, 3, H_lq, W_lq] float32 in [-1, 1] (LQ input)
    - "LQ_latent": [1, C, H_z, W_z] pre-computed LQ latent
    - "degrade_sigma": degradation level associated with the latent
    - "caption": list of strings (used for text conditioning during inference)

    Converts tensors to numpy (1, H, W, C) uint8 for the standard eval interface.
    Samples are stored as 6-tuples: (name, lq_np, hq_np, caption, lq_latent, degrade_sigma).
    """

    def __init__(self, fix_batch_dir: str, name: str = "FixBatch"):
        super().__init__(name=name)
        self.fix_batch_dir = fix_batch_dir

    def load_all_samples(self, dp_rank: int, dp_world_size: int) -> None:
        """Load fix_batch .pt files for this DP rank to CPU memory."""
        pt_files = _list_pt_files(self.fix_batch_dir)

        # Distribute using modulo arithmetic
        my_files = [pt_files[i] for i in range(len(pt_files)) if i % dp_world_size == dp_rank]

        log.info(
            f"{self.name}: DP rank {dp_rank} loading {len(my_files)}/{len(pt_files)} fix_batch files to CPU",
            rank0_only=False,
        )

        self.samples = []
        for fname in my_files:
            try:
                fpath = os.path.join(self.fix_batch_dir, fname)
                from pid._src.inference.inference_utils import load_fix_batch

                data = load_fix_batch(fpath)

                hq_tensor = data["HQ_video_or_image"]  # [1, 3, H, W]
                hq_np = _image_tensor_to_uint8_bhwc(hq_tensor)  # (1, H, W, C)

                lq_tensor = data["LQ_video_or_image"]  # [1, 3, H_lq, W_lq]
                lq_np = _image_tensor_to_uint8_bhwc(lq_tensor)  # (1, H_lq, W_lq, C)

                sample_name = os.path.splitext(fname)[0]
                caption = data["caption"]  # list of strings or None

                lq_latent = data["LQ_latent"]  # tensor, [1, C, H, W]

                # degrade_sigma: per-sample noise level of the LQ latent (0 for clean,
                # >0 for xt-step samples generated by create_dataset.py --save_xt_steps).
                degrade_sigma = _to_float(data["degrade_sigma"])

                self.samples.append((sample_name, lq_np, hq_np, caption, lq_latent, degrade_sigma))

            except Exception as e:
                log.error(f"{self.name}: Failed to load {fname}: {e}", rank0_only=False)
                continue

        log.info(f"{self.name}: DP rank {dp_rank} loaded {len(self.samples)} samples to CPU", rank0_only=False)
