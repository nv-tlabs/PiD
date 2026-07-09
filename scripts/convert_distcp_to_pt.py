#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Convert a training checkpoint saved in PyTorch Distributed Checkpoint (DCP / distcp)
# format into a single consolidated .pth file for inference or finetune resume.
#
# Background:
#   Training with ckpt_type=dcp saves shards under {iter_dir}/model/ (e.g. __0_0.distcp).
#   Many inference scripts and finetune configs expect a single torch.save() file instead.
#   This script wraps torch.distributed.checkpoint.format_utils.dcp_to_torch_save.
#
# Input (positional, first argument):
#   - Path to the local distcp directory, e.g.
#       /path/to/checkpoints/iter_000100000/model
#     The script expects the .../model shard directory.
#
# Output (written under the second positional argument, output_dir):
#   model.pth              — full state dict (net.*, net_ema.*, optimizer keys, etc.)
#   model_ema_fp32.pth     — EMA weights only, fp32, keys renamed net_ema.* -> net.*  (--ema)
#   model_ema_bf16.pth     — same EMA weights cast to bf16 where applicable           (--ema)
#
# Usage:
#   # Basic conversion (local DCP -> single .pth)
#   python scripts/convert_distcp_to_pt.py \
#       /path/to/iter_000100000/model \
#       /path/to/output_dir
#
#   # Also export EMA-only checkpoints for inference (fp32 + bf16)
#   python scripts/convert_distcp_to_pt.py \
#       /path/to/iter_000100000/model \
#       /path/to/output_dir \
#       --ema
#
#   # Only write model.pth, skip EMA extraction (default without --ema)
#   python scripts/convert_distcp_to_pt.py \
#       /path/to/iter_000100000/model \
#       /path/to/output_dir \
#       --keep-original
#
# Notes:
#   - Re-running with the same output_dir overwrites existing output files.
#   - After conversion, rename model.pth as needed for your config, e.g.
#       mv output_dir/model.pth checkpoints/vixeldit/my_experiment_iter_100000.pth
#
# Example (from vixeldit finetune config):
#   python scripts/convert_distcp_to_pt.py \
#       imaginaire4/.../pixeldit_stage3_finetune_2048px_shift6_gcp/checkpoints/iter_000100000/model \
#       checkpoints/vixeldit/

"""Convert DCP (distcp) checkpoints to consolidated .pth files."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the local distcp directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to save the converted checkpoints.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Export EMA weights.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original DCP state dict without EMA extraction or dtype conversion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pt_path = args.output_dir / "model.pth"
    pt_path.unlink(missing_ok=True)
    pt_ema_fp32_path = args.output_dir / "model_ema_fp32.pth"
    pt_ema_fp32_path.unlink(missing_ok=True)
    pt_ema_bf16_path = args.output_dir / "model_ema_bf16.pth"
    pt_ema_bf16_path.unlink(missing_ok=True)

    distcp_dir = args.input_dir

    # Convert distributed checkpoint to torch single checkpoint
    dcp_to_torch_save(distcp_dir, pt_path)
    print(f"Converted '{distcp_dir}' to '{pt_path}'")

    if args.keep_original:
        return

    if not args.ema:
        return

    # Drop Reg keys and save EMA weights only in fp32 precision
    state_dict: dict[str, Any] = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict_ema_fp32: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            key = key.replace("net_ema.", "net.")
            state_dict_ema_fp32[key] = value
    if not state_dict_ema_fp32:
        raise ValueError("Model doesn't contain EMA weights")
    torch.save(state_dict_ema_fp32, pt_ema_fp32_path)
    print(f"Saved EMA fp32 weights from '{pt_path}' to '{pt_ema_fp32_path}'")

    # Save EMA weights only in bf16 precision
    state_dict_ema_bf16: dict[str, Any] = {}
    for key, value in state_dict_ema_fp32.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.bfloat16()
        state_dict_ema_bf16[key] = value
    torch.save(state_dict_ema_bf16, pt_ema_bf16_path)
    print(f"fp32 -> bf16: '{pt_ema_fp32_path}' to '{pt_ema_bf16_path}'")


if __name__ == "__main__":
    main()
