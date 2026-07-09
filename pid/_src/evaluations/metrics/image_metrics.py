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

"""Image quality metrics.

Pyiqa-backed:   PSNR, SSIM, LPIPS, NIQE, MUSIQ, CLIPIQA, MANIQA, LIQE, QualiCLIP, QAlign.
mPLUG-Owl2 NR:  DeQAScore (vendored DeQA-Score adaptation, ~[1, 5] MOS scale).
VLM-backed:     VisualQuality-R1 (CoT, 1-5), Qwen3-VL (JSON, 0-10).

The VLM metrics downscale inputs to max-side 512 (aspect preserved) to cap vision-token
cost on 2K SR outputs. Transformers backend only for now — vLLM can be added later.
"""

import os

# vLLM env vars — set at module top so they land before any lazy `from vllm import`
# in this file (or transitively in consumers), and so vLLM's EngineCore spawn
# children inherit them via os.environ. Uses setdefault so users can still
# override on the command line.
#   VLLM_ATTENTION_BACKEND=FLASH_ATTN : on Blackwell (compute cap 10.x) + torch 2.7
#     + vllm 0.10.x, the default FlashInfer decode path hits a
#     trtllm_paged_attention_decode arg-type mismatch; FLASH_ATTN avoids it.
#   VLLM_NO_USAGE_STATS=1 / DO_NOT_TRACK=1 : disable vllm/usage/usage_lib.py's
#     background stats thread, which calls `cpuinfo.get_cpu_info()` → `lscpu -J`
#     and JSON-parses the output. On aarch64 images `lscpu -J` returns empty /
#     non-JSON so the thread spams a JSONDecodeError traceback per EngineCore.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import json  # noqa: E402
import re  # noqa: E402
from typing import List, Optional, Union  # noqa: E402

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from torchmetrics.image import PeakSignalNoiseRatio  # noqa: E402

from pid._src.evaluations.metrics.base import BaseMetric, MetricRegistry  # noqa: E402

try:
    import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


def _batch_to_tensor(frames: Union[np.ndarray, th.Tensor], device: str, to_chw: bool = False) -> th.Tensor:
    """
    Convert batch of frames to GPU tensor.

    Args:
        frames: numpy array of shape (T, H, W, C) or (T, C, H, W), or torch tensor already on device
        device: target device
        to_chw: if True, transpose from HWC to CHW format

    Returns:
        Tensor of shape (T, C, H, W) on device
    """
    # If already a tensor on the correct device, return as-is (avoid redundant transfer)
    if isinstance(frames, th.Tensor):
        if str(frames.device) == device:
            # Already on the correct device, just ensure correct shape
            if to_chw and frames.ndim == 4 and frames.shape[3] in [1, 3]:
                # (T, H, W, C) -> (T, C, H, W) - unlikely but handle it
                frames = frames.permute(0, 3, 1, 2)
            return frames
        else:
            # On different device, move it
            return frames.to(device)

    # Convert from numpy
    if to_chw and frames.ndim == 4 and frames.shape[3] in [1, 3]:
        # (T, H, W, C) -> (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
    return th.from_numpy(frames.copy()).float().to(device)


def _to_numpy_hwc(img: Union[np.ndarray, th.Tensor]) -> np.ndarray:
    """
    Convert image to numpy array in HWC format with uint8 dtype.

    Args:
        img: Image tensor or array.
             For torch: expects (C, H, W) in range [-1, 1] or [0, 1]
             For numpy: expects (H, W, C) in range [0, 255] or [0, 1]

    Returns:
        Numpy array in (H, W, C) format with uint8 dtype
    """
    if isinstance(img, th.Tensor):
        img = img.detach().cpu()
        # Handle different tensor formats
        if img.ndim == 4:  # (B, C, H, W) - take first batch
            img = img[0]
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = img.permute(1, 2, 0)
        img = img.numpy()

    # Normalize to [0, 255] if needed
    if img.max() <= 1.0:
        img = img * 255.0
    elif img.min() < 0:  # Range [-1, 1]
        img = (img + 1) * 127.5

    return img.clip(0, 255).astype(np.uint8)


def _to_torch_chw(img: Union[np.ndarray, th.Tensor], device: str = "cuda") -> th.Tensor:
    """
    Convert image to torch tensor in CHW format with float32 dtype in [-1, 1].

    Args:
        img: Image tensor or array
        device: Target device

    Returns:
        Torch tensor in (C, H, W) format with float32 dtype in [-1, 1]
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] in [1, 3]:  # (H, W, C)
            img = np.transpose(img, (2, 0, 1))
        img = th.from_numpy(img).float()
    else:
        img = img.float()
        if img.ndim == 3 and img.shape[2] in [1, 3]:  # (H, W, C)
            img = img.permute(2, 0, 1)

    # Normalize to [-1, 1]
    if img.max() > 1.0:
        img = img / 127.5 - 1.0

    return img.to(device)


def _resize_for_vlm(img_hwc: np.ndarray, max_side: int = 512):
    """HWC uint8/float ndarray -> PIL.Image downscaled so max(H, W) <= max_side.

    2K SR outputs would otherwise blow up VLM vision-token counts (~20K tokens at
    2K vs ~1.3K at 512). Passes through untouched when already small enough.
    Aspect ratio is preserved (VLM processors use it for spatial reasoning).
    """
    from PIL import Image as PILImage

    if img_hwc.dtype != np.uint8:
        arr = img_hwc
        if arr.max() <= 1.0:
            arr = arr * 255.0
        elif arr.min() < 0:
            arr = (arr + 1) * 127.5
        img_hwc = arr.clip(0, 255).astype(np.uint8)

    pil = PILImage.fromarray(img_hwc)
    w, h = pil.size
    m = max(h, w)
    if m > max_side:
        scale = max_side / m
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS)
    return pil


@MetricRegistry.register("psnr")
class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric using torchmetrics."""

    def __init__(self, device: str = "cuda", data_range: float = 255.0):
        """
        Initialize PSNR metric.

        Args:
            device: Device to run computation on
            data_range: Maximum value of the data range
        """
        super().__init__(name="PSNR", device=device)
        self.data_range = data_range
        self._metric = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        """
        Compute PSNR between prediction and target.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            PSNR value in dB
        """
        # Convert to torch tensors on GPU
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        if isinstance(target, np.ndarray):
            target_th = th.from_numpy(target).float().to(self.device)
        else:
            target_th = target.float().to(self.device)

        # Ensure shape is (C, H, W) for torchmetrics
        if pred_th.ndim == 3 and pred_th.shape[2] in [1, 3]:  # (H, W, C)
            pred_th = pred_th.permute(2, 0, 1)  # -> (C, H, W)
        if target_th.ndim == 3 and target_th.shape[2] in [1, 3]:  # (H, W, C)
            target_th = target_th.permute(2, 0, 1)  # -> (C, H, W)

        # Add batch dimension if needed
        if pred_th.ndim == 3:
            pred_th = pred_th.unsqueeze(0)  # -> (1, C, H, W)
        if target_th.ndim == 3:
            target_th = target_th.unsqueeze(0)  # -> (1, C, H, W)

        with th.no_grad():
            psnr_value = self._metric(pred_th, target_th)

        return float(psnr_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute PSNR for a batch of frames in parallel on GPU.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames, shape (T, H, W, C)
            batch_size: Number of frames per batch. None means all frames at once.

        Returns:
            List of PSNR values for each frame
        """
        num_frames = pred.shape[0]

        # If no batch_size specified, process all at once
        if batch_size is None or batch_size >= num_frames:
            # Convert to (T, C, H, W)
            pred_th = _batch_to_tensor(pred, self.device, to_chw=True)
            target_th = _batch_to_tensor(target, self.device, to_chw=True)

            with th.no_grad():
                # Compute PSNR for each frame individually using torchmetrics
                psnr_values = []
                for i in range(pred_th.shape[0]):
                    psnr_val = self._metric(pred_th[i : i + 1], target_th[i : i + 1])
                    psnr_values.append(float(psnr_val.item()))

            return psnr_values

        # Process in batches
        all_psnr = []
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_th = _batch_to_tensor(pred[i:end_idx], self.device, to_chw=True)
            target_th = _batch_to_tensor(target[i:end_idx], self.device, to_chw=True)

            with th.no_grad():
                # Compute PSNR for each frame in the batch
                for j in range(pred_th.shape[0]):
                    psnr_val = self._metric(pred_th[j : j + 1], target_th[j : j + 1])
                    all_psnr.append(float(psnr_val.item()))

        return all_psnr


@MetricRegistry.register("ssim")
class SSIM(BaseMetric):
    """Structural Similarity Index Measure using fused-ssim for fast computation."""

    def __init__(
        self,
        device: str = "cuda",
        data_range: float = 255.0,
        channel_axis: int = 2,
        win_size: int = 11,
    ):
        """
        Initialize SSIM metric.

        Args:
            device: Device to run computation on
            data_range: Maximum value of the data range (default: 255.0)
            channel_axis: Axis for color channels (deprecated, kept for API compatibility)
            win_size: Size of the sliding window (deprecated for fused-ssim, kept for API compatibility)

        Note:
            fused-ssim uses a fixed 11x11 window size and expects inputs in [0, 1] range.
            This implementation will automatically normalize inputs based on data_range.
        """
        super().__init__(name="SSIM", device=device)
        self.data_range = data_range
        self.channel_axis = channel_axis
        self.win_size = win_size

        if not FUSED_SSIM_AVAILABLE:
            raise ImportError(
                "fused-ssim is not available. Install it with: "
                "pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation"
            )

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        """
        Compute SSIM between prediction and target using fused-ssim.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            SSIM value (0 to 1)
        """
        # Convert to torch tensors on GPU in CHW format
        if isinstance(pred, np.ndarray):
            # Assume HWC format from numpy
            if pred.ndim == 3 and pred.shape[2] in [1, 3]:
                pred = np.transpose(pred, (2, 0, 1))
            pred_th = th.from_numpy(pred).float().unsqueeze(0).to(self.device)
        else:
            pred_th = pred.float()
            if pred_th.ndim == 3:
                pred_th = pred_th.unsqueeze(0)
            pred_th = pred_th.to(self.device)

        if isinstance(target, np.ndarray):
            # Assume HWC format from numpy
            if target.ndim == 3 and target.shape[2] in [1, 3]:
                target = np.transpose(target, (2, 0, 1))
            target_th = th.from_numpy(target).float().unsqueeze(0).to(self.device)
        else:
            target_th = target.float()
            if target_th.ndim == 3:
                target_th = target_th.unsqueeze(0)
            target_th = target_th.to(self.device)

        # Normalize to [0, 1] range for fused-ssim
        # Intelligently detect if normalization is needed
        # Check data type and actual range to determine if input is in [0, 255] or [0, 1]
        if pred_th.dtype == th.uint8 or pred_th.max() > 1.0:
            # Data is likely in [0, 255] range
            pred_th = pred_th / self.data_range
        if target_th.dtype == th.uint8 or target_th.max() > 1.0:
            # Data is likely in [0, 255] range
            target_th = target_th / self.data_range

        # Compute SSIM using fused-ssim (no gradient needed for evaluation)
        with th.no_grad():
            ssim_value = fused_ssim.fused_ssim(pred_th, target_th, padding="same", train=False)

        return float(ssim_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute SSIM for a batch of frames in parallel on GPU using fused-ssim.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames, shape (T, H, W, C)
            batch_size: Number of frames per batch. None means all frames at once.

        Returns:
            List of SSIM values for each frame
        """
        num_frames = pred.shape[0]

        # If no batch_size specified, process all at once
        if batch_size is None or batch_size >= num_frames:
            pred_th = _batch_to_tensor(pred, self.device, to_chw=True)
            target_th = _batch_to_tensor(target, self.device, to_chw=True)

            # Normalize to [0, 1] range for fused-ssim
            # Intelligently detect if normalization is needed
            # Check data type and actual range to determine if input is in [0, 255] or [0, 1]
            if pred_th.dtype == th.uint8 or pred_th.max() > 1.0:
                # Data is likely in [0, 255] range
                pred_th = pred_th / self.data_range
            if target_th.dtype == th.uint8 or target_th.max() > 1.0:
                # Data is likely in [0, 255] range
                target_th = target_th / self.data_range

            with th.no_grad():
                # fused-ssim processes batch and returns per-image SSIM
                # Shape: (B,) with mean SSIM per image
                ssim_per_image = []
                for i in range(pred_th.shape[0]):
                    ssim_val = fused_ssim.fused_ssim(
                        pred_th[i : i + 1], target_th[i : i + 1], padding="same", train=False
                    )
                    ssim_per_image.append(float(ssim_val.item()))

            return ssim_per_image

        # Process in batches
        all_ssim = []
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_th = _batch_to_tensor(pred[i:end_idx], self.device, to_chw=True)
            target_th = _batch_to_tensor(target[i:end_idx], self.device, to_chw=True)

            # Normalize to [0, 1] range for fused-ssim
            # Intelligently detect if normalization is needed
            # Check data type and actual range to determine if input is in [0, 255] or [0, 1]
            if pred_th.dtype == th.uint8 or pred_th.max() > 1.0:
                # Data is likely in [0, 255] range
                pred_th = pred_th / self.data_range
            if target_th.dtype == th.uint8 or target_th.max() > 1.0:
                # Data is likely in [0, 255] range
                target_th = target_th / self.data_range

            with th.no_grad():
                for j in range(pred_th.shape[0]):
                    ssim_val = fused_ssim.fused_ssim(
                        pred_th[j : j + 1], target_th[j : j + 1], padding="same", train=False
                    )
                    all_ssim.append(float(ssim_val.item()))

        return all_ssim


@MetricRegistry.register("lpips")
class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric using pyiqa.

    Default backbone is VGG as it is more sensitive to perceptual differences
    and produces ~50% higher LPIPS values compared to AlexNet on the same images.
    """

    def __init__(self, device: str = "cuda", net: str = "vgg"):
        """
        Initialize LPIPS metric.

        Args:
            device: Device to run computation on
            net: Network backbone ('alex', 'vgg', or 'squeeze'). Default is 'vgg'.
                 VGG is more sensitive to differences and recommended for most use cases.
                 AlexNet is faster but less sensitive.
        """
        super().__init__(name="LPIPS", device=device)
        self.net = net
        self._model = None

    def _load_model(self):
        """Lazy load LPIPS model."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("LPIPS requires the 'pyiqa' package. Install it with: pip install pyiqa")
            # Map net names to pyiqa LPIPS metric names
            metric_name = f"lpips-{self.net}" if self.net != "alex" else "lpips"
            self._model = pyiqa.create_metric(metric_name, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        """
        Compute LPIPS between prediction and target.

        Args:
            pred: Predicted image
            target: Ground truth image

        Returns:
            LPIPS value (lower is better, 0 means identical)
        """
        self._load_model()

        # Convert to torch tensor
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        if isinstance(target, np.ndarray):
            target_th = th.from_numpy(target).float().to(self.device)
        else:
            target_th = target.float().to(self.device)

        # Normalize to [0, 1] range (pyiqa requirement)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0
        if target_th.max() > 1.0:
            target_th = target_th / 255.0

        # pyiqa models expect (B, C, H, W) format
        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:  # (H, W, C)
                pred_th = pred_th.permute(2, 0, 1)  # -> (C, H, W)
            pred_th = pred_th.unsqueeze(0)  # -> (1, C, H, W)
        elif pred_th.ndim == 2:  # Grayscale (H, W)
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

        if target_th.ndim == 3:
            if target_th.shape[2] in [1, 3]:  # (H, W, C)
                target_th = target_th.permute(2, 0, 1)  # -> (C, H, W)
            target_th = target_th.unsqueeze(0)  # -> (1, C, H, W)
        elif target_th.ndim == 2:  # Grayscale (H, W)
            target_th = target_th.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

        with th.no_grad():
            lpips_value = self._model(pred_th, target_th)

        return float(lpips_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute LPIPS for a batch of frames on GPU.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames, shape (T, H, W, C)
            batch_size: Number of frames per batch. None means all frames at once.
                        Note: LPIPS uses more GPU memory, so smaller batches may be needed.

        Returns:
            List of LPIPS values for each frame
        """
        self._load_model()

        num_frames = pred.shape[0]

        # Default batch_size
        if batch_size is None:
            batch_size = num_frames

        lpips_values = []

        # Process in batches
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]
            target_batch = target[i:end_idx]

            # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)
            target_th = _batch_to_tensor(target_batch, self.device, to_chw=True)

            # Normalize to [0, 1] range (pyiqa requirement)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0
            if target_th.max() > 1.0:
                target_th = target_th / 255.0

            with th.no_grad():
                # pyiqa models expect (B, C, H, W) and can process batches directly
                batch_lpips = self._model(pred_th, target_th)
                # Handle both single value and batch outputs
                if batch_lpips.numel() == 1:
                    lpips_values.append(float(batch_lpips.item()))
                else:
                    # Flatten to 1D and convert to list of floats
                    lpips_values.extend([float(x) for x in batch_lpips.flatten().cpu().tolist()])

        return lpips_values


@MetricRegistry.register("lq_color_de2000")
class LQColorDE2000(BaseMetric):
    """Mean CIEDE2000 color difference between SR output and LQ input.

    The SR image is resized to the LQ resolution before converting both images
    to CIELAB. This is intended to catch global color/tone drift in SR outputs
    without penalizing high-frequency detail that is absent from the LQ image.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(name="LQ_COLOR_DE2000", device=device)
        try:
            import cv2
            from skimage.color import deltaE_ciede2000, rgb2lab
        except ImportError as e:
            raise ImportError("LQColorDE2000 requires cv2 and scikit-image.") from e

        self._cv2 = cv2
        self._rgb2lab = rgb2lab
        self._delta_e = deltaE_ciede2000

    @staticmethod
    def _to_numpy_nhwc(frames: Union[np.ndarray, th.Tensor]) -> np.ndarray:
        """Convert one image or a frame batch to NHWC numpy format."""
        if isinstance(frames, th.Tensor):
            frames = frames.detach().cpu()
            if frames.ndim == 3:
                if frames.shape[0] in [1, 3, 4]:
                    frames = frames.permute(1, 2, 0)
                frames = frames.unsqueeze(0)
            elif frames.ndim == 4:
                if frames.shape[1] in [1, 3, 4]:
                    frames = frames.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Expected image/frame batch tensor with 3 or 4 dims, got shape={tuple(frames.shape)}")
            return frames.numpy()

        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        elif frames.ndim == 4:
            if frames.shape[1] in [1, 3, 4] and frames.shape[-1] not in [1, 3, 4]:
                frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            raise ValueError(f"Expected image/frame batch array with 3 or 4 dims, got shape={frames.shape}")
        return frames

    @staticmethod
    def _to_rgb01(frames: np.ndarray) -> np.ndarray:
        """Normalize NHWC RGB-like arrays to float32 RGB in [0, 1]."""
        frames = frames.astype(np.float32, copy=False)
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] >= 3:
            frames = frames[..., :3]
        else:
            raise ValueError(f"Expected 1 or >=3 channels, got shape={frames.shape}")

        if frames.min() < 0:
            frames = (frames + 1.0) * 0.5
        elif frames.max() > 1.0:
            frames = frames / 255.0
        return np.clip(frames, 0.0, 1.0)

    def _resize_to_target(self, pred_rgb01: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if pred_rgb01.shape[:2] == (target_h, target_w):
            return pred_rgb01

        src_h, src_w = pred_rgb01.shape[:2]
        src_area = src_h * src_w
        target_area = target_h * target_w
        interpolation = self._cv2.INTER_AREA if target_area <= src_area else self._cv2.INTER_CUBIC
        return self._cv2.resize(pred_rgb01, (target_w, target_h), interpolation=interpolation)

    def _compute_pair_rgb01(self, pred_rgb01: np.ndarray, target_rgb01: np.ndarray) -> float:
        target_h, target_w = target_rgb01.shape[:2]
        pred_resized = self._resize_to_target(pred_rgb01, target_h, target_w)
        pred_lab = self._rgb2lab(pred_resized)
        target_lab = self._rgb2lab(target_rgb01)
        delta_e = self._delta_e(pred_lab, target_lab)
        return float(np.mean(delta_e))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        if target is None:
            raise ValueError("lq_color_de2000 requires the LQ image as target.")

        pred_rgb01 = self._to_rgb01(self._to_numpy_nhwc(pred))[0]
        target_rgb01 = self._to_rgb01(self._to_numpy_nhwc(target))[0]
        return self._compute_pair_rgb01(pred_rgb01, target_rgb01)

    def compute_batch(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
        batch_size: int = None,
    ) -> List[float]:
        if target is None:
            raise ValueError("lq_color_de2000 requires the LQ image batch as target.")

        pred_np = self._to_numpy_nhwc(pred)
        target_np = self._to_numpy_nhwc(target)
        if pred_np.shape[0] != target_np.shape[0]:
            raise ValueError(f"Frame count mismatch: pred={pred_np.shape[0]}, target={target_np.shape[0]}")

        num_frames = pred_np.shape[0]
        if batch_size is None or batch_size <= 0:
            batch_size = num_frames

        scores = []
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            pred_rgb01 = self._to_rgb01(pred_np[start:end])
            target_rgb01 = self._to_rgb01(target_np[start:end])
            for pred_frame, target_frame in zip(pred_rgb01, target_rgb01):
                scores.append(self._compute_pair_rgb01(pred_frame, target_frame))
        return scores


@MetricRegistry.register("niqe")
class NIQE(BaseMetric):
    """Natural Image Quality Evaluator (NIQE) - No-reference image quality metric."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize NIQE metric.

        Args:
            device: Device to run computation on (note: NIQE runs on CPU)
        """
        super().__init__(name="NIQE", device=device)
        self._model = None

    def _load_model(self):
        """Lazy load NIQE model."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("NIQE requires the 'pyiqa' package. Install it with: pip install pyiqa")
            # NIQE is a no-reference metric, so we only need to pass predicted images
            self._model = pyiqa.create_metric("niqe", device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """
        Compute NIQE for prediction (no-reference metric, target is ignored).

        Args:
            pred: Predicted image
            target: Ground truth image (ignored, kept for API consistency)

        Returns:
            NIQE value (lower is better)
        """
        self._load_model()

        # Convert to torch tensor
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        # Normalize to [0, 1] range (pyiqa requirement)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        # pyiqa models expect (B, C, H, W) format
        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:  # (H, W, C)
                pred_th = pred_th.permute(2, 0, 1)  # -> (C, H, W)
            pred_th = pred_th.unsqueeze(0)  # -> (1, C, H, W)
        elif pred_th.ndim == 2:  # Grayscale (H, W)
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

        with th.no_grad():
            niqe_value = self._model(pred_th)

        return float(niqe_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute NIQE for a batch of frames.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames (ignored, kept for API consistency)
            batch_size: Number of frames per batch. None means all frames at once.

        Returns:
            List of NIQE values for each frame
        """
        self._load_model()

        num_frames = pred.shape[0]

        # Default batch_size
        if batch_size is None:
            batch_size = num_frames

        niqe_values = []

        # Process in batches
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]

            # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)

            # Normalize to [0, 1] range (pyiqa requirement)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0

            with th.no_grad():
                # pyiqa models expect (B, C, H, W) and can process batches directly
                batch_niqe = self._model(pred_th)
                # Handle both single value and batch outputs
                if batch_niqe.numel() == 1:
                    niqe_values.append(float(batch_niqe.item()))
                else:
                    # Flatten to 1D and convert to list of floats
                    niqe_values.extend([float(x) for x in batch_niqe.flatten().cpu().tolist()])

        return niqe_values


@MetricRegistry.register("musiq")
class MUSIQ(BaseMetric):
    """Multi-Scale Image Quality Transformer (MUSIQ) - No-reference image quality metric.

    Available models:
    - musiq: Original MUSIQ model (default)
    - musiq-ava: MUSIQ trained on AVA dataset
    - musiq-paq2piq: MUSIQ trained on PaQ-2-PiQ dataset
    - musiq-spaq: MUSIQ trained on SPAQ dataset
    """

    def __init__(self, device: str = "cuda", model: str = "musiq"):
        """
        Initialize MUSIQ metric.

        Args:
            device: Device to run computation on
            model: MUSIQ model variant to use. Options:
                   'musiq', 'musiq-ava', 'musiq-paq2piq', 'musiq-spaq'
                   Default is 'musiq'.
        """
        super().__init__(name="MUSIQ", device=device)
        self.model = model
        self._model = None

    def _load_model(self):
        """Lazy load MUSIQ model."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("MUSIQ requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric(self.model, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """
        Compute MUSIQ for prediction (no-reference metric, target is ignored).

        Args:
            pred: Predicted image
            target: Ground truth image (ignored, kept for API consistency)

        Returns:
            MUSIQ value (higher is better, range: 0-100)
        """
        self._load_model()

        # Convert to torch tensor
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        # Normalize to [0, 1] range (pyiqa requirement)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        # pyiqa models expect (B, C, H, W) format
        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:  # (H, W, C)
                pred_th = pred_th.permute(2, 0, 1)  # -> (C, H, W)
            pred_th = pred_th.unsqueeze(0)  # -> (1, C, H, W)
        elif pred_th.ndim == 2:  # Grayscale (H, W)
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

        with th.no_grad():
            musiq_value = self._model(pred_th)

        return float(musiq_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute MUSIQ for a batch of frames.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames (ignored, kept for API consistency)
            batch_size: Number of frames per batch. None means all frames at once.

        Returns:
            List of MUSIQ values for each frame
        """
        self._load_model()

        num_frames = pred.shape[0]

        # Default batch_size
        if batch_size is None:
            batch_size = num_frames

        musiq_values = []

        # Process in batches
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]

            # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)

            # Normalize to [0, 1] range (pyiqa requirement)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0

            with th.no_grad():
                # pyiqa models expect (B, C, H, W) and can process batches directly
                batch_musiq = self._model(pred_th)
                # Handle both single value and batch outputs
                if batch_musiq.numel() == 1:
                    musiq_values.append(float(batch_musiq.item()))
                else:
                    # Flatten to 1D and convert to list of floats
                    musiq_values.extend([float(x) for x in batch_musiq.flatten().cpu().tolist()])

        return musiq_values


@MetricRegistry.register("musiq_paq2piq")
class MUSIQPaQ2PiQ(MUSIQ):
    """MUSIQ trained on PaQ-2-PiQ — authentic-distortion IQA, range ~[0, 100].
    Separate registration so it can be reported alongside the KonIQ-default MUSIQ.
    """

    def __init__(self, device: str = "cuda", model: str = "musiq-paq2piq"):
        super().__init__(device=device, model=model)
        self.name = "MUSIQ-PaQ2PiQ"


@MetricRegistry.register("musiq_spaq")
class MUSIQSPAQ(MUSIQ):
    """MUSIQ trained on SPAQ — smartphone-photography IQA, range ~[0, 100].
    Separate registration so it can be reported alongside the KonIQ-default MUSIQ.
    """

    def __init__(self, device: str = "cuda", model: str = "musiq-spaq"):
        super().__init__(device=device, model=model)
        self.name = "MUSIQ-SPAQ"


@MetricRegistry.register("clipiqa")
class CLIPIQA(BaseMetric):
    """CLIP-based Image Quality Assessment (CLIPIQA) - No-reference image quality metric.

    Available models:
    - clipiqa: Original CLIPIQA (zero-shot, hand-written antonym prompts, RN50, 224²)
    - clipiqa+: CLIPIQA+ with CoOp-learned prompts trained on KonIQ-10k (RN50, 224²)
    - clipiqa+_vitL14_512: CLIPIQA+ with ViT-L/14 backbone at 512² input
    - clipiqa+_rn50_512: CLIPIQA+ with RN50 backbone at 512² input

    Default here is the zero-shot 'clipiqa' because it's more OOD-robust to synthesized /
    diffusion imagery. For a learned counterpart with higher KonIQ PLCC, see the sibling
    'clipiqa_plus' metric (defaults to clipiqa+_vitL14_512).
    """

    def __init__(self, device: str = "cuda", model: str = "clipiqa"):
        """
        Initialize CLIPIQA metric.

        Args:
            device: Device to run computation on
            model: CLIPIQA model variant to use. Options:
                   'clipiqa', 'clipiqa+', 'clipiqa+_vitL14_512', 'clipiqa+_rn50_512'
                   Default is 'clipiqa' (zero-shot).
        """
        super().__init__(name="CLIPIQA", device=device)
        self.model = model
        self._model = None

    def _load_model(self):
        """Lazy load CLIPIQA model."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("CLIPIQA requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric(self.model, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """
        Compute CLIPIQA for prediction (no-reference metric, target is ignored).

        Args:
            pred: Predicted image
            target: Ground truth image (ignored, kept for API consistency)

        Returns:
            CLIPIQA value (higher is better, range: 0-1)
        """
        self._load_model()

        # Convert to torch tensor
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        # Normalize to [0, 1] range (pyiqa requirement)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        # pyiqa models expect (B, C, H, W) format
        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:  # (H, W, C)
                pred_th = pred_th.permute(2, 0, 1)  # -> (C, H, W)
            pred_th = pred_th.unsqueeze(0)  # -> (1, C, H, W)
        elif pred_th.ndim == 2:  # Grayscale (H, W)
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

        with th.no_grad():
            clipiqa_value = self._model(pred_th)

        return float(clipiqa_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """
        Compute CLIPIQA for a batch of frames.

        Args:
            pred: Predicted frames, shape (T, H, W, C)
            target: Ground truth frames (ignored, kept for API consistency)
            batch_size: Number of frames per batch. None means all frames at once.

        Returns:
            List of CLIPIQA values for each frame
        """
        self._load_model()

        num_frames = pred.shape[0]

        # Default batch_size
        if batch_size is None:
            batch_size = num_frames

        clipiqa_values = []

        # Process in batches
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]

            # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)

            # Normalize to [0, 1] range (pyiqa requirement)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0

            with th.no_grad():
                # pyiqa models expect (B, C, H, W) and can process batches directly
                batch_clipiqa = self._model(pred_th)
                # Handle both single value and batch outputs
                if batch_clipiqa.numel() == 1:
                    clipiqa_values.append(float(batch_clipiqa.item()))
                else:
                    # Flatten to 1D and convert to list of floats
                    clipiqa_values.extend([float(x) for x in batch_clipiqa.flatten().cpu().tolist()])

        return clipiqa_values


@MetricRegistry.register("clipiqa_plus")
class CLIPIQAPlus(CLIPIQA):
    """CLIPIQA+ with CoOp-learned prompts — separate registration so it can be reported
    side-by-side with zero-shot CLIPIQA in the eval viewer.

    Defaults to the ViT-L/14 @ 512² variant: strongest backbone and double the input
    resolution of the 224² default — a meaningful upgrade for 2K outputs where the 224²
    resize throws away detail. Swap via --clipiqa_plus_model at the CLI.

    Reported KonIQ-10k SRCC: ~0.89 (CLIPIQA+) vs ~0.71 (zero-shot CLIPIQA). The two
    disagreeing on a generated image is itself a useful signal — zero-shot is more
    OOD-robust to diffusion artifacts, learned is more accurate on KonIQ-style
    authentic distortions.
    """

    def __init__(self, device: str = "cuda", model: str = "clipiqa+_vitL14_512"):
        """
        Args:
            device: Device to run computation on.
            model: CLIPIQA+ variant. Options:
                   'clipiqa+', 'clipiqa+_vitL14_512' (default), 'clipiqa+_rn50_512'.
        """
        super().__init__(device=device, model=model)
        self.name = "CLIPIQA+"


@MetricRegistry.register("liqe")
class LIQE(BaseMetric):
    """LIQE (Language-guided IQE) — NR, higher is better, ~[1, 5] MOS-like.

    Multi-task IQA that uses CLIP + natural-language prompts to jointly infer
    distortion type, scene, and a quality score. More OOD-robust than
    single-task learned metrics like MUSIQ on synthetic / generative imagery.

    Variants (from pyiqa.list_models()):
        - liqe       : original LIQE (Zhang et al. CVPR'23)
        - liqe_mix   : mixture-of-datasets variant, generally more robust (default)
    """

    def __init__(self, device: str = "cuda", model: str = "liqe_mix"):
        super().__init__(name="LIQE", device=device)
        self.model = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("LIQE requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric(self.model, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        self._load_model()

        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:
                pred_th = pred_th.permute(2, 0, 1)
            pred_th = pred_th.unsqueeze(0)
        elif pred_th.ndim == 2:
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)

        with th.no_grad():
            value = self._model(pred_th)
        return float(value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        self._load_model()

        num_frames = pred.shape[0]
        if batch_size is None:
            batch_size = num_frames

        values: List[float] = []
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0
            with th.no_grad():
                batch_v = self._model(pred_th)
                if batch_v.numel() == 1:
                    values.append(float(batch_v.item()))
                else:
                    values.extend([float(x) for x in batch_v.flatten().cpu().tolist()])
        return values


@MetricRegistry.register("qualiclip")
class QualiCLIP(BaseMetric):
    """QualiCLIP — contrastive CLIP-based IQA, NR, higher is better, ~[0, 1].

    Trained to rank quality via self-supervised contrastive objective over
    degraded-image pairs. Complements CLIPIQA (zero-shot antonym prompts) with
    a learned ranking head.

    Variant:
        - qualiclip : zero-shot-style default configuration (KonIQ-aligned).

    See QualiCLIPPlus (registered as 'qualiclip_plus') for learned-prompt
    variants with higher KonIQ correlation.
    """

    def __init__(self, device: str = "cuda", model: str = "qualiclip"):
        super().__init__(name="QualiCLIP", device=device)
        self.model = model
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("QualiCLIP requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric(self.model, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        self._load_model()

        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:
                pred_th = pred_th.permute(2, 0, 1)
            pred_th = pred_th.unsqueeze(0)
        elif pred_th.ndim == 2:
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)

        with th.no_grad():
            value = self._model(pred_th)
        return float(value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        self._load_model()

        num_frames = pred.shape[0]
        if batch_size is None:
            batch_size = num_frames

        values: List[float] = []
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)
            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0
            with th.no_grad():
                batch_v = self._model(pred_th)
                if batch_v.numel() == 1:
                    values.append(float(batch_v.item()))
                else:
                    values.extend([float(x) for x in batch_v.flatten().cpu().tolist()])
        return values


@MetricRegistry.register("qualiclip_plus")
class QualiCLIPPlus(QualiCLIP):
    """QualiCLIP+ — learned-prompt variant of QualiCLIP. Registered separately so
    both zero-shot and learned variants can be reported side-by-side (mirrors the
    CLIPIQA / CLIPIQAPlus split).

    Variants:
        - qualiclip+          (default, KonIQ-10k learned)
        - qualiclip+-clive    (CLIVE learned)
        - qualiclip+-flive    (FLIVE learned)
        - qualiclip+-spaq     (SPAQ learned)
    """

    def __init__(self, device: str = "cuda", model: str = "qualiclip+"):
        super().__init__(device=device, model=model)
        self.name = "QualiCLIP+"


@MetricRegistry.register("maniqa")
class MANIQA(BaseMetric):
    """Multi-dimension Attention Network for Image Quality Assessment (MANIQA) — no-reference.

    Value range: roughly [0, 1], higher is better.

    Available pyiqa variants (checked against pyiqa.list_models()):
        - maniqa           (default, KonIQ-10k trained)
        - maniqa-kadid     (KADID-10k)
        - maniqa-pipal     (PIPAL)
    """

    def __init__(self, device: str = "cuda", model: str = "maniqa"):
        """
        Args:
            device: Device to run computation on.
            model: MANIQA variant — 'maniqa', 'maniqa-kadid', or 'maniqa-pipal'.
        """
        super().__init__(name="MANIQA", device=device)
        self.model = model
        self._model = None

    def _load_model(self):
        """Lazy load MANIQA model."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("MANIQA requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric(self.model, device=th.device(self.device))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """
        Compute MANIQA for prediction (no-reference, target is ignored).

        Returns:
            MANIQA value (higher is better, ~[0, 1]).
        """
        self._load_model()

        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)

        # Normalize to [0, 1] range (pyiqa requirement)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0

        # pyiqa models expect (B, C, H, W) format
        if pred_th.ndim == 3:
            if pred_th.shape[2] in [1, 3]:  # (H, W, C)
                pred_th = pred_th.permute(2, 0, 1)
            pred_th = pred_th.unsqueeze(0)
        elif pred_th.ndim == 2:
            pred_th = pred_th.unsqueeze(0).unsqueeze(0)

        with th.no_grad():
            maniqa_value = self._model(pred_th)

        return float(maniqa_value.item())

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """Compute MANIQA for a batch of frames."""
        self._load_model()

        num_frames = pred.shape[0]
        if batch_size is None:
            batch_size = num_frames

        maniqa_values = []
        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            pred_batch = pred[i:end_idx]
            pred_th = _batch_to_tensor(pred_batch, self.device, to_chw=True)

            if pred_th.max() > 1.0:
                pred_th = pred_th / 255.0

            with th.no_grad():
                batch_maniqa = self._model(pred_th)
                if batch_maniqa.numel() == 1:
                    maniqa_values.append(float(batch_maniqa.item()))
                else:
                    maniqa_values.extend([float(x) for x in batch_maniqa.flatten().cpu().tolist()])

        return maniqa_values


@MetricRegistry.register("qalign")
class QAlign(BaseMetric):
    """Q-Align quality + aesthetic scores via pyiqa (NR, higher is better, ~[1, 5] MOS scale).

    For quality: uses multi-crop (grid of 448×448 crops) to preserve local detail at high resolutions.
    For aesthetic: uses default full-image resize (global context matters for aesthetics).

    Multi-crop strategy avoids the CLIPImageProcessor's resize-to-448 which destroys fine SR details:
    - ≤448px: single crop (whole image, same as default)
    - 449-512px: single center crop
    - >512px: 5 crops (4 corners + center), scores averaged

    pyiqa enforces batch_size == 1, so caller must loop per-sample.
    Adds ~14GB GPU memory in fp16.
    """

    CROP_SIZE = 448  # CLIPImageProcessor target size, must match Q-Align training resolution

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(name="QAlign", device=device)
        self._model = None

    def _load_model(self):
        """Lazy load Q-Align model, or move it back to GPU if previously offloaded to CPU."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("QAlign requires the 'pyiqa' package. Install it with: pip install pyiqa")
            self._model = pyiqa.create_metric("qalign", device=th.device(self.device))
        elif next(self._model.parameters()).device.type == "cpu":
            self._model = self._model.to(self.device)

    def offload_to_cpu(self):
        """Move Q-Align model to CPU to free GPU memory. Call _load_model() to reload."""
        if self._model is not None:
            self._model = self._model.to("cpu")
            th.cuda.empty_cache()

    def _get_image_processor(self):
        """Access the CLIPImageProcessor from the underlying QAlign arch."""
        return self._model.net.image_processor

    def _preprocess_single(self, pil_img):
        """Run CLIPImageProcessor on a single PIL image → [1, C, H, W] half tensor."""
        from pyiqa.archs.qalign_arch import expand2square

        pil_img = expand2square(pil_img)
        tensor = self._get_image_processor().preprocess(pil_img, return_tensors="pt")["pixel_values"].half()
        return tensor.to(self.device)

    def _score_tensor(self, image_tensor, task_):
        """Call the underlying mPLUG-Owl2 model.score() with pre-processed tensor."""
        return float(
            self._model.net.model.score(images=None, image_tensor=image_tensor, task_=task_, input_="image").item()
        )

    def _extract_crops(self, img_chw):
        """Extract 448×448 crops from a (C, H, W) tensor in [0, 1].

        Strategy:
        - H, W ≤ 448: single crop (whole image, handled by default preprocess)
        - 448 < H, W ≤ 512: single center crop
        - H, W > 512: 5 crops — 4 corners + center

        Returns list of (C, H, W) tensors.
        """
        C, H, W = img_chw.shape
        cs = self.CROP_SIZE
        if H <= cs and W <= cs:
            return [img_chw]

        # Center crop position
        cy, cx = (H - cs) // 2, (W - cs) // 2

        if H <= 512 and W <= 512:
            return [img_chw[:, cy : cy + cs, cx : cx + cs]]

        # 5-crop for >512: 4 corners + center
        crops = [
            img_chw[:, 0:cs, 0:cs],  # top-left
            img_chw[:, 0:cs, W - cs : W],  # top-right
            img_chw[:, H - cs : H, 0:cs],  # bottom-left
            img_chw[:, H - cs : H, W - cs : W],  # bottom-right
            img_chw[:, cy : cy + cs, cx : cx + cs],  # center
        ]
        return crops

    def _to_pil_list(self, crops):
        """Convert list of (C, H, W) tensors in [0, 1] to list of PIL images."""
        import torchvision.transforms.functional as F

        return [F.to_pil_image(c.cpu().clamp(0, 1)) for c in crops]

    def _batch_score(self, pil_images, task_):
        """Batch score PIL images by calling model.score() directly (bypasses pyiqa batch=1 limit).

        Args:
            pil_images: list of PIL images
            task_: 'quality' or 'aesthetic'

        Returns:
            list of float scores
        """
        with th.no_grad():
            scores = self._model.net.model.score(images=pil_images, task_=task_, input_="image")
        return scores.tolist()

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> dict:
        """Compute Q-Align quality and aesthetic scores (no-reference, target is ignored).

        Bypasses pyiqa's batch_size=1 restriction by calling model.score() directly,
        batching all quality crops + aesthetic full-image in fewer forward passes.

        Args:
            pred: Predicted image in [0, 1] or [0, 255] range, shape (C, H, W) or (H, W, C)

        Returns:
            dict with three float scores:
              - qalign_quality         — mean over 448×448 patch crops (detail-preserving)
              - qalign_quality_native  — quality on the full image with default pyiqa
                                         preprocess (resize→448), matches the aesthetic
                                         path and the "vanilla" Q-Align quality number
              - qalign_aesthetic       — full-image aesthetic score
        """
        self._load_model()

        import torchvision.transforms.functional as F

        # Convert to [C, H, W] float tensor in [0, 1]
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0
        if pred_th.ndim == 3 and pred_th.shape[2] in [1, 3]:
            pred_th = pred_th.permute(2, 0, 1)

        # --- Quality (multi-crop): batch all 448² crops in one forward pass ---
        crops = self._extract_crops(pred_th)
        crop_pils = self._to_pil_list(crops)
        quality_scores = self._batch_score(crop_pils, task_="quality")
        quality = sum(quality_scores) / len(quality_scores)

        # --- Native quality + aesthetic: full-image resize (global context) ---
        pil_full = F.to_pil_image(pred_th.cpu().clamp(0, 1))
        quality_native = self._batch_score([pil_full], task_="quality")[0]
        aesthetic = self._batch_score([pil_full], task_="aesthetic")[0]

        return {
            "qalign_quality": quality,
            "qalign_quality_native": quality_native,
            "qalign_aesthetic": aesthetic,
        }

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> dict:
        """Compute QAlign quality and aesthetic scores for a batch of frames.

        Args:
            pred: Predicted frames, shape (T, H, W, C) in [0, 255] uint8
            target: Ignored (no-reference metric)
            batch_size: Ignored (pyiqa enforces batch_size=1 internally)

        Returns:
            dict with 'qalign_quality', 'qalign_quality_native', and 'qalign_aesthetic'
            lists of per-frame scores
        """
        quality_values, quality_native_values, aesthetic_values = [], [], []
        for i in range(pred.shape[0]):
            scores = self.compute(pred[i])
            quality_values.append(scores["qalign_quality"])
            quality_native_values.append(scores["qalign_quality_native"])
            aesthetic_values.append(scores["qalign_aesthetic"])
        # NOTE: the model stays on GPU after this call. The caller is responsible for
        # invoking offload_to_cpu() once the whole eval run is done — offloading here
        # would trigger a ~14GB PCIe reload for every sample.
        return {
            "qalign_quality": quality_values,
            "qalign_quality_native": quality_native_values,
            "qalign_aesthetic": aesthetic_values,
        }


@MetricRegistry.register("qalign_native")
class QAlignQualityNative(BaseMetric):
    """Q-Align "native quality" — pyiqa's default full-image-resize-to-448 quality
    path, a single float per image. Batchable across images via a single
    `model.score()` call, bypassing pyiqa's batch_size=1 restriction.

    This is the vanilla Q-Align quality number the paper reports. The multi-crop
    variant and the aesthetic head are intentionally not provided here — they
    are inherently per-image (multi-crop set depends on input H/W) and didn't
    add enough signal to justify the ~2× cost on our benchmarks.

    Model size: ~14GB GPU memory in fp16.
    """

    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(name="QAlign-QualityNative", device=device)
        self._model = None

    def _load_model(self):
        """Lazy-load pyiqa's qalign metric; move back to GPU if previously offloaded."""
        if self._model is None:
            try:
                import pyiqa
            except ImportError:
                raise ImportError("QAlignQualityNative requires the 'pyiqa' package. Install with: pip install pyiqa")
            self._model = pyiqa.create_metric("qalign", device=th.device(self.device))
        elif next(self._model.parameters()).device.type == "cpu":
            self._model = self._model.to(self.device)

    def offload_to_cpu(self):
        """Move the model to CPU to free GPU memory (e.g. before the all_reduce)."""
        if self._model is not None:
            self._model = self._model.to("cpu")
            th.cuda.empty_cache()

    def _normalize_to_chw_01(self, pred: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        """Numpy/torch (HWC or CHW) in [0, 255] or [0, 1] -> (C, H, W) float on device in [0, 1]."""
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0
        if pred_th.ndim == 3 and pred_th.shape[2] in [1, 3]:
            pred_th = pred_th.permute(2, 0, 1)
        return pred_th

    def _batch_score(self, pil_images) -> List[float]:
        """Score N PIL images in one forward pass via the underlying mPLUG-Owl2
        `model.score()`. Bypasses pyiqa's `batch_size=1` wrapper.
        """
        with th.no_grad():
            scores = self._model.net.model.score(images=pil_images, task_="quality", input_="image")
        return scores.tolist()

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        self._load_model()
        import torchvision.transforms.functional as F

        pred_th = self._normalize_to_chw_01(pred)
        pil = F.to_pil_image(pred_th.cpu().clamp(0, 1))
        return self._batch_score([pil])[0]

    def compute_batch_list(self, preds, targets=None) -> List[float]:
        """All images scored in a single model.score() call."""
        self._load_model()
        import torchvision.transforms.functional as F

        pils = [F.to_pil_image(self._normalize_to_chw_01(p).cpu().clamp(0, 1)) for p in preds]
        return self._batch_score(pils)

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """(T, H, W, C) -> list of per-frame scores. Delegates to compute_batch_list."""
        return self.compute_batch_list([pred[i] for i in range(pred.shape[0])])


@MetricRegistry.register("deqa_score")
class DeQAScore(BaseMetric):
    """DeQA-Score (NR-IQA, ~[1, 5] MOS scale, higher is better) via mPLUG-Owl2 +
    score-distribution fine-tune. Vendored adaptation of zhiyuanyou/DeQA-Score
    living at linearvsr/_src/evaluations/metrics/deqa_score/ (the upstream code
    is pinned to transformers==4.36.1; the vendored copy is patched to work on
    transformers>=4.46).

    Same backbone as QAlign (~14 GB GPU memory in fp16) but a different fine-
    tune; the two metrics give complementary signal. The score is a continuous
    softmax-weighted mean over the 5 quality words, not the integer level.

    pyiqa is NOT involved — we load the HuggingFace checkpoint directly through
    our vendored Scorer wrapper.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "zhiyuanyou/DeQA-Score-Mix3",
        **kwargs,
    ):
        super().__init__(name="DeQAScore", device=device)
        self._model_path = model_path
        self._scorer = None

    def _load_model(self):
        """Lazy-load the vendored Scorer; move it back to GPU if previously offloaded."""
        if self._scorer is None:
            from pid._src.evaluations.metrics.deqa_score.scorer import Scorer

            self._scorer = Scorer(pretrained=self._model_path, device=self.device)
        elif next(self._scorer.parameters()).device.type == "cpu":
            self._scorer = self._scorer.to(self.device)
            self._scorer.input_ids = self._scorer.input_ids.to(self.device)
            self._scorer.weight_tensor = self._scorer.weight_tensor.to(self.device)

    def offload_to_cpu(self):
        """Move the model to CPU to free GPU memory (e.g. before an all_reduce)."""
        if self._scorer is not None:
            self._scorer = self._scorer.to("cpu")
            self._scorer.input_ids = self._scorer.input_ids.to("cpu")
            self._scorer.weight_tensor = self._scorer.weight_tensor.to("cpu")
            th.cuda.empty_cache()

    def _normalize_to_chw_01(self, pred: Union[np.ndarray, th.Tensor]) -> th.Tensor:
        """Numpy/torch (HWC or CHW) in [0, 255] or [0, 1] -> (C, H, W) float on device in [0, 1]."""
        if isinstance(pred, np.ndarray):
            pred_th = th.from_numpy(pred).float().to(self.device)
        else:
            pred_th = pred.float().to(self.device)
        if pred_th.max() > 1.0:
            pred_th = pred_th / 255.0
        if pred_th.ndim == 3 and pred_th.shape[2] in [1, 3]:
            pred_th = pred_th.permute(2, 0, 1)
        return pred_th

    def _batch_score(self, pil_images: List) -> List[float]:
        """Score N PIL images in one forward pass via the vendored Scorer."""
        scores = self._scorer(pil_images)
        return scores.float().tolist()

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        self._load_model()
        import torchvision.transforms.functional as F

        pred_th = self._normalize_to_chw_01(pred)
        pil = F.to_pil_image(pred_th.cpu().clamp(0, 1))
        return self._batch_score([pil])[0]

    def compute_batch_list(self, preds, targets=None) -> List[float]:
        """All images scored in a single forward pass."""
        self._load_model()
        import torchvision.transforms.functional as F

        pils = [F.to_pil_image(self._normalize_to_chw_01(p).cpu().clamp(0, 1)) for p in preds]
        return self._batch_score(pils)

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """(T, H, W, C) -> list of per-frame scores. Delegates to compute_batch_list."""
        return self.compute_batch_list([pred[i] for i in range(pred.shape[0])])


# =============================================================================
# VLM-based NR quality metrics
# =============================================================================
# Both classes share the same shape:
#   - Lazy-load a VLM (~7-8B params) on first compute()
#   - Downscale input to max-side 512 via _resize_for_vlm to cap vision tokens
#   - Run greedy generate, parse a single float score out of the response
#   - Backend is selectable: backend="auto" (default) prefers vLLM when installed
#     and falls back to transformers otherwise. Pass backend="transformers" or
#     backend="vllm" to force one.
# Prompts and parsing logic are ports of the filter implementations at
# curation/filters/{visualquality_r1_filter.py, qwen3vl_filter.py}.
#
# vLLM caveat with torchrun: vLLM internally uses RANK/LOCAL_RANK env vars for
# its own workers; launching with torchrun + tensor_parallel_size=1 can conflict.
# Single-GPU eval or rank-per-vllm-instance launches are the tested configs.


def _resolve_vlm_backend(requested: str) -> str:
    """'auto' -> 'vllm' if importable, else 'transformers'. Else pass-through."""
    if requested == "auto":
        try:
            import vllm  # noqa: F401

            return "vllm"
        except ImportError:
            return "transformers"
    if requested not in ("transformers", "vllm"):
        raise ValueError(f"backend must be 'auto' | 'transformers' | 'vllm', got {requested!r}")
    return requested


_CACHED_LOCAL_RANK: Optional[str] = None


def _isolate_env_for_vllm() -> None:
    """Strip torchrun's distributed env vars and pin CUDA_VISIBLE_DEVICES to LOCAL_RANK.

    vLLM's V1 EngineCore is a `multiprocessing.spawn` child that inherits the
    parent's env. Under torchrun, that means every rank's EngineCore sees the
    same `MASTER_ADDR:MASTER_PORT` + `RANK`/`LOCAL_RANK`, so their internal
    TCPStores fight over torchrun's rendezvous socket and every front-end →
    EngineCore handshake times out after 600s (see eval_invsr1 job 875103).

    Spawn children also don't inherit `torch.cuda.set_device(local_rank)` —
    CUDA contexts aren't preserved across spawn — so without `CUDA_VISIBLE_DEVICES`
    narrowed per-rank, all 4 EngineCores would fall back to physical GPU 0 and
    OOM instead of distributing across 4 cards.

    Must be called after `dist.init_process_group()` (which stops needing these
    env vars once the PG is built) and before any `from vllm import LLM`.

    Idempotent across multiple VLM engines: we cache LOCAL_RANK at module scope
    because this function POPS it from os.environ, so a second call (when the
    evaluator loads a second VLM one-at-a-time) can't re-read it and would
    otherwise default all ranks to "0" — collapsing every EngineCore onto GPU 0.
    """
    import os

    global _CACHED_LOCAL_RANK
    if _CACHED_LOCAL_RANK is None:
        _CACHED_LOCAL_RANK = os.environ.get("LOCAL_RANK", "0")
    local_rank = _CACHED_LOCAL_RANK
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCHELASTIC_ERROR_FILE",
    ):
        os.environ.pop(key, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = local_rank


VQR1_PROMPT = (
    "You are doing the image quality assessment task. Here is the question: "
    "What is your overall rating on the quality of this picture? The rating should "
    "be a float between 1 and 5, rounded to two decimal places, with 1 representing "
    "very poor quality and 5 representing excellent quality."
)
VQR1_QUESTION_TEMPLATE = (
    "{Question} First output the thinking process in <think> </think> tags and "
    "then output the final answer with only one score in <answer> </answer> tags."
)


@MetricRegistry.register("visualquality_r1")
class VisualQualityR1(BaseMetric):
    """VisualQuality-R1 — Chain-of-Thought VLM IQA, NR, higher is better, 1-5 MOS-like.

    The model (TianheWu/VisualQuality-R1-7B, a Qwen2.5-VL-7B fine-tune) is
    lazy-loaded on first compute(); inputs are resized to max-side 512 before
    preprocessing to keep the vision-token count tractable on 2K SR outputs.

    Backends:
      - "auto" (default): vLLM if importable, else transformers
      - "transformers":   HuggingFace transformers.generate()
      - "vllm":           vllm.LLM synchronous engine (faster, fewer batch wins
                          per-image; bigger speedup if you ever batch-score)
    """

    DEFAULT_MODEL = "TianheWu/VisualQuality-R1-7B"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = DEFAULT_MODEL,
        max_side: int = 512,
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
        backend: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = False,
        max_model_len: int = 32768,
    ):
        super().__init__(name="VisualQuality-R1", device=device)
        self.model_name = model_name
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.backend = backend
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        # max_model_len: IQA prompts are an image (downscaled to max_side=512, ~324
        # visual tokens) + a CoT instruction + up to max_new_tokens of output +
        # <think> reasoning — realistically 2-4k tokens per seq. 32k gives an
        # order-of-magnitude safety margin (accounting for Q-Insight's longer CoT
        # and possible batched concurrent seqs) while still avoiding the Qwen2.5-VL
        # card default of 128k, which provisions a ~7 GiB per-seq KV ceiling that
        # previously caused "KV cache needed > available" failures when a second
        # VLM tried to load after the first released.
        self.max_model_len = max_model_len
        self._resolved_backend: str = ""
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            # transformers backend only: re-promote from CPU if previously offloaded.
            if self._resolved_backend == "transformers" and next(self._model.parameters()).device.type == "cpu":
                self._model = self._model.to(self.device)
            return
        self._resolved_backend = _resolve_vlm_backend(self.backend)
        if self._resolved_backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def offload_to_cpu(self):
        """Move the transformers model to CPU between evals so it doesn't squat on
        GPU memory during the next training chunk. No-op for the vLLM backend —
        vllm.LLM holds a pre-allocated KV-cache reservation that has no public
        release API short of tearing down the engine, which is why callers running
        inside an in-training callback should pass backend="transformers".
        """
        if self._model is None or self._resolved_backend != "transformers":
            return
        self._model = self._model.to("cpu")
        th.cuda.empty_cache()

    def _load_transformers(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_dtype = {"bfloat16": th.bfloat16, "float16": th.float16}.get(self.dtype, th.bfloat16)
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        # Left-pad tokenizer so batched generate() advances from the rightmost
        # real token of every sequence (decoder-only models can't continue from
        # right-padded inputs). Single-image calls are unaffected because
        # padding=True is a no-op for a batch of 1.
        self._processor.tokenizer.padding_side = "left"
        if self._processor.tokenizer.pad_token_id is None:
            self._processor.tokenizer.pad_token_id = self._processor.tokenizer.eos_token_id
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()

    def _load_vllm(self):
        _isolate_env_for_vllm()
        from transformers import AutoProcessor
        from vllm import LLM

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._processor.tokenizer.padding_side = "left"
        self._model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={"image": 1},
            # Defaults True on this image (torch 2.7 nvidia + vllm 0.10/0.11.dev).
            # Rationale:
            # (1) 0.11.3.dev's torch.compile backend is incompatible with torch 2.7
            #     (`VllmBackend.__call__() got 'options'`); 0.10.2.dev's FlashInfer
            #     decode path hits a `trtllm_paged_attention_decode` arg-type mismatch
            #     unless VLLM_ATTENTION_BACKEND=FLASH_ATTN is set.
            # (2) Even with that workaround, CUDA graphs don't reliably speed up
            #     sampling-based VLMs because FlashInfer's sampler can't graph per-
            #     request seeds and falls back to PyTorch-native per-step kernels.
            # Toggle via the `enforce_eager=False` constructor kwarg when benchmarking.
            enforce_eager=self.enforce_eager,
        )

    def _build_chat_text(self) -> str:
        """Build the templated prompt string used for every image (image is a placeholder)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VQR1_QUESTION_TEMPLATE.format(Question=VQR1_PROMPT)},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """Return a float in [1, 5] (clamped); NaN if parsing fails entirely."""
        self._load_model()
        pil = _resize_for_vlm(_to_numpy_hwc(pred), self.max_side)
        text = self._build_chat_text()
        if self._resolved_backend == "vllm":
            response = self._generate_vllm(text, pil)
        else:
            response = self._generate_transformers(text, pil)
        return self._parse_score(response)

    def _generate_transformers(self, text: str, pil) -> str:
        return self._generate_transformers_batch([text], [pil])[0]

    def _generate_transformers_batch(self, texts: List[str], pils: List) -> List[str]:
        """Single batched HF generate() over N images. Greedy decoding is
        deterministic per-input regardless of batch size, so this is numerically
        equivalent to a per-image loop modulo small fp reduction-order noise from
        flash-attn batching (verified to produce identical parsed scores in
        test/quality_metrics/verify_vqr1_transformers_batch.py).
        """
        inputs = self._processor(text=texts, images=pils, return_tensors="pt", padding=True).to(self.device)
        with th.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        # Left padding makes inputs.input_ids.shape[1] uniform across the batch,
        # so slicing past the (padded) prompt cleanly yields only generated tokens.
        prompt_len = inputs.input_ids.shape[1]
        trimmed = [o[prompt_len:] for o in out_ids]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _generate_vllm(self, text: str, pil) -> str:
        from vllm import SamplingParams

        inputs = [{"prompt": text, "multi_modal_data": {"image": pil}}]
        outputs = self._model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                stop_token_ids=[self._processor.tokenizer.eos_token_id],
            ),
        )
        return outputs[0].outputs[0].text

    def _generate_vllm_batch(self, texts: List[str], pils: List) -> List[str]:
        """Real vLLM continuous batching — single `LLM.generate([...])` over N requests."""
        from vllm import SamplingParams

        inputs = [{"prompt": t, "multi_modal_data": {"image": p}} for t, p in zip(texts, pils)]
        outputs = self._model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                stop_token_ids=[self._processor.tokenizer.eos_token_id],
            ),
        )
        # vLLM preserves input order; idx(out) == idx(input).
        return [o.outputs[0].text for o in outputs]

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract score from '<answer>X.XX</answer>'; fall back to last number in
        the response. Clamps to [1, 5]. Returns NaN on total parse failure (so it
        pollutes the all-reduce rather than silently biasing averages).
        """
        try:
            answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_matches:
                num_match = re.search(r"\d+(\.\d+)?", answer_matches[-1])
                if num_match:
                    return max(1.0, min(5.0, float(num_match.group())))

            numbers = re.findall(r"\d+\.\d+|\d+", text)
            if numbers:
                score = float(numbers[-1])
                if 1.0 <= score <= 5.0:
                    return score
        except Exception:
            pass
        return float("nan")

    def compute_batch_list(self, preds, targets=None) -> List[float]:
        """Score a list of images. Both backends do real batching now: vLLM via
        continuous batching, transformers via left-padded HF generate().
        """
        self._load_model()
        pils = [_resize_for_vlm(_to_numpy_hwc(p), self.max_side) for p in preds]
        text = self._build_chat_text()  # prompt is identical for every image
        if self._resolved_backend == "vllm":
            responses = self._generate_vllm_batch([text] * len(pils), pils)
        else:
            responses = self._generate_transformers_batch([text] * len(pils), pils)
        return [self._parse_score(r) for r in responses]

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """(T, H, W, C) frame batch -> list of per-frame scores. Kept for parity
        with video-eval callers; delegates to compute_batch_list.
        """
        return self.compute_batch_list([pred[i] for i in range(pred.shape[0])])


# -----------------------------------------------------------------------------
# Q-Insight — ByteDance's RL-finetuned Qwen2.5-VL IQA model
# -----------------------------------------------------------------------------
# Prompts and generation hyper-params are copied verbatim from the official
# demo at https://github.com/bytedance/Q-Insight/blob/main/src/eval/demo_score.py
# so reported scores stay comparable to the paper's benchmarks. Note the use
# of stochastic sampling (temperature=1.0, top_k=50, top_p=0.95) — this is
# intentional, and is what gives Q-Insight fine-grained score discrimination
# that greedy generic-VLM prompting failed to deliver. `set_seed()` is called
# before each generate so per-image scores are reproducible across runs.

QINSIGHT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning "
    "process here </think><answer> answer here </answer>"
)
QINSIGHT_SCORE_QUESTION_PROMPT = (
    "What is your overall rating on the quality of this picture? The rating should be a float "
    "between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and "
    "5 representing excellent quality. Return the final answer in JSON format with the "
    'following keys: "rating": The score.'
)
QINSIGHT_TEMPLATE_SUFFIX = (
    "First output the thinking process in <think> </think> tags and then output the final "
    "answer in <answer> </answer> tags. Output the final answer in JSON format."
)


@MetricRegistry.register("q_insight")
class QInsight(BaseMetric):
    """Q-Insight — ByteDance's RL-finetuned Qwen2.5-VL IQA model, NR, higher is better, 1-5.

    Uses the 'score_degradation' checkpoint subfolder of the ByteDance/Q-Insight HF repo.
    Unlike qwen3vl / visualquality_r1 this model is RL-trained specifically for IQA and
    uses sampling decoding — the combination gives much finer score discrimination than
    greedy-decoded generic VLMs. Sampling is seeded per compute() call so scores are
    reproducible.

    Backends: "auto" / "transformers" / "vllm" (same contract as VisualQualityR1).
    Reference: https://github.com/bytedance/Q-Insight
    """

    DEFAULT_MODEL = "ByteDance/Q-Insight"
    DEFAULT_SUBFOLDER = "score_degradation"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = DEFAULT_MODEL,
        subfolder: str = DEFAULT_SUBFOLDER,
        max_side: int = 512,
        max_new_tokens: int = 1024,
        dtype: str = "bfloat16",
        backend: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        seed: int = 42,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        enforce_eager: bool = False,
        max_model_len: int = 32768,
    ):
        super().__init__(name="Q-Insight", device=device)
        self.model_name = model_name
        self.subfolder = subfolder
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.backend = backend
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.temperature = temperature
        self.enforce_eager = enforce_eager
        self.top_k = top_k
        self.top_p = top_p
        # See VisualQualityR1.__init__ for rationale — cap vLLM's KV-cache seq-len
        # ceiling so a short-prompt IQA workload doesn't provision for 128k context.
        self.max_model_len = max_model_len
        self._resolved_backend: str = ""
        self._model = None
        self._processor = None
        self._gen_config = None  # transformers GenerationConfig, built in _load_transformers

    def _load_model(self):
        if self._model is not None:
            # transformers backend only: re-promote from CPU if previously offloaded.
            if self._resolved_backend == "transformers" and next(self._model.parameters()).device.type == "cpu":
                self._model = self._model.to(self.device)
            return
        self._resolved_backend = _resolve_vlm_backend(self.backend)
        if self._resolved_backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def offload_to_cpu(self):
        """Move the transformers model to CPU between evals so it doesn't squat on
        GPU memory during the next training chunk. No-op for the vLLM backend —
        vllm.LLM holds a pre-allocated KV-cache reservation that has no public
        release API short of tearing down the engine, which is why callers running
        inside an in-training callback should pass backend="transformers".
        """
        if self._model is None or self._resolved_backend != "transformers":
            return
        self._model = self._model.to("cpu")
        th.cuda.empty_cache()

    def _load_transformers(self):
        from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

        torch_dtype = {"bfloat16": th.bfloat16, "float16": th.float16}.get(self.dtype, th.bfloat16)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, subfolder=self.subfolder, trust_remote_code=True
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            subfolder=self.subfolder,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
        self._gen_config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )

    def _load_vllm(self):
        # vLLM's LLM(...) does not accept a subfolder kwarg portably, so snapshot-download
        # just the subfolder to the standard HF cache and point vllm at the resulting path.
        import os

        _isolate_env_for_vllm()
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor
        from vllm import LLM

        snapshot_dir = snapshot_download(repo_id=self.model_name, allow_patterns=[f"{self.subfolder}/*"])
        local_model_path = os.path.join(snapshot_dir, self.subfolder)

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, subfolder=self.subfolder, trust_remote_code=True
        )
        self._processor.tokenizer.padding_side = "left"
        self._model = LLM(
            model=local_model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={"image": 1},
            # Defaults True on this image (torch 2.7 nvidia + vllm 0.10/0.11.dev).
            # Rationale:
            # (1) 0.11.3.dev's torch.compile backend is incompatible with torch 2.7
            #     (`VllmBackend.__call__() got 'options'`); 0.10.2.dev's FlashInfer
            #     decode path hits a `trtllm_paged_attention_decode` arg-type mismatch
            #     unless VLLM_ATTENTION_BACKEND=FLASH_ATTN is set.
            # (2) Even with that workaround, CUDA graphs don't reliably speed up
            #     sampling-based VLMs because FlashInfer's sampler can't graph per-
            #     request seeds and falls back to PyTorch-native per-step kernels.
            # Toggle via the `enforce_eager=False` constructor kwarg when benchmarking.
            enforce_eager=self.enforce_eager,
        )

    def _build_chat_text(self) -> str:
        """Build the Q-Insight prompt (same for every image; image is a placeholder)."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": QINSIGHT_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": QINSIGHT_SCORE_QUESTION_PROMPT + " " + QINSIGHT_TEMPLATE_SUFFIX,
                    },
                    {"type": "image"},  # placeholder; PIL passed via images=[pil]
                ],
            },
        ]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        """Return a float in [1, 5] (clamped); NaN if parsing fails entirely."""
        self._load_model()
        pil = _resize_for_vlm(_to_numpy_hwc(pred), self.max_side)
        text = self._build_chat_text()
        if self._resolved_backend == "vllm":
            response = self._generate_vllm(text, pil)
        else:
            response = self._generate_transformers(text, pil)
        return self._parse_score(response)

    def _generate_transformers(self, text: str, pil) -> str:
        from transformers import set_seed

        # Per-compute seeding: sampling + fixed seed => per-image reproducibility.
        set_seed(self.seed)
        inputs = self._processor(text=[text], images=[pil], return_tensors="pt", padding=True).to(self.device)
        with th.no_grad():
            out_ids = self._model.generate(**inputs, generation_config=self._gen_config, use_cache=True)
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _generate_vllm(self, text: str, pil) -> str:
        from vllm import SamplingParams

        inputs = [{"prompt": text, "multi_modal_data": {"image": pil}}]
        outputs = self._model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                seed=self.seed,
                stop_token_ids=[self._processor.tokenizer.eos_token_id],
            ),
        )
        return outputs[0].outputs[0].text

    def _generate_vllm_batch(self, texts: List[str], pils: List) -> List[str]:
        """Real vLLM continuous batching — a single `LLM.generate([...])` over N requests.
        One shared seed is used (SamplingParams doesn't take per-request seeds when
        running a batch via synchronous LLM.generate). This is still deterministic
        for a given batch, just not per-index as in the single-image path.
        """
        from vllm import SamplingParams

        inputs = [{"prompt": t, "multi_modal_data": {"image": p}} for t, p in zip(texts, pils)]
        outputs = self._model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                seed=self.seed,
                stop_token_ids=[self._processor.tokenizer.eos_token_id],
            ),
        )
        return [o.outputs[0].text for o in outputs]

    @staticmethod
    def _parse_score(text: str) -> float:
        """Expected output: <think>...</think><answer>{"rating": 3.45}</answer>.
        Falls back to any number inside <answer>, then to last number in the whole response.
        Clamps to [1, 5]. Returns NaN on total parse failure.
        """
        # Primary path: <answer>{"rating": X.XX}</answer>
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            try:
                data = json.loads(inner)
                if isinstance(data, dict) and "rating" in data:
                    return max(1.0, min(5.0, float(data["rating"])))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            # Secondary path: any number inside <answer>
            num = re.search(r"\d+(?:\.\d+)?", inner)
            if num:
                try:
                    return max(1.0, min(5.0, float(num.group())))
                except ValueError:
                    pass

        # Tertiary fallback: last number in the whole response
        numbers = re.findall(r"\d+\.\d+|\d+", text)
        if numbers:
            try:
                score = float(numbers[-1])
                if 1.0 <= score <= 5.0:
                    return score
            except ValueError:
                pass

        return float("nan")

    def compute_batch_list(self, preds, targets=None) -> List[float]:
        """Score a list of images. vLLM path uses real continuous batching;
        transformers path is a per-image for-loop (maintains per-image seeding).
        """
        self._load_model()
        pils = [_resize_for_vlm(_to_numpy_hwc(p), self.max_side) for p in preds]
        text = self._build_chat_text()  # same prompt template for every image
        if self._resolved_backend == "vllm":
            responses = self._generate_vllm_batch([text] * len(pils), pils)
        else:
            responses = [self._generate_transformers(text, p) for p in pils]
        return [self._parse_score(r) for r in responses]

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        """(T, H, W, C) frame batch -> list of per-frame scores. Kept for parity
        with video-eval callers; delegates to compute_batch_list.
        """
        return self.compute_batch_list([pred[i] for i in range(pred.shape[0])])


# -----------------------------------------------------------------------------
# UniPercept — Shanghai-AI-Lab/USTC's RL-tuned InternVL-Chat perceptual scorer
# -----------------------------------------------------------------------------
# UniPercept (Cao et al., 2025; arXiv:2512.21675) exposes one VLM that scores
# three perceptual aspects through a 101-token softmax-weighted prediction
# head: aesthetics (IAA), quality (IQA), structure-and-texture richness (ISTA).
# All three share the same forward path — only the prompt's `desc` word
# changes — so we register three lightweight BaseMetric subclasses backed by a
# single module-level engine singleton; instantiating two or three of them
# costs one ~14-20 GB model on GPU instead of N copies.
#
# Reference: official scoring path at UniPercept/src/eval/eval_vr.py and
# InternVLChatModel.score() in src/internvl/model/internvl_chat/modeling_unipercept.py.
# Output range is [0, 100], higher is better.

UNIPERCEPT_DEFAULT_MODEL = "Thunderbolt215215/UniPercept"
UNIPERCEPT_INPUT_SIZE = 448  # InternVL-Chat training resolution; do not change
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# UniPercept maps a 0-100 score to a two-letter token from this 101-entry list,
# trained as the model's preferential-output vocab. Inlined from
# UniPercept/src/internvl/model/internvl_chat/aes_tokens.py to avoid a sys.path
# dep on the local repo (the HF Hub revision ships modeling_internvl_chat.py +
# conversation.py but not this file).
_UNIPERCEPT_TOKEN_LIST: List[str] = (
    [chr(ord("a") + 0) + chr(ord("a") + i) for i in range(26)]  # 0-25:  aa..az
    + [chr(ord("a") + 2) + chr(ord("a") + i) for i in range(25)]  # 26-50: ca..cy
    + [chr(ord("a") + 3) + chr(ord("a") + i) for i in range(25)]  # 51-75: da..dy
    + [chr(ord("a") + 4) + chr(ord("a") + i) for i in range(25)]  # 76-100: ea..ey
)
assert len(_UNIPERCEPT_TOKEN_LIST) == 101, "UniPercept token list must have 101 entries"


class _UniPerceptEngine:
    """Module-private singleton wrapping InternVLChatModel + tokenizer.

    Loaded once per process; shared across UniPerceptIAA / UniPerceptIQA /
    UniPerceptISTA so the three metrics together still cost one model on GPU.

    The HF Hub revision (`Thunderbolt215215/UniPercept`) ships an
    `InternVLChatModel` (via trust_remote_code) that has `chat()` / `generate()`
    but no `score()`. Rather than monkey-patch a method onto the loaded class,
    we reimplement the score forward inline in `score_one()` against the model's
    public methods (`extract_feature`, `language_model.get_input_embeddings()`,
    direct `language_model(...)` LM-head call) — numerically identical to the
    upstream `score()` in UniPercept/src/internvl/model/internvl_chat/modeling_unipercept.py:413-475.
    """

    _instance: Optional["_UniPerceptEngine"] = None

    @classmethod
    def get(cls, device: str, model_name: str, dtype: str) -> "_UniPerceptEngine":
        # First caller wins config; subsequent callers reuse, even if their
        # kwargs differ (matches the existing MetricRegistry singleton flavor —
        # if you really need a second config you'd have to drop the cache).
        if cls._instance is None:
            cls._instance = _UniPerceptEngine(device=device, model_name=model_name, dtype=dtype)
        return cls._instance

    def __init__(self, device: str, model_name: str, dtype: str):
        self.device = device
        self.model_name = model_name
        self.dtype = dtype
        self._torch_dtype = {"bfloat16": th.bfloat16, "float16": th.float16, "float32": th.float32}.get(
            dtype, th.bfloat16
        )
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._get_conv_template = None
        # Cached per-tokenizer: token id of <IMG_CONTEXT> and the 101-entry
        # softmax-target id list. Both are tokenizer-stable so we resolve once.
        self._img_context_token_id: Optional[int] = None
        self._preferential_ids: Optional[List[int]] = None

    def _build_transform(self):
        # Mirrors UniPercept/src/eval/eval_vr.py:23-29 verbatim — bicubic resize
        # to 448×448 then ImageNet normalize. The model's `score()` head was
        # trained on this exact pipeline so we don't deviate.
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((UNIPERCEPT_INPUT_SIZE, UNIPERCEPT_INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _ensure_loaded(self):
        if self._model is not None:
            # Re-promote from CPU if previously offloaded.
            if next(self._model.parameters()).device.type == "cpu":
                self._model = self._model.to(self.device)
            return
        import sys

        from transformers import AutoModel, AutoTokenizer

        self._model = (
            AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        # Reach into the trust_remote_code-loaded module to grab `get_conv_template`.
        # The HF dynamic-module loader registers each repo's modeling code as
        # `transformers_modules.<hash>.modeling_internvl_chat`; sibling modules
        # (here, conversation.py) are imported by that file at load time and live
        # in the same package namespace.
        self._get_conv_template = sys.modules[type(self._model).__module__].get_conv_template

        self._img_context_token_id = self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._preferential_ids = [self._tokenizer.convert_tokens_to_ids(w) for w in _UNIPERCEPT_TOKEN_LIST]
        if self._transform is None:
            self._build_transform()

    def offload_to_cpu(self):
        """Move model to CPU between evals (mirrors VisualQualityR1.offload_to_cpu)."""
        if self._model is None:
            return
        self._model = self._model.to("cpu")
        th.cuda.empty_cache()

    def _to_pixel_values(self, pil) -> th.Tensor:
        """PIL.Image -> (1, 3, 448, 448) tensor on device, in self._torch_dtype."""
        return self._transform(pil).unsqueeze(0).to(self._torch_dtype).to(self.device)

    @staticmethod
    def _build_question(desc: str) -> str:
        # Verbatim from UniPercept's score() prompt (modeling_unipercept.py:426).
        return (
            f"<image>Rate the {desc} score of the image in 0-100. In the output format, "
            "numbers are replaced by 2 corresponding letters, and the mapping relationship "
            "is: score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az, \n"
            "score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy, \n"
            "score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy, \n"
            "score 76 to 100: 76-ea, 77-eb, 73-ec, 74-ed, ..., 100-ey. \n"
            "The answer only outputs 2 corresponding letters."
        )

    @th.no_grad()
    def score_one(self, pil, desc: str) -> float:
        """Single-image score. Returns float in [0, 100], higher is better.

        Behaviorally identical to the upstream `InternVLChatModel.score()` in
        modeling_unipercept.py:413-475. We inline the prompt build + LLM-head
        forward here so we don't need the local UniPercept repo on sys.path.
        """
        self._ensure_loaded()
        pixel_values = self._to_pixel_values(pil)
        model = self._model
        tokenizer = self._tokenizer
        device = self.device

        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        question = self._build_question(desc)

        # The HF model expects this attribute set before forward (see how its
        # `chat()`/`generate()` methods do it). Cached so successive calls don't
        # re-resolve the token id.
        model.img_context_token_id = self._img_context_token_id

        # Build a fresh conversation template for this call, matching upstream.
        template = self._get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        num_patches = pixel_values.shape[0]
        image_tokens = IMG_START + IMG_CTX * model.num_image_token * num_patches + IMG_END
        query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        # Inline of generate_logits (modeling_unipercept.py:520-559): one LLM
        # forward, no autoregressive generation, take last-position logits.
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = flat_ids == self._img_context_token_id
        assert selected.sum() != 0, "no <IMG_CONTEXT> tokens found in prompt"
        flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device, flat_embeds.dtype)
        input_embeds = flat_embeds.reshape(B, N, C)

        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Last-token logits at the answer slot, gather over the 101 score tokens,
        # softmax-weighted average to a 0-100 float (matches upstream exactly).
        last_logits = outputs.logits[:, -1, :].detach()
        output_logits = last_logits[:, self._preferential_ids]
        weight = th.arange(101, device=device, dtype=output_logits.dtype)
        score = th.softmax(output_logits, dim=-1) @ weight
        return float(score.item())

    def score_batch(self, pils: List, desc: str) -> List[float]:
        # Per-image loop matches the upstream eval_vr.py reference (single-image
        # forward per call) and gives identical scores. True batched-prompt
        # forward over N tiled prompts is a follow-up optimization.
        self._ensure_loaded()
        return [self.score_one(p, desc) for p in pils]


class _UniPerceptMetricBase(BaseMetric):
    """Shared scaffolding for the three UniPercept aspect metrics.

    Subclasses set DESC (the `desc` argument to InternVLChatModel.score). All
    three share one `_UniPerceptEngine` singleton, so registering more than
    one of them does NOT load extra model copies.
    """

    DESC: str = ""  # subclass override

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = UNIPERCEPT_DEFAULT_MODEL,
        dtype: str = "bfloat16",
        name: str = "UniPercept",
    ):
        super().__init__(name=name, device=device)
        self.model_name = model_name
        self.dtype = dtype

    def _engine(self) -> _UniPerceptEngine:
        return _UniPerceptEngine.get(self.device, self.model_name, self.dtype)

    def offload_to_cpu(self):
        # Every UniPercept* metric shares one engine, so any one of them can
        # offload. The callback pattern (one offload per metric per eval run)
        # is therefore over-eager but harmless — the second call is a no-op
        # because the model is already on CPU.
        if _UniPerceptEngine._instance is not None:
            _UniPerceptEngine._instance.offload_to_cpu()

    def _to_pil(self, img):
        from PIL import Image as PILImage

        return PILImage.fromarray(_to_numpy_hwc(img))

    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor] = None,
    ) -> float:
        return self._engine().score_one(self._to_pil(pred), self.DESC)

    def compute_batch(
        self,
        pred: np.ndarray,
        target: np.ndarray = None,
        batch_size: int = None,
    ) -> List[float]:
        # batch_size is accepted for API parity with FR metrics; we don't
        # micro-batch internally because score_batch is already a per-image
        # loop. compute_batch_list / compute_batch are 1:1 with input frames.
        pils = [self._to_pil(pred[i]) for i in range(pred.shape[0])]
        return self._engine().score_batch(pils, self.DESC)

    def compute_batch_list(self, preds, targets=None) -> List[float]:
        pils = [self._to_pil(p) for p in preds]
        return self._engine().score_batch(pils, self.DESC)


@MetricRegistry.register("unipercept_iaa")
class UniPerceptIAA(_UniPerceptMetricBase):
    """UniPercept Image Aesthetics Assessment — NR, higher is better, [0, 100]."""

    DESC = "aesthetics"

    def __init__(self, device: str = "cuda", model_name: str = UNIPERCEPT_DEFAULT_MODEL, dtype: str = "bfloat16"):
        super().__init__(device=device, model_name=model_name, dtype=dtype, name="UniPercept-IAA")


@MetricRegistry.register("unipercept_iqa")
class UniPerceptIQA(_UniPerceptMetricBase):
    """UniPercept Image Quality Assessment — NR, higher is better, [0, 100]."""

    DESC = "quality"

    def __init__(self, device: str = "cuda", model_name: str = UNIPERCEPT_DEFAULT_MODEL, dtype: str = "bfloat16"):
        super().__init__(device=device, model_name=model_name, dtype=dtype, name="UniPercept-IQA")


@MetricRegistry.register("unipercept_ista")
class UniPerceptISTA(_UniPerceptMetricBase):
    """UniPercept Structure & Texture Assessment — NR, higher is better, [0, 100]."""

    DESC = "structure and texture richness"

    def __init__(self, device: str = "cuda", model_name: str = UNIPERCEPT_DEFAULT_MODEL, dtype: str = "bfloat16"):
        super().__init__(device=device, model_name=model_name, dtype=dtype, name="UniPercept-ISTA")


class VideoMetricWrapper:
    """Wrapper to compute image metrics on video frames."""

    def __init__(self, metric: BaseMetric):
        """
        Initialize video metric wrapper.

        Args:
            metric: Base image metric to use
        """
        self.metric = metric
        self._frame_values: list = []

    def compute_video(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        """
        Compute metric over all video frames and return average.

        Args:
            pred: Predicted video, shape (T, H, W, C) for numpy or (C, T, H, W) for torch
            target: Ground truth video, same shape as pred

        Returns:
            Average metric value across all frames
        """
        if isinstance(pred, th.Tensor):
            # (C, T, H, W) -> (T, H, W, C)
            pred = pred.permute(1, 2, 3, 0)
            target = target.permute(1, 2, 3, 0)

        if isinstance(pred, th.Tensor):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()

        num_frames = pred.shape[0]
        frame_values = []

        for t in range(num_frames):
            value = self.metric.compute(pred[t], target[t])
            frame_values.append(value)
            self.metric.update(pred[t], target[t])

        self._frame_values = frame_values
        return float(np.mean(frame_values))

    def get_frame_values(self) -> list:
        """Get per-frame metric values from last video computation."""
        return self._frame_values

    def reset(self):
        """Reset the underlying metric."""
        self.metric.reset()
        self._frame_values = []
