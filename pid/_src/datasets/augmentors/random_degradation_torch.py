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
import logging
import platform
import random
from typing import Optional

# torchvision.io.encode_jpeg / decode_jpeg segfaults on aarch64 (GB200). Fall back to PIL there.
_JPEG_USE_PIL = platform.machine() == "aarch64"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
from PIL import Image

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._src.datasets.augmentors import blur_kernels

try:
    import av

    has_av = True
except ImportError:
    has_av = False


class UnsharpMasking(Augmentor):
    """Apply unsharp masking to an image or a sequence of images.

    Args:
        input_keys (list): The keys whose values are processed.
        output_keys (list): Not used, will add "_unsharp" suffix to input keys.
        args (dict): Should contain:
            - kernel_size (int): The kernel_size of the Gaussian kernel (must be odd).
            - sigma (float): The standard deviation of the Gaussian.
            - weight (float): The weight of the "details" in the final output.
            - threshold (float): Pixel differences larger than this value are regarded as "details".

    Added keys are "xxx_unsharp", where "xxx" are the attributes specified in "input_keys".
    """

    def __init__(self, input_keys: list, output_keys: list, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        params = args if args is not None else {}
        self.kernel_size = params.get("kernel_size", 51)
        self.sigma = params.get("sigma", 0)
        self.weight = params.get("weight", 0.5)
        self.threshold = params.get("threshold", 10)
        self.input_keys = input_keys
        self.output_keys = output_keys

        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number, but got {}.".format(self.kernel_size))

        # only generate 1D Gaussian kernel, not generate huge 2D matrix
        # cv2 返回的是 (K, 1) 的 float64 numpy 数组
        kernel_1d = cv2.getGaussianKernel(self.kernel_size, self.sigma)

        # convert to Tensor and transpose to (1, 1, 1, K) for broadcasting
        self.kernel_1d = torch.from_numpy(kernel_1d.T).float()  # Shape: (1, 1, 1, K)
        self.padding = self.kernel_size // 2
        self.input_normalized_range = args.get("input_normalized_range", False)

    def _gaussian_blur(self, x, kernel_1d):
        """Helper function to perform separable gaussian blur."""
        b, c, h, w = x.shape

        k_h = kernel_1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        k_v = kernel_1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

        # Step 1: Horizontal Conv (1 x K)
        # groups=c for depthwise convolution
        x_h = F.conv2d(x, k_h, padding=(0, self.padding), groups=c)

        # Step 2: Vertical Conv (K x 1)
        output = F.conv2d(x_h, k_v, padding=(self.padding, 0), groups=c)

        return output

    def _unsharp_masking(self, x_nchw):
        """Unsharp masking function."""
        is_normalized_range = self.input_normalized_range
        clamp_min = -1.0 if is_normalized_range else 0.0
        clamp_max = 1.0 if is_normalized_range else 1.0

        kernel_1d = self.kernel_1d.to(device=x_nchw.device, dtype=x_nchw.dtype)

        blurred = self._gaussian_blur(x_nchw, kernel_1d)

        residue = x_nchw - blurred

        # For normalized range [-1, 1], the value range is 2.0 instead of 255.0
        # So we need to scale threshold accordingly: th = threshold / 255.0 for [0,1] range
        # For [-1,1] range: residue_norm = residue_255 / 127.5, so th should be threshold / 127.5
        # This is equivalent to (threshold / 255.0) * 2.0 = threshold / 127.5
        value_range = 127.5 if is_normalized_range else 255.0
        th = float(self.threshold) / value_range
        mask = (residue.abs() > th).to(x_nchw.dtype)

        soft_mask = self._gaussian_blur(mask, kernel_1d)
        soft_mask = torch.clamp(soft_mask, clamp_min, clamp_max)

        sharpened = torch.clamp(x_nchw + self.weight * residue, clamp_min, clamp_max)
        outputs = soft_mask * sharpened + (1.0 - soft_mask) * x_nchw

        return outputs

    def __call__(self, data_dict: dict) -> dict:
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            if in_key in data_dict:
                data_dict[out_key] = self._unsharp_masking(data_dict.pop(in_key))
        return data_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}, weight={self.weight}, "
            f"threshold={self.threshold}, input_keys={self.input_keys}, output_keys={self.output_keys})"
        )


class RandomBlur(Augmentor):
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}

    def get_kernel(self, num_kernels: int):
        """This is the function to create kernel.

        Args:
            num_kernels (int): the number of kernels

        Returns:
            _type_: _description_
        """
        kernel_type = np.random.choice(self.params["kernel_list"], p=self.params["kernel_prob"])
        kernel_size = random.choice(self.params["kernel_size"])

        sigma_x_range = self.params.get("sigma_x", [0, 0])
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_x_step = self.params.get("sigma_x_step", 0)

        sigma_y_range = self.params.get("sigma_y", [0, 0])
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        sigma_y_step = self.params.get("sigma_y_step", 0)

        rotate_angle_range = self.params.get("rotate_angle", [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0], rotate_angle_range[1])
        rotate_angle_step = self.params.get("rotate_angle_step", 0)

        beta_gau_range = self.params.get("beta_gaussian", [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get("beta_gaussian_step", 0)

        beta_pla_range = self.params.get("beta_plateau", [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get("beta_plateau_step", 0)

        omega_range = self.params.get("omega", None)
        omega_step = self.params.get("omega_step", 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3.0, np.pi]
            else:
                omega_range = [np.pi / 5.0, np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = blur_kernels.random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma_x, sigma_x],
                [sigma_y, sigma_y],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)

            # update kernel parameters
            sigma_x += np.random.uniform(-sigma_x_step, sigma_x_step)
            sigma_y += np.random.uniform(-sigma_y_step, sigma_y_step)
            rotate_angle += np.random.uniform(-rotate_angle_step, rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma_x = np.clip(sigma_x, sigma_x_range[0], sigma_x_range[1])
            sigma_y = np.clip(sigma_y, sigma_y_range[0], sigma_y_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0], rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels

    def _apply_random_blur(self, x_nchw):
        """Apply blur with a shared kernel across the batch for speed."""

        # [Safety Fix 1] Make sure input is contiguous to avoid CUDA out-of-bounds due to stride issues
        if not x_nchw.is_contiguous():
            x_nchw = x_nchw.contiguous()

        # Generate Kernel
        kernel_np = self.get_kernel(num_kernels=1)[0]
        kernel = torch.as_tensor(kernel_np, dtype=x_nchw.dtype, device=x_nchw.device)

        # [Safety Fix 2] Check if kernel contains NaNs to prevent data corruption
        if torch.isnan(kernel).any() or torch.isinf(kernel).any():
            # Fallback: If the generated kernel is broken, return the input or raise an error
            print("Warning: Generated blur kernel contains NaN. Skipping blur.")
            return x_nchw

        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1,1,k,k
        # Ensure kernel channel count matches input
        num_channels = x_nchw.shape[1]
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        padding = kernel.shape[-1] // 2

        num_frames = x_nchw.shape[0]
        kernel_size = kernel.shape[-1]

        # Estimate whether or not to chunk
        is_large_input = num_frames > 80 and x_nchw.shape[2] * x_nchw.shape[3] > 500 * 800
        is_large_kernel = kernel_size >= 15

        if is_large_input and is_large_kernel:
            chunk_size = 25
            # [Performance Optimization] Pre-allocate output memory to avoid memory spikes from cat at the end
            output = torch.empty_like(x_nchw)

            for i in range(0, num_frames, chunk_size):
                end_idx = min(i + chunk_size, num_frames)

                # [Key Fix] Even for slices, best to call contiguous, or just pass to conv2d directly
                # Note: conv2d can handle non-contiguous input, but contiguous is safer
                chunk = x_nchw[i:end_idx].contiguous()

                # Execute convolution
                # Use try-catch only to catch OOM error for this small step, not for illegal accesses
                try:
                    out_chunk = F.conv2d(chunk, kernel, padding=padding, groups=num_channels)
                    output[i:end_idx] = out_chunk
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        raise e
                    else:
                        raise e
            return output
        else:
            # Normal processing
            return F.conv2d(x_nchw, kernel, padding=padding, groups=num_channels)

    def __call__(self, data_dict):
        """Call this augmentor."""
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                try:
                    input_tensor = data_dict[key].contiguous()
                    data_dict[key] = self._apply_random_blur(input_tensor)
                    torch.cuda.synchronize()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: OOM triggered in RandomBlur for key {key}, skipping blur.")
                        # Clear cache to prevent impact on subsequent steps
                        torch.cuda.empty_cache()
                        return data_dict
                    else:
                        # If it's illegal memory access, it must crash, not swallow it!
                        print(f"CRITICAL ERROR in RandomBlur: {str(e)}")
                        raise e

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomJPEGCompression(Augmentor):
    """Apply random JPEG compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
            quality: [min, max]
            quality_step: step size for quality
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}
        self.input_normalized_range = args.get("input_normalized_range", False)

    @staticmethod
    def _jpeg_roundtrip_pil(frame_chw_uint8: torch.Tensor, quality: int, device) -> torch.Tensor:
        # PIL-based JPEG encode/decode. torchvision.io.encode_jpeg segfaults on aarch64 (GB200).
        np_hwc = frame_chw_uint8.permute(1, 2, 0).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(np_hwc, "RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        result = np.array(Image.open(buf).convert("RGB"))  # [H, W, C] uint8
        return torch.from_numpy(result).to(device).permute(2, 0, 1)  # [C, H, W]

    def _apply_random_compression(self, x_nchw):
        # Record input dtype and value range
        input_dtype = x_nchw.dtype
        is_normalized_range = self.input_normalized_range

        # Convert to [0, 1] range if needed
        if is_normalized_range:
            # Input is in [-1, 1] range, convert to [0, 1]
            x_work = (x_nchw + 1.0) / 2.0
            x_work = torch.clamp(x_work, 0.0, 1.0)
        else:
            # Input is already in [0, 1] range
            x_work = x_nchw

        quality = self.params["quality"]
        quality_step = self.params.get("quality_step", 0)

        # Convert to uint8 [0, 255]
        x_work_uint8 = (x_work * 255).to(torch.uint8)

        N = x_work.shape[0]

        # Generate qualities for each frame
        qualities = []
        curr_quality = round(np.random.uniform(quality[0], quality[1]))
        for _ in range(N):
            qualities.append(curr_quality)
            curr_quality += np.random.uniform(-quality_step, quality_step)
            curr_quality = round(np.clip(curr_quality, quality[0], quality[1]))

        if _JPEG_USE_PIL:
            # aarch64 (GB200): torchvision.io.encode_jpeg segfaults, use PIL instead.
            outputs_list = []
            for i in range(N):
                decoded = self._jpeg_roundtrip_pil(x_work_uint8[i], qualities[i], x_work.device)
                outputs_list.append(decoded)
            outputs = torch.stack(outputs_list)
        else:
            # x86_64: use torchvision NVJPEG path for GPU-accelerated decode.
            outputs = None
            if all(q == qualities[0] for q in qualities):
                frames_list = [x_work_uint8[i] for i in range(N)]
                encoded_list = tv_io.encode_jpeg(frames_list, quality=qualities[0])
                lengths = [t.numel() for t in encoded_list]
                packed_cpu = torch.cat(encoded_list).cpu()
                encoded_list_cpu = list(packed_cpu.split(lengths))
                decoded_list = tv_io.decode_jpeg(encoded_list_cpu, device=str(x_work.device))
                outputs = torch.stack(decoded_list)
            if outputs is None:
                outputs_list = []
                for i in range(N):
                    frame = x_work_uint8[i]
                    encoded = tv_io.encode_jpeg(frame, quality=qualities[i]).cpu()
                    decoded = tv_io.decode_jpeg(encoded, device=str(x_work.device))
                    outputs_list.append(decoded)
                outputs = torch.stack(outputs_list)

        # Convert back to [0, 1] range with target dtype
        outputs = outputs.to(input_dtype) / 255.0

        # Convert back to original range if needed
        if is_normalized_range:
            # Convert from [0, 1] to [-1, 1]
            outputs = outputs * 2.0 - 1.0
            outputs = torch.clamp(outputs, -1.0, 1.0)

        return outputs

    def __call__(self, data_dict):
        """Call this augmentor."""
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_compression(data_dict[key])

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomNoise(Augmentor):
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}
        self.input_normalized_range = args.get("input_normalized_range", False)

    def _apply_gaussian_noise(self, x_nchw):
        """Apply gaussian noise in a single vectorized pass."""
        is_normalized_range = self.input_normalized_range
        clamp_min = -1.0 if is_normalized_range else 0.0
        clamp_max = 1.0 if is_normalized_range else 1.0

        sigma_range = self.params["gaussian_sigma"]
        sigma = float(np.random.uniform(sigma_range[0], sigma_range[1]))
        gray_noise_prob = self.params["gaussian_gray_noise_prob"]
        is_gray_noise = np.random.uniform() < gray_noise_prob

        # For normalized range [-1, 1], the value range is 2.0 instead of 1.0 (for [0,1])
        # To match mmagic behavior on [0,255] data:
        # - mmagic: noise_255 = randn() * sigma, added to [0,255] data
        # - here: noise_norm = randn() * sigma / 255.0, added to [0,1] or [-1,1] data
        # For [-1,1] range, we need sigma / 127.5 to get equivalent relative noise level
        # This means: sigma_here / 127.5 ≈ sigma_mmagic / 255 → use 127.5 for [-1,1]
        value_range = 127.5 if is_normalized_range else 255.0

        if is_gray_noise:
            noise = (
                torch.randn(
                    x_nchw.shape[0], 1, x_nchw.shape[2], x_nchw.shape[3], device=x_nchw.device, dtype=x_nchw.dtype
                )
                * sigma
                / value_range
            )
            noise = noise.repeat(1, x_nchw.shape[1], 1, 1)
        else:
            noise = torch.randn_like(x_nchw) * sigma / value_range

        output = torch.clamp(x_nchw + noise, clamp_min, clamp_max)
        return output

    def _apply_poisson_noise(self, x_nchw):
        """Ultra-fast Poisson noise with gray noise support (Fully Vectorized).

        Args:
            x_nchw: Input tensor with shape (N, C, H, W), float [0, 1]

        Returns:
            Output tensor with shape (N, C, H, W), float [0, 1]
        """
        device = x_nchw.device
        N, C, H, W = x_nchw.shape

        is_normalized_range = self.input_normalized_range
        clamp_min = -1.0 if is_normalized_range else 0.0
        clamp_max = 1.0 if is_normalized_range else 1.0

        # -------------------------------------------------------
        # 1. 极速生成 Scale 参数 (Vectorized)
        # -------------------------------------------------------
        scale_range = self.params["poisson_scale"]
        scale_step = self.params.get("poisson_scale_step", 0)

        # 初始 scale
        init_scale = np.random.uniform(scale_range[0], scale_range[1])

        if scale_step > 0:
            # 模拟 scale 的随机游走: scale[i] = scale[i-1] + random_step
            # 使用 cumsum 一次性算完 N 个
            steps = np.random.uniform(-scale_step, scale_step, size=(N,))
            steps[0] += init_scale  # 起始点偏移
            scales = np.cumsum(steps)
            scales = np.clip(scales, scale_range[0], scale_range[1])
        else:
            # 或者每张图随机一个不同的 scale
            scales = np.random.uniform(scale_range[0], scale_range[1], size=(N,))

        # 转为 Tensor: (N, 1, 1, 1) 用于广播
        scales_tensor = torch.as_tensor(scales, device=device, dtype=x_nchw.dtype).view(N, 1, 1, 1)

        # -------------------------------------------------------
        # 2. Determine if gray noise should be applied
        # -------------------------------------------------------
        gray_noise_prob = self.params.get("poisson_gray_noise_prob", 0)
        is_gray_noise = np.random.uniform() < gray_noise_prob

        # -------------------------------------------------------
        # 3. 核心计算 (Batch & Channel 并行)
        # -------------------------------------------------------

        # 转换域到 [0, 255]
        # 使用 clamp 确保数值安全
        if is_normalized_range:
            frame_255 = (x_nchw * 127.5 + 127.5).clamp(0, 255)
        else:
            frame_255 = torch.clamp(x_nchw * 255.0, 0, 255)

        # [优化] 移除 torch.unique，直接使用常数 255.0
        unique_val = 255.0

        if is_gray_noise:
            # Convert to grayscale for noise computation
            # Use standard RGB to grayscale conversion weights
            # Assuming input is RGB (first 3 channels)
            # noise_base shape: (N, 1, H, W)
            if C >= 3:
                # Standard RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
                weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=x_nchw.dtype).view(1, 3, 1, 1)
                noise_base = torch.sum(frame_255[:, :3, :, :] * weights, dim=1, keepdim=True)
            else:
                # If less than 3 channels, use mean
                noise_base = torch.mean(frame_255, dim=1, keepdim=True)
        else:
            # Use all channels for noise computation
            noise_base = frame_255

        # 1. Round: 模拟离散的光子计数基底
        noise_base_rounded = torch.clamp(noise_base.round(), 0, 255)

        # 2. Poisson Sampling
        # 公式: poisson(x * val) / val - x
        scaled_lambda = noise_base_rounded * unique_val
        poisson_samples = torch.poisson(scaled_lambda)
        noise = poisson_samples / unique_val - noise_base_rounded

        # If gray noise, broadcast to all channels
        if is_gray_noise:
            noise = noise.repeat(1, C, 1, 1)

        # 3. Apply Noise & Scale
        # output = input + noise * scale
        # For normalized range [-1, 1], the value range is 2.0 instead of 1.0 (for [0,1])
        # To match mmagic behavior: scale_here / 127.5 ≈ scale_mmagic / 255 → use 127.5 for [-1,1]
        value_range = 127.5 if is_normalized_range else 255.0
        output = x_nchw + noise * (scales_tensor / value_range)

        return torch.clamp(output, clamp_min, clamp_max)

    def _apply_random_noise(self, x_nchw):
        """This is the function used to apply random noise on images.

        Args:
            imgs (Tensor): training images

        Returns:
            _type_: _description_
        """
        noise_type = np.random.choice(self.params["noise_type"], p=self.params["noise_prob"])

        if noise_type.lower() == "gaussian":
            x_nchw = self._apply_gaussian_noise(x_nchw)
        elif noise_type.lower() == "poisson":
            x_nchw = self._apply_poisson_noise(x_nchw)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is not implemented.')
        return x_nchw

    def __call__(self, data_dict):
        """Call this augmentor."""
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_noise(data_dict[key])

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomResize(Augmentor):
    """Randomly resize the input.

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}

        self.resize_dict = dict(bilinear="bilinear", bicubic="bicubic", area="area")

        # If True, get target size from GT in data_dict and apply resize_scale
        self.get_resize_target_from_data_batch = self.params.get("get_resize_target_from_data_batch", False)
        self.gt_key_for_target = self.params.get("gt_key_for_target", "gt")

    def _random_resize(self, x_nchw, data_dict=None):
        """This is the function used to randomly resize images for training
        augmentation.

        Args:
            imgs (list): training images (list of numpy arrays).
            data_dict (dict): data dictionary to get GT size if needed.

        Returns:
            list: images after randomly resized
        """
        # Safety check 1: Check input tensor validity
        if torch.isnan(x_nchw).any():
            logging.warning(f"[RandomResize] Input tensor contains NaN values! Shape: {x_nchw.shape}")
            # Replace NaN with 0
            x_nchw = torch.nan_to_num(x_nchw, nan=0.0)

        if torch.isinf(x_nchw).any():
            logging.warning(f"[RandomResize] Input tensor contains Inf values! Shape: {x_nchw.shape}")
            # Replace Inf with large finite values
            x_nchw = torch.nan_to_num(x_nchw, posinf=1.0, neginf=0.0)

        h, w = x_nchw.shape[2], x_nchw.shape[3]

        resize_opt = self.params["resize_opt"]
        resize_prob = self.params["resize_prob"]
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f"resize_opt [{resize_opt}] is not implemented")
        resize_opt = self.resize_dict[resize_opt]

        # determine the target size
        target_size = self.params.get("target_size", None)

        # Get target size from GT in data_dict
        if self.get_resize_target_from_data_batch and data_dict is not None:
            gt_key = self.gt_key_for_target
            if gt_key in data_dict:
                gt_nchw = data_dict[gt_key]
                gt_h, gt_w = gt_nchw.shape[2], gt_nchw.shape[3]
                target_scale_from_gt = self.params["target_scale_from_gt"]
                target_h = int(gt_h * target_scale_from_gt[0])
                target_w = int(gt_w * target_scale_from_gt[1])
                target_size = (target_h, target_w)
            else:
                raise ValueError(f"GT key [{gt_key}] not found in data_dict")

        if target_size is None:
            resize_mode = np.random.choice(["up", "down", "keep"], p=self.params["resize_mode_prob"])
            resize_scale = self.params["resize_scale"]
            if resize_mode == "up":
                scale_factor = np.random.uniform(1, resize_scale[1])
            elif resize_mode == "down":
                scale_factor = np.random.uniform(resize_scale[0], 1)
            else:
                scale_factor = 1

            # determine output size
            h_out, w_out = h * scale_factor, w * scale_factor
            if self.params.get("is_size_even", False):
                h_out, w_out = 2 * (h_out // 2), 2 * (w_out // 2)
            target_size = (int(h_out), int(w_out))

        # Apply division_factor alignment if specified
        division_factor = self.params.get("division_factor", None)
        if division_factor is not None and division_factor > 1:
            target_h, target_w = target_size
            # Round down to nearest multiple of division_factor
            target_h = (target_h // division_factor) * division_factor
            target_w = (target_w // division_factor) * division_factor
            target_size = (target_h, target_w)

        # Safety check 2: Validate target_size
        min_size = 64  # Minimum reasonable size
        if target_size[0] < min_size or target_size[1] < min_size:
            logging.warning(
                f"[RandomResize] Target size too small: {target_size}, input size: ({h}, {w}). "
                f"Clamping to minimum size {min_size}."
            )
            target_size = (max(target_size[0], min_size), max(target_size[1], min_size))

        # Safety check 3: Log detailed info before interpolate
        try:
            # Additional check: ensure tensor is contiguous and properly allocated
            if not x_nchw.is_contiguous():
                x_nchw = x_nchw.contiguous()

            # PyTorch官方建议：antialias 选项仅支持 'bilinear' 和 'bicubic'，area 不需要额外AA且此参数设置会报错
            # 参考: https://pytorch.org/docs/2.7/interpolate.html
            if resize_opt in ["bilinear", "bicubic"]:
                outputs = F.interpolate(x_nchw, size=target_size, mode=resize_opt, align_corners=False, antialias=True)
            elif resize_opt == "area":
                # area 模式无需 align_corners 和 antialias，仅本身即可实现视觉友好的下采样
                outputs = F.interpolate(x_nchw, size=target_size, mode="area")
            else:
                # 其他模式: 保留最基础调用（未来兼容其它插值）
                outputs = F.interpolate(x_nchw, size=target_size, mode=resize_opt, antialias=True)

            # Safety check 4: Check output validity
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logging.error(
                    f"[RandomResize] Output contains NaN/Inf after interpolate! "
                    f"Input shape: {x_nchw.shape}, target_size: {target_size}, resize_opt: {resize_opt}"
                )
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)

            return outputs

        except RuntimeError as e:
            logging.error(
                f"[RandomResize] CUDA error in F.interpolate:\n"
                f"  Input shape: {x_nchw.shape}\n"
                f"  Input dtype: {x_nchw.dtype}\n"
                f"  Input device: {x_nchw.device}\n"
                f"  Input min/max: {x_nchw.min().item():.4f}/{x_nchw.max().item():.4f}\n"
                f"  Target size: {target_size}\n"
                f"  resize_opt: {resize_opt}\n"
                f"  Error: {str(e)}"
            )
            # Try to recover by returning a zero tensor of the target size
            logging.warning(f"[RandomResize] Attempting recovery by creating zero tensor")
            outputs = torch.zeros(
                (x_nchw.shape[0], x_nchw.shape[1], target_size[0], target_size[1]),
                dtype=x_nchw.dtype,
                device=x_nchw.device,
            )
            return outputs

    def __call__(self, data_dict):
        """Call this augmentor."""
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                # Pass data_dict to _random_resize so it can access GT if needed
                data_dict[key] = self._random_resize(data_dict[key], data_dict)

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomVideoCompression(Augmentor):
    """Apply random video compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert has_av, "Please install av to use video compression."

        self.keys = input_keys
        self.params = args if args is not None else {}
        self.skip_flag = args.get("skip_flag", "skip_video_compression")
        logging.getLogger("libav").setLevel(50)

        self.input_normalized_range = args.get("input_normalized_range", False)

    def _apply_random_compression_optim(self, x_nchw):
        """
        Optimized random compression with encoding preset optimization.

        性能优化版本：通过设置编码预设参数，可获得约3.38倍加速！

        Input: torch float [T,C,H,W] on GPU/CPU

        优化点：
        1. 使用 preset 参数（默认 ultrafast）加速编码，性能提升约3.38倍
        2. 使用 tune=zerolatency 进一步优化延迟
        3. 可选的硬件加速支持（通过配置启用）

        配置参数（在 self.params 中设置）：
        - preset: 编码预设，可选值 ultrafast/superfast/veryfast/faster/fast/medium (默认: ultrafast)
        - tune: 编码调优，推荐 zerolatency (默认: zerolatency)
        - use_hw_accel: 是否使用硬件加速 (默认: False)
        - hw_codec: 硬件编码器，如 h264_nvenc (仅当 use_hw_accel=True 时生效)

        Note on quality vs mmagic:
        - This implementation uses preset=ultrafast + tune=zerolatency for speed
        - mmagic uses preset=medium (default) with no tune setting
        - At the same bitrate, ultrafast+zerolatency produces ~25-50% lower quality than medium
        - To match mmagic quality, either increase bitrate by 1.5-2x, or set preset="medium", tune=None
        """
        # Record input dtype and value range
        input_dtype = x_nchw.dtype
        is_normalized_range = self.input_normalized_range

        # Convert to [0, 1] range if needed
        if is_normalized_range:
            # Input is in [-1, 1] range, convert to [0, 1]
            x_work = (x_nchw + 1.0) / 2.0
            x_work = torch.clamp(x_work, 0.0, 1.0)
        else:
            # Input is already in [0, 1] range
            x_work = x_nchw

        # 1. 准备参数
        use_hw_accel = self.params.get("use_hw_accel", False)

        if use_hw_accel:
            # 硬件加速模式
            codec = self.params.get("hw_codec", "h264_nvenc")
            preset_key = "preset_hw"
            tune_key = "tune_hw"
            default_preset = "p1"  # NVENC preset: p1-p7, p1 最快
            default_tune = "ll"  # 低延迟
        else:
            # 软件编码模式
            codec = random.choices(self.params["codec"], self.params["codec_prob"])[0]
            preset_key = "preset"
            tune_key = "tune"
            default_preset = "ultrafast"  # 默认最快预设，可获得约3.38倍加速
            default_tune = "zerolatency"

        preset = self.params.get(preset_key, default_preset)
        tune = self.params.get(tune_key, default_tune)

        bitrate = self.params["bitrate"]
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        T, C, H, W = x_work.shape

        # 2. 预处理：GPU 上转 uint8，减少 PCIe 传输量 (float32 -> uint8)
        # [T, C, H, W] -> [T, H, W, C]
        x_nhwc = x_work.permute(0, 2, 3, 1)
        # 保持在 GPU 上做数学运算，最后只传 uint8 到 CPU
        # For bfloat16, the precision is sufficient for uint8 conversion
        x_nhwc_uint8 = (x_nhwc.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        x_nhwc_np = x_nhwc_uint8.cpu().numpy()  # 此时传输的是 uint8，带宽节省 75%

        buf = io.BytesIO()

        # 3. 编码阶段（核心优化点）
        # 使用 "nut" 格式比 "mp4" 更适合内存流，因为它不需要回写文件头
        with av.open(buf, "w", format="nut") as container:
            stream = container.add_stream(codec, rate=1)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = bitrate

            stream.options = {
                "preset": preset,
                "tune": tune,
            }

            # 优化：直接处理，避免额外的 list 循环开销
            for img in x_nhwc_np:
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pict_type = 0
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        # 4. 解码阶段
        buf.seek(0)

        output_np = np.empty((T, H, W, 3), dtype=np.uint8)

        with av.open(buf, "r", format="nut") as container:
            if container.streams.video:
                stream = container.streams.video[0]

                # 使用 enumerate 直接填入预分配的数组
                for i, frame in enumerate(container.decode(stream)):
                    if i >= T:
                        break  # 安全检查
                    output_np[i] = frame.to_rgb().to_ndarray()

        # 5. 后处理：CPU (uint8) -> GPU (uint8) -> GPU (float with target dtype)
        outputs = torch.as_tensor(output_np, dtype=torch.uint8, device=x_nchw.device)

        # 在 GPU 上进行 float 转换和维度重排，使用目标 dtype
        outputs = outputs.permute(0, 3, 1, 2).to(input_dtype) / 255.0

        # Convert back to original range if needed
        if is_normalized_range:
            # Convert from [0, 1] to [-1, 1]
            outputs = outputs * 2.0 - 1.0
            outputs = torch.clamp(outputs, -1.0, 1.0)

        return outputs

    def __call__(self, data_dict):
        """Call this augmentor."""
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        if self.skip_flag in data_dict and data_dict[self.skip_flag]:
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_compression_optim(data_dict[key])

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str
