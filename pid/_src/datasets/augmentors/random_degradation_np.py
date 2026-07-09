# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import random
from typing import Optional

import cv2
import numpy as np

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._src.datasets.augmentors import blur_kernels

try:
    import av

    has_av = True
except ImportError:
    has_av = False


class UnsharpMasking(Augmentor):
    """Apply unsharp masking to an image or a sequence of images (NumPy version).

    Args:
        input_keys (list): The keys whose values are processed.
        output_keys (list): Not used, will add "_unsharp" suffix to input keys.
        args (dict): Should contain:
            - kernel_size (int): The kernel_size of the Gaussian kernel (must be odd).
            - sigma (float): The standard deviation of the Gaussian.
            - weight (float): The weight of the "details" in the final output.
            - threshold (float): Pixel differences larger than this value are regarded as "details".

    Added keys are "xxx_unsharp", where "xxx" are the attributes specified in "input_keys".
    Input: NHWC uint8 [0, 255]
    Output: NHWC uint8 [0, 255]
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

        # cv2 returns (K, 1) float64 numpy array
        self.kernel_1d = cv2.getGaussianKernel(self.kernel_size, self.sigma)

    def _unsharp_masking(self, x_nhwc):
        """Unsharp masking function.

        Args:
            x_nhwc (np.ndarray): Input image with shape (N, H, W, C), uint8.

        Returns:
            np.ndarray: Processed image, uint8.
        """
        # Work in float32 for precision
        x_float = x_nhwc.astype(np.float32)
        outputs = np.zeros_like(x_float)

        # Apply separable gaussian blur for each image in the batch
        # cv2.sepFilter2D works on (H, W, C)
        for i in range(x_nhwc.shape[0]):
            img = x_float[i]
            blurred = cv2.sepFilter2D(img, -1, self.kernel_1d, self.kernel_1d, borderType=cv2.BORDER_REFLECT_101)

            residue = img - blurred

            # Thresholding
            # Note: PyTorch implementation divided threshold by 255.0 because input was 0-1.
            # Here input is 0-255, so we use threshold as is.
            mask = (np.abs(residue) > self.threshold).astype(np.float32)

            # Soft mask
            soft_mask = cv2.sepFilter2D(mask, -1, self.kernel_1d, self.kernel_1d, borderType=cv2.BORDER_REFLECT_101)
            soft_mask = np.clip(soft_mask, 0.0, 1.0)

            sharpened = np.clip(img + self.weight * residue, 0.0, 255.0)
            output_frame = soft_mask * sharpened + (1.0 - soft_mask) * img
            outputs[i] = output_frame

        return np.clip(outputs, 0, 255).astype(np.uint8)

    def __call__(self, data_dict: dict) -> dict:
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            if in_key in data_dict:
                data_dict[out_key] = self._unsharp_masking(data_dict[in_key])
        return data_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}, weight={self.weight}, "
            f"threshold={self.threshold}, keys={self.keys})"
        )


class RandomBlur(Augmentor):
    """Apply random blur to the input (NumPy version).

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
        Same logic as PyTorch version, returns list of numpy kernels.
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
        if omega_range is None:
            if kernel_size < 13:
                omega_range = [np.pi / 3.0, np.pi]
            else:
                omega_range = [np.pi / 5.0, np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

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

    def _apply_random_blur(self, x_nhwc):
        """Apply blur with per-frame kernel variation to match mmagic behavior.

        mmagic generates a different kernel for each frame using the _step parameters
        to create a random walk of kernel parameters across the video sequence.
        """
        N = x_nhwc.shape[0]
        # Generate one kernel per frame to match mmagic behavior
        kernels = self.get_kernel(num_kernels=N)

        # x_nhwc is uint8. cv2.filter2D supports uint8 input and will perform internal calculations
        # with higher precision and saturate_cast the result back to uint8 if ddepth=-1.
        outputs = np.empty_like(x_nhwc)

        for i in range(N):
            outputs[i] = cv2.filter2D(x_nhwc[i], -1, kernels[i], borderType=cv2.BORDER_REFLECT_101)

        return outputs

    def __call__(self, data_dict):
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_blur(data_dict[key])

        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomJPEGCompression(Augmentor):
    """Apply random JPEG compression to the input (NumPy version).

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): A dictionary specifying the degradation settings (params).
            Should contain 'color_type' and 'bgr2rgb' along with compression params.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}
        # color_type and bgr2rgb are inherited from torch params but less relevant for strict numpy pipe unless user specifies BGR input
        self.color_type = self.params.get("color_type", "color")
        self.bgr2rgb = self.params.get("bgr2rgb", False)

    def _apply_random_compression(self, x_nhwc):
        quality = self.params["quality"]
        quality_step = self.params.get("quality_step", 0)

        N = x_nhwc.shape[0]

        # Generate qualities for each frame
        qualities = []
        curr_quality = round(np.random.uniform(quality[0], quality[1]))
        for _ in range(N):
            qualities.append(curr_quality)
            curr_quality += np.random.uniform(-quality_step, quality_step)
            curr_quality = round(np.clip(curr_quality, quality[0], quality[1]))

        outputs_list = []
        for i in range(N):
            frame = x_nhwc[i]  # H, W, C

            # cv2.imencode expects BGR. If our input is RGB (common), we should convert.
            # However, if we want to be color-space agnostic, maybe we shouldn't?
            # But JPEG compression relies on YCbCr conversion which depends on RGB input.
            # Assuming standard RGB input for "images".
            # If the user pipeline is strictly RGB, we convert to BGR for opencv, then back.

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(qualities[i])]
            result, encimg = cv2.imencode(".jpg", frame_bgr, encode_param)

            if result:
                decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)  # Decodes to BGR
                # Convert back to RGB
                decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
                outputs_list.append(decimg)
            else:
                # Fallback if encoding fails? Should not happen usually.
                outputs_list.append(frame)

        return np.stack(outputs_list, axis=0)

    def __call__(self, data_dict):
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_compression(data_dict[key])

        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomNoise(Augmentor):
    """Apply random noise to the input (NumPy version).

    Currently support Gaussian noise and Poisson noise.
    Input/Output: NHWC uint8 [0, 255]
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}

    def _apply_gaussian_noise(self, x_nhwc):
        """Apply gaussian noise with per-frame sigma update to match mmagic behavior.

        mmagic updates sigma after each frame using gaussian_sigma_step parameter,
        creating a random walk of noise levels across the video sequence.
        """
        x_float = x_nhwc.astype(np.float32)

        sigma_range = self.params["gaussian_sigma"]
        sigma = float(np.random.uniform(sigma_range[0], sigma_range[1]))
        sigma_step = self.params.get("gaussian_sigma_step", 0)

        gray_noise_prob = self.params["gaussian_gray_noise_prob"]
        is_gray_noise = np.random.uniform() < gray_noise_prob

        N, H, W, C = x_nhwc.shape
        outputs = []

        for i in range(N):
            img = x_float[i]
            # Generate noise for this frame
            noise = np.float32(np.random.randn(H, W, C)) * sigma
            if is_gray_noise:
                # Match mmagic: use first channel slice for gray noise
                noise = noise[:, :, :1]
            outputs.append(img + noise)

            # Update sigma per frame (match mmagic behavior)
            sigma += np.random.uniform(-sigma_step, sigma_step)
            sigma = np.clip(sigma, sigma_range[0], sigma_range[1])

        output = np.stack(outputs, axis=0)
        return np.clip(output, 0.0, 255.0).astype(np.uint8)

    def _apply_poisson_noise(self, x_nhwc):
        """Poisson noise implementation with per-frame processing to match mmagic behavior.

        Key differences from previous implementation:
        1. unique_val is computed dynamically based on unique pixel values (match mmagic)
        2. scale is updated per frame using poisson_scale_step
        3. Processing is done frame-by-frame like mmagic

        Args:
            x_nhwc: Input array with shape (N, H, W, C), uint8 [0, 255]

        Returns:
            Output array with shape (N, H, W, C), uint8 [0, 255]
        """
        N, H, W, C = x_nhwc.shape

        # Get scale parameters
        scale_range = self.params["poisson_scale"]
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scale_step = self.params.get("poisson_scale_step", 0)

        # Determine if gray noise should be applied
        gray_noise_prob = self.params.get("poisson_gray_noise_prob", 0)
        is_gray_noise = np.random.uniform() < gray_noise_prob

        # Convert to float32 for computation
        frame_float = x_nhwc.astype(np.float32)

        outputs = []
        for i in range(N):
            img = frame_float[i]
            noise = img.copy()

            if is_gray_noise:
                # Convert to grayscale for noise computation (match mmagic)
                # Use standard RGB to grayscale weights
                noise = np.dot(noise[..., :3], [0.299, 0.587, 0.114])
                noise = noise[..., np.newaxis]

            noise = np.clip(noise.round(), 0, 255)

            # CRITICAL: Match mmagic's dynamic unique_val calculation
            # This computes 2^ceil(log2(num_unique_values)) which typically gives 256-512
            # Using fixed 255.0 would result in ~2x different noise strength
            unique_val = 2 ** np.ceil(np.log2(len(np.unique(noise))))

            # Poisson sampling: poisson(x * val) / val - x
            noise = np.random.poisson(noise * unique_val).astype(np.float32) / unique_val - noise

            outputs.append(img + noise * scale)

            # Update scale per frame (match mmagic behavior)
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        output = np.stack(outputs, axis=0)
        return np.clip(output, 0.0, 255.0).astype(np.uint8)

    def _apply_random_noise(self, x_nhwc):
        noise_type = np.random.choice(self.params["noise_type"], p=self.params["noise_prob"])

        if noise_type.lower() == "gaussian":
            x_nhwc = self._apply_gaussian_noise(x_nhwc)
        elif noise_type.lower() == "poisson":
            x_nhwc = self._apply_poisson_noise(x_nhwc)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is not implemented.')
        return x_nhwc

    def __call__(self, data_dict):
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_noise(data_dict[key])
        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomResize(Augmentor):
    """Randomly resize the input (NumPy version).

    Modified keys are the attributed specified in "keys".
    Input: NHWC uint8
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.keys = input_keys
        self.params = args if args is not None else {}

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR, bicubic=cv2.INTER_CUBIC, area=cv2.INTER_AREA, lanczos=cv2.INTER_LANCZOS4
        )

        self.get_resize_target_from_data_batch = self.params.get("get_resize_target_from_data_batch", False)
        self.gt_key_for_target = self.params.get("gt_key_for_target", "gt")

    def _random_resize(self, x_nhwc, data_dict=None):
        """Randomly resize with optional per-frame scale variation to match mmagic behavior.

        mmagic supports resize_step parameter to vary the scale factor across frames,
        creating a random walk of sizes across the video sequence.
        """
        N, H, W, C = x_nhwc.shape

        resize_opt = self.params["resize_opt"]
        resize_prob = self.params["resize_prob"]
        resize_opt = np.random.choice(resize_opt, p=resize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f"resize_opt [{resize_opt}] is not implemented")
        interpolation = self.resize_dict[resize_opt]

        # Get resize_step for per-frame variation (match mmagic)
        resize_step = self.params.get("resize_step", 0)

        # determine the target size
        target_size = self.params.get("target_size", None)
        scale_factor = None
        resize_scale = self.params.get("resize_scale", [0.5, 1.5])

        if self.get_resize_target_from_data_batch and data_dict is not None:
            gt_key = self.gt_key_for_target
            if gt_key in data_dict:
                gt_cthw = data_dict[gt_key]  # C T H W
                gt_h, gt_w = gt_cthw.shape[2], gt_cthw.shape[3]  # C T H W
                target_scale_from_gt = self.params["target_scale_from_gt"]
                target_h = int(gt_h * target_scale_from_gt[0])
                target_w = int(gt_w * target_scale_from_gt[1])
                target_size = (target_w, target_h)  # cv2 uses (W, H)
                resize_step = 0  # Fixed target size, no step
            else:
                raise ValueError(f"GT key [{gt_key}] not found in data_dict")

        if target_size is None:
            resize_mode = np.random.choice(["up", "down", "keep"], p=self.params["resize_mode_prob"])
            if resize_mode == "up":
                scale_factor = np.random.uniform(1, resize_scale[1])
            elif resize_mode == "down":
                scale_factor = np.random.uniform(resize_scale[0], 1)
            else:
                scale_factor = 1

            h_out, w_out = H * scale_factor, W * scale_factor
            if self.params.get("is_size_even", False):
                h_out, w_out = 2 * (int(h_out) // 2), 2 * (int(w_out) // 2)
            target_size = (int(w_out), int(h_out))  # cv2 uses (W, H)
        else:
            # Ensure target_size is (W, H)
            # PyTorch usually uses (H, W). If target_size from params is (H, W), we need to swap for cv2.
            if isinstance(target_size, (list, tuple)):
                target_size = (target_size[1], target_size[0])
            resize_step = 0  # Fixed target size, no step

        # Apply division_factor alignment if specified
        # Note: target_size is (W, H) format for cv2
        division_factor = self.params.get("division_factor", None)
        if division_factor is not None and division_factor > 1:
            target_w, target_h = target_size
            # Round down to nearest multiple of division_factor
            target_w = (target_w // division_factor) * division_factor
            target_h = (target_h // division_factor) * division_factor
            target_size = (target_w, target_h)

        # Store initial target size for consistency check
        initial_target_size = target_size

        outputs = []
        for i in range(N):
            # cv2.resize(src, dsize=(width, height), ...)
            resized = cv2.resize(x_nhwc[i], target_size, interpolation=interpolation)
            outputs.append(resized)

            # Update scale factor per frame if resize_step > 0 (match mmagic behavior)
            # Note: This may result in different sizes per frame. We track this below.
            if resize_step > 0 and scale_factor is not None:
                scale_factor += np.random.uniform(-resize_step, resize_step)
                scale_factor = np.clip(scale_factor, resize_scale[0], resize_scale[1])

                # Recompute target size for next frame
                h_out, w_out = H * scale_factor, W * scale_factor
                if self.params.get("is_size_even", False):
                    h_out, w_out = 2 * (int(h_out) // 2), 2 * (int(w_out) // 2)
                target_size = (int(w_out), int(h_out))

                # Apply division_factor alignment
                if division_factor is not None and division_factor > 1:
                    target_w, target_h = target_size
                    target_w = (target_w // division_factor) * division_factor
                    target_h = (target_h // division_factor) * division_factor
                    target_size = (target_w, target_h)

        # Check if all outputs have the same shape for stacking
        # If resize_step caused different sizes, resize all to the first frame's size
        if resize_step > 0 and len(outputs) > 1:
            first_shape = outputs[0].shape
            needs_resize = any(o.shape != first_shape for o in outputs)
            if needs_resize:
                # Resize all frames to match the first frame's size for consistency
                target_h, target_w = first_shape[:2]
                outputs = [
                    cv2.resize(o, (target_w, target_h), interpolation=interpolation) if o.shape != first_shape else o
                    for o in outputs
                ]

        return np.stack(outputs, axis=0)

    def __call__(self, data_dict):
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._random_resize(data_dict[key], data_dict)
        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class RandomVideoCompression(Augmentor):
    """Apply random video compression to the input (NumPy version).

    Modified keys are the attributed specified in "keys".
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert has_av, "Please install av to use video compression."

        self.keys = input_keys
        self.params = args if args is not None else {}
        self.skip_flag = args.get("skip_flag", "skip_video_compression")
        logging.getLogger("libav").setLevel(50)

    def _apply_random_compression(self, x_nhwc):
        """Apply random compression; input NHWC uint8."""
        codec = random.choices(self.params["codec"], self.params["codec_prob"])[0]
        bitrate = self.params["bitrate"]
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        # x_nhwc is already uint8 numpy
        N, H, W, C = x_nhwc.shape

        buf = io.BytesIO()
        with av.open(buf, "w", "mp4") as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = H
            stream.width = W
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = bitrate

            for img in x_nhwc:
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame.pict_type = 0
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, "r", "mp4") as container:
            if container.streams.video:
                for frame in container.decode(**{"video": 0}):
                    # Match mmagic: decode to float32
                    img_decoded = frame.to_rgb().to_ndarray().astype(np.float32)
                    outputs.append(img_decoded)

        # If decode fails or yields fewer frames, we might have issues.
        # But assuming it works like PyTorch version.
        if len(outputs) == 0:
            return x_nhwc.astype(np.float32)

        outputs = np.stack(outputs, axis=0)
        # Match mmagic: return float32 instead of uint8
        return outputs

    def __call__(self, data_dict):
        if np.random.uniform() > self.params.get("prob", 1):
            return data_dict

        if self.skip_flag in data_dict and data_dict[self.skip_flag]:
            return data_dict

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = self._apply_random_compression(data_dict[key])

        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str
