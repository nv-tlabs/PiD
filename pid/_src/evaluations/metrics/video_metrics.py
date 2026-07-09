# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video quality metrics: DOVER."""

from typing import List, Union

import numpy as np
import torch

try:
    # DOVER imports (requires PYTHONPATH to include /root/henry/DOVER)
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER as DOVERModel
except ImportError:
    print("DOVER is not installed, DOVER metric is not available.")
    DOVERModel = None
    UnifiedFrameSampler = None
    spatial_temporal_view_decomposition = None

from pid._src.evaluations.metrics.base import BaseMetric, MetricRegistry


def fuse_results(results: list):
    """Fuse aesthetic and technical scores. From DOVER's evaluate_a_set_of_videos.py."""
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


@MetricRegistry.register("dover")
class DOVER(BaseMetric):
    """
    DOVER metric using the official DOVER package.

    Accepts video path and computes aesthetic/technical/overall scores.
    """

    def __init__(self, device: str = "cuda", config_path: str = None):
        super().__init__(name="DOVER", device=device)

        import yaml

        if config_path is None:
            config_path = "/root/henry/LinearVSR/evaluations/configs/dover.yml"

        with open(config_path, "r") as f:
            self.opt = yaml.safe_load(f)

        # Load model
        self.evaluator = DOVERModel(**self.opt["model"]["args"]).to(device)
        self.evaluator.load_state_dict(torch.load(self.opt["test_load_path"], map_location=device))
        self.evaluator.eval()

        # Get sample types config
        self.sample_types = self.opt["data"]["val-l1080p"]["args"]["sample_types"]

        # Build samplers
        self.samplers = {}
        for stype, sopt in self.sample_types.items():
            if "t_frag" not in sopt:
                self.samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
            else:
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        # Normalization
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        # Score storage
        self._aesthetic_scores: List[float] = []
        self._technical_scores: List[float] = []
        self._overall_scores: List[float] = []

    def compute_from_path(self, video_path: str) -> dict:
        """Compute DOVER scores from video path."""
        # Use DOVER's spatial_temporal_view_decomposition
        data, _ = spatial_temporal_view_decomposition(video_path, self.sample_types, self.samplers, is_train=False)

        # Normalize
        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        # Prepare video dict for model
        video = {}
        for key in ["aesthetic", "technical"]:
            if key in data:
                video[key] = data[key].to(self.device)
                c, t, h, w = video[key].shape
                video[key] = video[key].unsqueeze(0)  # (1, C, T, H, W)

                num_clips = self.sample_types[key]["num_clips"]
                # Reshape: (B, C, T, H, W) -> (B*num_clips, C, T//num_clips, H, W)
                video[key] = (
                    video[key]
                    .reshape(1, c, num_clips, t // num_clips, h, w)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(num_clips, c, t // num_clips, h, w)
                )

        # Inference
        with torch.no_grad():
            results = self.evaluator(video, reduce_scores=False)
            results = [np.mean(r.cpu().numpy()) for r in results]

        return fuse_results(results)

    def compute(
        self,
        pred: Union[str, np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor] = None,
    ) -> float:
        """Compute DOVER overall score. pred can be video path (str) or frames."""
        if isinstance(pred, str):
            scores = self.compute_from_path(pred)
        else:
            raise NotImplementedError("Direct frame input not supported. Use video path.")
        return scores["overall"]

    def update(
        self, pred: Union[str, np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor] = None
    ) -> float:
        """Compute and accumulate DOVER scores."""
        if isinstance(pred, str):
            scores = self.compute_from_path(pred)
        else:
            raise NotImplementedError("Direct frame input not supported. Use video path.")

        self._aesthetic_scores.append(scores["aesthetic"])
        self._technical_scores.append(scores["technical"])
        self._overall_scores.append(scores["overall"])
        self._values.append(scores["overall"])

        return scores["overall"]

    def get_detailed_scores(self) -> dict:
        """Get all accumulated detailed scores."""
        return {
            "aesthetic": self._aesthetic_scores.copy(),
            "technical": self._technical_scores.copy(),
            "overall": self._overall_scores.copy(),
        }

    def get_summary(self) -> dict:
        """Get summary statistics."""
        summary = super().get_summary()
        if self._aesthetic_scores:
            summary.update(
                {
                    "DOVER_aesthetic_mean": float(np.mean(self._aesthetic_scores)),
                    "DOVER_aesthetic_std": float(np.std(self._aesthetic_scores)),
                    "DOVER_technical_mean": float(np.mean(self._technical_scores)),
                    "DOVER_technical_std": float(np.std(self._technical_scores)),
                }
            )
        return summary

    def reset(self):
        """Reset all accumulated scores."""
        super().reset()
        self._aesthetic_scores = []
        self._technical_scores = []
        self._overall_scores = []
