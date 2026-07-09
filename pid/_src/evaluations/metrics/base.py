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

"""Base classes for metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, device: str = "cuda"):
        """
        Initialize the metric.

        Args:
            name: Name of the metric
            device: Device to run computation on
        """
        self.name = name
        self.device = device
        self._values: List[float] = []

    @abstractmethod
    def compute(
        self,
        pred: Union[np.ndarray, th.Tensor],
        target: Union[np.ndarray, th.Tensor],
    ) -> float:
        """
        Compute the metric between prediction and target.

        Args:
            pred: Predicted image/video, shape (H, W, C) or (T, H, W, C) for numpy,
                  or (C, H, W) / (C, T, H, W) for torch
            target: Ground truth image/video, same shape as pred

        Returns:
            Metric value as a float
        """
        pass

    def compute_batch_list(
        self,
        preds: List,
        targets: Optional[List] = None,
    ) -> List[Any]:
        """Run the metric over a list of per-image inputs.

        Default strategy:
          - If every pred (and target, when relevant) is a same-shape numpy array
            AND the subclass defines a `compute_batch((T, H, W, C))` method, stack
            them into a batch and call `compute_batch` once. This preserves the
            subclass's resize behavior — we do NOT inject any new resize.
          - Otherwise, fall back to a per-image `self.compute(pred, target)` loop.

        Subclasses whose `compute_batch` returns a non-`List[float]` (e.g. a
        dict-per-image metric like `QAlign`) MUST override this to produce a
        `List` whose length equals `len(preds)`.

        Args:
            preds: list of per-image predictions (HWC numpy arrays, typically).
            targets: parallel list of ground truths for FR metrics; None or list
                of Nones for NR metrics.

        Returns:
            List with one output per input image.
        """
        if targets is None:
            targets = [None] * len(preds)
        if len(preds) != len(targets):
            raise ValueError(f"preds ({len(preds)}) and targets ({len(targets)}) length mismatch")

        has_compute_batch = hasattr(self, "compute_batch")
        if has_compute_batch and len(preds) > 0:
            shapes = {getattr(p, "shape", None) for p in preds}
            shapes.discard(None)
            has_targets = any(t is not None for t in targets)
            uniform = len(shapes) == 1
            target_ok = not has_targets or all(
                t is not None and getattr(t, "shape", None) == next(iter(shapes)) for t in targets
            )
            if uniform and target_ok:
                pred_batch = np.stack(preds, axis=0)
                if has_targets:
                    target_batch = np.stack(targets, axis=0)
                    return list(self.compute_batch(pred_batch, target_batch))
                return list(self.compute_batch(pred_batch))

        # Fallback: per-image loop. Works whether or not the subclass's compute
        # accepts `target=None`.
        return [(self.compute(p, t) if t is not None else self.compute(p)) for p, t in zip(preds, targets)]

    def update(self, pred: Union[np.ndarray, th.Tensor], target: Union[np.ndarray, th.Tensor]) -> float:
        """
        Compute metric and add to running values.

        Args:
            pred: Predicted image/video
            target: Ground truth image/video

        Returns:
            Computed metric value
        """
        value = self.compute(pred, target)
        self._values.append(value)
        return value

    def reset(self):
        """Reset accumulated values."""
        self._values = []

    def get_mean(self) -> float:
        """Get mean of accumulated values."""
        if not self._values:
            return 0.0
        return float(np.mean(self._values))

    def get_std(self) -> float:
        """Get standard deviation of accumulated values."""
        if not self._values:
            return 0.0
        return float(np.std(self._values))

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            f"{self.name}_mean": self.get_mean(),
            f"{self.name}_std": self.get_std(),
            f"{self.name}_min": float(np.min(self._values)) if self._values else 0.0,
            f"{self.name}_max": float(np.max(self._values)) if self._values else 0.0,
            f"{self.name}_count": len(self._values),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"


class MetricRegistry:
    """Registry for managing multiple metrics.

    Supports singleton caching: when multiple callers request the same metric
    with the same kwargs, they share one instance instead of loading duplicate
    models (e.g. QAlign ~8GB). This is process-local, so multi-GPU training
    (DDP/FSDP with one process per rank) works correctly — no cross-rank sharing.
    """

    _metrics: Dict[str, type] = {}
    _instances: Dict[tuple, BaseMetric] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a metric class."""

        def decorator(metric_class: type):
            cls._metrics[name.lower()] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMetric:
        """
        Get a (cached) metric instance by name.

        Instances are cached by (name, kwargs) so that multiple callbacks
        sharing the same metric config reuse one model instead of loading
        duplicate weights into memory.

        Args:
            name: Name of the metric (case-insensitive)
            **kwargs: Additional arguments to pass to the metric constructor

        Returns:
            Metric instance
        """
        name_lower = name.lower()
        if name_lower not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available metrics: {available}")

        cache_key = (name_lower, tuple(sorted(kwargs.items())))
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._metrics[name_lower](**kwargs)
        return cls._instances[cache_key]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available metrics."""
        return list(cls._metrics.keys())

    @classmethod
    def create_multiple(cls, names: List[str], **kwargs) -> Dict[str, BaseMetric]:
        """
        Create multiple metric instances.

        Args:
            names: List of metric names
            **kwargs: Additional arguments to pass to metric constructors

        Returns:
            Dictionary mapping metric names to instances
        """
        return {name: cls.get(name, **kwargs) for name in names}
