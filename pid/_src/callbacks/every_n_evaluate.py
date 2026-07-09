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
Evaluation callback for PixelDiT SR image training.

Runs full evaluation on fix_batch .pt datasets at specified intervals during training,
computing metrics directly in-memory without saving any media files.

Key features:
- Pre-loads all fix_batch data to CPU memory at training start (one-time cost)
- Transfers data to GPU only during evaluation (minimal redundant transfers)
- DP-aware with modulo distribution of samples across ranks

Supported metrics:
- Full-reference: PSNR, SSIM, LPIPS (require ground truth)
- LQ-reference: LQ_COLOR_DE2000 (compares SR output with the LQ input color)
- No-reference: NIQE, MUSIQ, CLIPIQA, QALIGN, VISUALQUALITY_R1 (no ground truth needed)

Note: DOVER metric is not supported.
"""

import json
import os
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from PIL import Image
from tabulate import tabulate

try:
    import wandb
except ImportError:
    wandb = None

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import distributed, log, misc
from pid._src.callbacks.eval_datasets import FixBatchImageDataset
from pid._src.callbacks.every_n import EveryN

metrics_better_symbol = {
    "PSNR": "↑",
    "SSIM": "↑",
    "LPIPS": "↓",
    "LQ_COLOR_DE2000": "↓",
    "NIQE": "↓",
    "MUSIQ": "↑",
    "MUSIQ_PAQ2PIQ": "↑",
    "MUSIQ_SPAQ": "↑",
    "CLIPIQA": "↑",
    "CLIPIQA_PLUS": "↑",
    "QALIGN_NATIVE": "↑",
    "VISUALQUALITY_R1": "↑",
    "UNIPERCEPT_IAA": "↑",
    "UNIPERCEPT_IQA": "↑",
    "UNIPERCEPT_ISTA": "↑",
}

# Display-name overrides for metrics whose registry name differs from the
# historical wandb tag. qalign_native produces the same number that the older
# qalign.compute_batch() returned under the 'qalign_quality_native' key, so
# we keep that display name on wandb / summary tables for dashboard continuity.
metric_display_name_overrides = {
    "QALIGN_NATIVE": "QALIGN_QUALITY_NATIVE",
}


def _display_name(metric_name_upper: str) -> str:
    return metric_display_name_overrides.get(metric_name_upper, metric_name_upper)


class EveryNEvaluate(EveryN):
    """Evaluation callback that runs metrics on evaluation datasets during training.

    This callback:
    1. Pre-loads all evaluation data to CPU memory at training start
    2. Distributes samples across DP ranks using modulo arithmetic
    3. Runs inference using generate_samples_from_batch() (like EveryNDrawSample)
    4. Computes metrics directly on tensors in memory
    5. Aggregates results across DP ranks
    6. Logs to wandb and saves JSON results

    Data flow:
    - on_train_start: Load all LQ/HQ data to CPU memory (one-time)
    - every_n_impl: Transfer sample to GPU -> inference -> metrics -> cleanup

    Args:
        every_n: Run evaluation every N iterations
        step_size: Step size for iteration counting
        fix_batch_dir: Directory of fix_batch .pt files to evaluate on (required).
        metrics: List of metrics to compute. Default: ["psnr", "ssim", "lpips", "niqe", "musiq", "clipiqa"]
        batch_size_in_evaluation: Batch size for metric forward calls.
        device: Device for metric computation
        guidance: Guidance scale for inference
        num_sampling_step: Number of sampling steps for inference
        run_at_start: Whether to run evaluation at training start
        is_ema: Whether to use EMA model for evaluation
    """

    def __init__(
        self,
        every_n: int,
        fix_batch_dir: str,
        step_size: int = 1,
        metrics: List[str] = None,
        batch_size_in_evaluation: int = None,
        device: str = "cuda",
        guidance: float = 1.0,
        num_sampling_step: int = 35,
        run_at_start: bool = False,
        is_ema: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.name = self.__class__.__name__ + (f"_{name}" if name is not None else "")

        fix_batch_name = f"FixBatch_{name}" if name is not None else "FixBatch"
        self.datasets = [FixBatchImageDataset(fix_batch_dir=fix_batch_dir, name=fix_batch_name)]
        log.info(f"Dataset: {[p.name for p in self.datasets]}")

        # Default to all supported metrics (excluding DOVER)
        if metrics is None:
            metrics = ["psnr", "ssim", "lpips", "niqe", "musiq", "clipiqa"]

        # Validate metrics - exclude DOVER
        if "dover" in metrics:
            log.warning("DOVER metric is not supported in EveryNEvaluate")
            metrics = [m for m in metrics if m != "dover"]

        self.metrics = metrics
        self.batch_size_in_evaluation = batch_size_in_evaluation
        self.device = device
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.is_ema = is_ema

        # Will be initialized in on_train_start
        self.metrics_dict = {}
        self.local_dir = None
        self.data_parallel_id = 0
        self.data_parallel_world_size = 1

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Initialize evaluation: setup directories, load data to CPU, initialize metrics.

        Args:
            model: The model being trained
            iteration: Current training iteration
        """
        # 1. Setup directories
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            os.makedirs(f"{self.local_dir}/results", exist_ok=True)
            log.info(f"{self.name}: local_dir: {self.local_dir}")

        # 2. Get DP rank and world size
        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
            self.data_parallel_world_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_id = distributed.get_rank()
            self.data_parallel_world_size = distributed.get_world_size()

        # 3. Load all datasets to CPU memory (one-time operation)
        for dataset in self.datasets:
            log.info(f"{self.name}: Loading {dataset.name} dataset to CPU...", rank0_only=False)
            dataset.load_all_samples(self.data_parallel_id, self.data_parallel_world_size)

        # 4. Initialize metrics. VLM metrics are pinned to backend="transformers"
        # because vLLM's pre-allocated KV-cache reservation can't be released back
        # to PyTorch's allocator without tearing the engine down — fine for the
        # standalone evaluator, fatal for an in-training callback that has to
        # share the GPU with FSDP shards + optimizer state every N steps.
        from pid._src.evaluations.metrics import MetricRegistry

        metric_kwargs: Dict[str, Dict[str, Any]] = {
            "lpips": {"net": "vgg"},
            "visualquality_r1": {"backend": "transformers"},
            "q_insight": {"backend": "transformers"},
        }
        for metric_name in self.metrics:
            kwargs = metric_kwargs.get(metric_name, {})
            self.metrics_dict[metric_name] = MetricRegistry.get(metric_name, device=self.device, **kwargs)

        log.info(f"Initialized metrics: {list(self.metrics_dict.keys())}", rank0_only=False)

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        """Main callback implementation: run inference and compute metrics on fix_batch samples.

        Args:
            trainer: The trainer object
            model: The model being trained
            data_batch: Current training batch (unused)
            output_batch: Model outputs (unused)
            loss: Current loss (unused)
            iteration: Current training iteration
        """
        # Use EMA scope if requested
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_evaluation")
        else:
            context = nullcontext

        with context():
            # Iterate over all datasets
            for dataset in self.datasets:
                log.info(f"DP{self.data_parallel_id}: Evaluating {dataset.name} dataset", rank0_only=False)

                # Step 1: Run inference on assigned samples
                log.info(
                    f"DP{self.data_parallel_id}: Running inference on {len(dataset)} samples from {dataset.name}",
                    rank0_only=False,
                )
                inference_results = self.run_inference(model, dataset, iteration)

                # Step 2: Compute metrics
                log.info(f"DP{self.data_parallel_id}: Computing metrics for {dataset.name}", rank0_only=False)
                evaluation_results = self.compute_metrics(inference_results)

                # Step 3: Barrier before aggregation
                if dist.is_initialized():
                    dist.barrier()

                # Step 4: Aggregate and log results
                self.aggregate_and_log(evaluation_results, dataset, iteration)

                log.info(f"DP{self.data_parallel_id}: Evaluation completed for {dataset.name}.", rank0_only=False)

                # Clean up
                torch.cuda.empty_cache()

            # Offload heavy metric models (e.g. QAlign ~14GB) back to CPU so they
            # don't squat on GPU memory during the next training chunk. Doing it
            # once here avoids the per-sample PCIe round-trip.
            for m in self.metrics_dict.values():
                if hasattr(m, "offload_to_cpu"):
                    m.offload_to_cpu()
            torch.cuda.empty_cache()

            log.info(f"DP{self.data_parallel_id}: All dataset evaluations completed.", rank0_only=False)

    def run_inference(self, model, dataset, iteration):
        """Run inference on assigned fix_batch samples.

        Data flow: CPU -> GPU -> pixel-space inference. Output is already pixel-space.

        Args:
            model: The model to use for inference
            dataset: The dataset to evaluate
            iteration: Current training iteration

        Returns:
            List of dicts with keys: name, lq_data (CPU), hq_data (CPU), sr_output (GPU)
        """
        inference_results = []

        for idx in range(len(dataset)):
            name, lq_data, hq_data, caption, lq_latent, degrade_sigma = dataset.get_sample(idx)

            log.info(f"DP{self.data_parallel_id}: Processing {name}", rank0_only=False)
            log.info(f"num_steps: {self.num_sampling_step}, guidance: {self.guidance}, shift: {model.config.shift}")

            data_batch = self.prepare_data_batch(
                lq_data,
                model,
                hq_images=hq_data,
                caption=caption,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
            )
            # generate_samples_from_batch returns [B, 3, 1, H, W] in [-1, 1]
            sr_output = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=self.guidance,
                shift=model.config.shift,
                num_steps=self.num_sampling_step,
            )
            del data_batch
            log.info(f"DP{self.data_parallel_id}: SR output shape: {sr_output.shape}", rank0_only=False)

            # Store results: HQ stays on CPU, SR stays on GPU for metrics
            inference_results.append(
                {
                    "name": name,
                    "lq_data": lq_data,  # CPU, (B, H_lq, W_lq, C) uint8
                    "hq_data": hq_data,  # CPU, (B, H, W, C) uint8
                    "sr_output": sr_output,  # GPU, (B, C, 1, H, W) in [-1, 1]
                }
            )

            # Clean up intermediate tensors
            torch.cuda.empty_cache()

        return inference_results

    def prepare_data_batch(
        self,
        lq_images: np.ndarray,
        model,
        hq_images: np.ndarray = None,
        caption=None,
        lq_latent=None,
        degrade_sigma=0.0,
    ):
        """Prepare latent-only PiD inference input or reconstruction input.

        PiD inference receives only caption, LQ latent, and degradation sigma.
        LQ/HQ pixels remain outside the model batch for metrics.

        Args:
            lq_images: LQ image (B, H, W, C) uint8 numpy array
            model: PixelDiTSRModel instance
            caption: list of caption strings, or None (uses empty strings)
            lq_latent: required pre-computed LQ latent

        Returns:
            Data batch dict ready for model.generate_samples_from_batch()
        """
        if getattr(model, "evaluate_fix_batch_on_hq", False):
            if hq_images is None:
                raise ValueError("hq_images must be provided when model.evaluate_fix_batch_on_hq is True")
            return self.prepare_reconstruction_data_batch(hq_images, model)

        B = lq_images.shape[0]
        input_caption_key = model.config.input_caption_key

        # Use captions from fix_batch .pt if available, otherwise empty strings
        if caption is None:
            captions = [""] * B
        elif isinstance(caption, str):
            captions = [caption] * B
        elif isinstance(caption, list) and len(caption) == 1 and B > 1:
            captions = caption * B
        else:
            captions = caption

        if not isinstance(lq_latent, torch.Tensor):
            raise ValueError("EveryNEvaluate requires a pre-computed LQ_latent")
        data_batch = {
            input_caption_key: captions,
            "LQ_latent": lq_latent,
            "degrade_sigma": torch.full((B,), float(degrade_sigma), dtype=torch.float32),
        }

        data_batch = misc.to(data_batch, **model.tensor_kwargs)
        return data_batch

    def prepare_reconstruction_data_batch(self, hq_images: np.ndarray, model):
        """Prepare HQ pixels for tokenizer/reconstruction eval.

        Reconstruction models do not consume LQ/caption/degradation fields; the
        fix_batch HQ image is the model input and metric target.
        """
        hq_tensor = torch.from_numpy(hq_images).float() / 127.5 - 1.0
        hq_tensor = rearrange(hq_tensor, "b h w c -> b c h w")
        input_key = model.config.input_image_key

        return misc.to({input_key: hq_tensor}, **model.tensor_kwargs)

    def compute_metrics(self, inference_results):
        """Compute all metrics on inference results (in-memory, no DOVER).

        Stacks samples by shape and runs each metric once per shape group.

        Args:
            inference_results: List of dicts from run_inference()

        Returns:
            List of dicts with per-sample metric results
        """
        FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips"}
        LQ_REFERENCE_METRICS = {"lq_color_de2000"}
        if not inference_results:
            return []
        return self._compute_metrics_image_mode(inference_results, FULL_REFERENCE_METRICS, LQ_REFERENCE_METRICS)

    def _compute_metrics_image_mode(self, inference_results, FULL_REFERENCE_METRICS, LQ_REFERENCE_METRICS):
        """Stack samples into (N, H, W, C) batches and run each metric once per shape group.

        Defensive shape grouping: FixBatchImageDataset doesn't guarantee
        uniform (H, W) across .pt files in one dir, even though curated dirs
        usually are. If shapes diverge we run one flat batch per shape group.
        """
        need_hq = any(m in self.metrics for m in FULL_REFERENCE_METRICS)
        need_lq = any(m in self.metrics for m in LQ_REFERENCE_METRICS)

        # by_shape: (SR shape, LQ shape) -> list of (sample_idx, sr, hq or None, lq or None)
        by_shape = defaultdict(list)
        for s_idx, result in enumerate(inference_results):
            sr = self.convert_to_numpy_hwc(result["sr_output"])  # (1, H, W, C) uint8
            hq = None
            lq = None
            if need_hq:
                hq = result["hq_data"]
                sr, hq = self.match_images(sr, hq)
            if need_lq:
                lq = result["lq_data"]
                sr, lq = self.match_sample_count(sr, lq)
            shape_key = (sr.shape[1:], lq.shape[1:] if lq is not None else None)
            by_shape[shape_key].append((s_idx, sr, hq, lq))

        log.info(
            f"DP{self.data_parallel_id}: image-mode compute_metrics "
            f"(N={len(inference_results)}, shape_groups={[(s, len(g)) for s, g in by_shape.items()]})",
            rank0_only=False,
        )

        per_sample_results = [{"name": r["name"]} for r in inference_results]

        for _shape, group in by_shape.items():
            flat_sr = np.concatenate([g[1] for g in group], axis=0)  # (N_g, H, W, C)
            flat_hq = np.concatenate([g[2] for g in group], axis=0) if need_hq else None
            flat_lq = np.concatenate([g[3] for g in group], axis=0) if need_lq else None
            sample_idx_for_image = [g[0] for g in group]  # 1:1 with the flat axis

            for metric_name in self.metrics:
                metric = self.metrics_dict[metric_name]
                try:
                    if metric_name in FULL_REFERENCE_METRICS:
                        flat_scores = metric.compute_batch(flat_sr, flat_hq, batch_size=self.batch_size_in_evaluation)
                    elif metric_name in LQ_REFERENCE_METRICS:
                        flat_scores = metric.compute_batch(flat_sr, flat_lq, batch_size=self.batch_size_in_evaluation)
                    else:
                        flat_scores = metric.compute_batch(
                            flat_sr, target=None, batch_size=self.batch_size_in_evaluation
                        )
                    for s_idx, score in zip(sample_idx_for_image, flat_scores):
                        per_sample_results[s_idx][metric_name] = float(score)
                except Exception as e:
                    log.error(
                        f"Error computing {metric_name} (image-mode, shape={_shape}, N={len(group)}): {e}",
                        rank0_only=False,
                    )
                    for s_idx in sample_idx_for_image:
                        per_sample_results[s_idx][metric_name] = None

        torch.cuda.empty_cache()
        return per_sample_results

    def convert_to_numpy_hwc(self, sr_output):
        """Convert model output to (B, H, W, C) in [0, 255].

        Args:
            sr_output: Model output tensor, (B, C, 1, H, W) in [-1, 1].

        Returns:
            numpy array of shape (B, H, W, C) in [0, 255], dtype uint8
        """
        if sr_output.dim() != 5 or sr_output.shape[2] != 1:
            raise ValueError(f"Expected image output [B, C, 1, H, W], got {tuple(sr_output.shape)}")
        sr = sr_output[:, :, 0]
        sr = (sr.clamp(-1, 1) + 1.0) / 2.0
        sr = rearrange(sr, "b c h w -> b h w c")

        sr_np = (sr.float().cpu().numpy() * 255).astype(np.uint8)
        return sr_np

    def match_sample_count(self, images_a, images_b):
        """Trim two image batches to the same batch length without resizing."""
        min_samples = min(images_a.shape[0], images_b.shape[0])
        return images_a[:min_samples], images_b[:min_samples]

    def match_images(self, sr_images, hq_images):
        """Ensure SR and HQ image batches match in count and resolution.

        Args:
            sr_images: SR image array (B, H, W, C)
            hq_images: HQ image array (B, H, W, C)

        Returns:
            Tuple of (sr_images, hq_images) with matched dimensions
        """
        # Handle batch count mismatch
        min_samples = min(sr_images.shape[0], hq_images.shape[0])
        sr_images = sr_images[:min_samples]
        hq_images = hq_images[:min_samples]

        # Handle resolution mismatch - resize HQ to match SR
        if sr_images.shape[1:3] != hq_images.shape[1:3]:
            log.warning(
                f"Resolution mismatch: SR={sr_images.shape[1:3]}, HQ={hq_images.shape[1:3]}. Resizing HQ to match SR.",
                rank0_only=False,
            )
            hq_images = self.resize_images(hq_images, sr_images.shape[1], sr_images.shape[2])

        return sr_images, hq_images

    def resize_images(self, images, target_h, target_w):
        """Resize images to target resolution.

        Args:
            images: Image batch array of shape (B, H, W, C)
            target_h: Target height
            target_w: Target width

        Returns:
            Resized image array of shape (B, target_h, target_w, C)
        """
        resized = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])
            img = img.resize((target_w, target_h), Image.BICUBIC)
            resized.append(np.array(img))
        return np.stack(resized, axis=0)

    def aggregate_and_log(self, per_sample_results, dataset, iteration):
        """Aggregate results across DP ranks and log to wandb.

        Args:
            per_sample_results: List of per-sample metric dicts
            dataset: The dataset that was evaluated
            iteration: Current training iteration
        """
        # Prepare local results for gathering
        local_results = {
            "dp_rank": self.data_parallel_id,
            "per_sample": per_sample_results,
            "num_samples": len(per_sample_results),
        }

        # Gather results from all ranks
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_results)
        else:
            gathered_results = [local_results]

        # Only rank 0 aggregates and logs
        if distributed.get_rank() == 0:
            # Combine all per-sample results
            all_sample_results = []
            for rank_result in gathered_results:
                if rank_result and rank_result["per_sample"]:
                    all_sample_results.extend(rank_result["per_sample"])

            # Aggregate statistics
            aggregated = self.compute_aggregate_statistics(all_sample_results)

            # Save detailed results locally
            self.save_results_json(all_sample_results, aggregated, dataset, iteration)

            # Log to wandb
            self.log_to_wandb(aggregated, dataset, iteration)

            # Print summary
            self.print_summary_table(aggregated, dataset)

    def compute_aggregate_statistics(self, all_sample_results):
        """Compute mean, std, min, max for each metric.

        Args:
            all_sample_results: List of per-sample metric dicts from all ranks

        Returns:
            Dict with aggregated statistics
        """
        aggregated = {"num_samples": len(all_sample_results)}

        for metric_name in self.metrics:
            values = [r[metric_name] for r in all_sample_results if r.get(metric_name) is not None]

            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))

        return aggregated

    def log_to_wandb(self, aggregated, dataset, iteration):
        """Log aggregated metrics to wandb.

        Args:
            aggregated: Dict with aggregated statistics
            dataset: The dataset that was evaluated
            iteration: Current training iteration
        """
        if not wandb or not wandb.run:
            return

        tag = "ema" if self.is_ema else "reg"
        dataset_name = dataset.name

        wandb_dict = {"trainer/global_step": iteration}

        for key, value in aggregated.items():
            if key.endswith("_mean"):
                metric_name = key.replace("_mean", "").upper()
                display = _display_name(metric_name)
                wandb_dict[f"eval/{dataset_name}/{tag}/{display}{metrics_better_symbol[metric_name]}"] = value

        wandb.log(wandb_dict, step=iteration)
        log.info(f"Logged {dataset_name} evaluation metrics to wandb at step {iteration}")

    def save_results_json(self, all_sample_results, aggregated, dataset, iteration):
        """Save detailed results to JSON file.

        Args:
            all_sample_results: List of per-sample metric dicts
            aggregated: Dict with aggregated statistics
            dataset: The dataset that was evaluated
            iteration: Current training iteration
        """
        tag = "ema" if self.is_ema else "reg"
        dataset_name = dataset.name

        results = {
            "dataset": dataset_name,
            "iteration": iteration,
            "tag": tag,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "aggregate": aggregated,
            "per_sample": all_sample_results,
        }

        output_path = f"{self.local_dir}/results/eval_{dataset_name}_{tag}_iter{iteration:09d}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"Saved evaluation results to {output_path}")

    def print_summary_table(self, aggregated, dataset):
        """Print summary table.

        Args:
            aggregated: Dict with aggregated statistics
            dataset: The dataset that was evaluated
        """
        dataset_name = dataset.name

        print(f"\n{'=' * 60}")
        print(f"     EVALUATION RESULTS - {dataset_name}")
        print(f"{'=' * 60}\n")

        metric_rows = []
        for key, value in aggregated.items():
            if key.endswith("_mean"):
                metric_name = key.replace("_mean", "").upper()
                std_val = aggregated.get(key.replace("_mean", "_std"), 0)
                min_val = aggregated.get(key.replace("_mean", "_min"), 0)
                max_val = aggregated.get(key.replace("_mean", "_max"), 0)

                metric_rows.append(
                    [
                        _display_name(metric_name),
                        f"{value:.4f}",
                        f"±{std_val:.4f}",
                        f"{min_val:.4f}",
                        f"{max_val:.4f}",
                    ]
                )

        print(tabulate(metric_rows, headers=["Metric", "Mean", "Std", "Min", "Max"], tablefmt="simple"))
        print(f"\n{'=' * 60}\n")
