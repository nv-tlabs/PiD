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
WandB callback for distillation training.

Extends WandbCallback to log additional distillation-specific losses:
- vsd_loss: Variational Score Distillation loss (student updates)
- dsm_loss: Denoising Score Matching loss (fake_score updates)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import wandb

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import distributed, log
from pid._ext.imaginaire.utils.easy_io import easy_io
from pid._src.callbacks.wandb_log import WandbCallback, _LossRecord


@dataclass
class _DistillLossRecord(_LossRecord):
    """Extended loss record for distillation training.

    Each loss tracks its own accumulation count so that averaging is correct
    regardless of which training phase (student / fake_score / joint) produced it.
    In DMD2 alternating mode, student updates produce vsd_loss and gan_loss_gen,
    while fake_score updates produce dsm_loss, gan_loss_disc, and gan_loss_r1.
    """

    # Distillation losses and per-loss counts
    vsd_loss: float = 0
    vsd_count: int = 0
    dsm_loss: float = 0
    dsm_count: int = 0
    # GAN losses (Projected GAN / APT from DMD2)
    gan_loss_gen: float = 0
    gan_gen_count: int = 0
    gan_loss_disc: float = 0
    gan_disc_count: int = 0
    gan_loss_r1: float = 0
    gan_r1_count: int = 0

    def reset(self) -> None:
        super().reset()
        self.vsd_loss = 0
        self.vsd_count = 0
        self.dsm_loss = 0
        self.dsm_count = 0
        self.gan_loss_gen = 0
        self.gan_gen_count = 0
        self.gan_loss_disc = 0
        self.gan_disc_count = 0
        self.gan_loss_r1 = 0
        self.gan_r1_count = 0

    def _reduce_loss(self, loss_value: float, count: int) -> float:
        """Reduce a loss value across all ranks."""
        if count == 0:
            return 0.0
        avg = loss_value / count
        # Convert to tensor if needed to ensure all_reduce works correctly
        if isinstance(avg, (int, float)):
            avg = torch.tensor(avg, device="cuda")
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        return avg.item()

    def get_stat(self) -> dict:
        """Return a dictionary of all averaged loss statistics."""
        stats = {}
        if self.iter_count > 0:
            avg_loss = self.loss / self.iter_count
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            stats["loss"] = avg_loss.item()
        else:
            stats["loss"] = 0.0

        # Each loss is averaged by its own count
        if self.vsd_count > 0:
            stats["vsd_loss"] = self._reduce_loss(self.vsd_loss, self.vsd_count)
        if self.dsm_count > 0:
            stats["dsm_loss"] = self._reduce_loss(self.dsm_loss, self.dsm_count)
        if self.gan_gen_count > 0 and self.gan_loss_gen != 0:
            stats["gan_loss_gen"] = self._reduce_loss(self.gan_loss_gen, self.gan_gen_count)
        if self.gan_disc_count > 0 and self.gan_loss_disc != 0:
            stats["gan_loss_disc"] = self._reduce_loss(self.gan_loss_disc, self.gan_disc_count)
        if self.gan_r1_count > 0 and self.gan_loss_r1 != 0:
            stats["gan_loss_r1"] = self._reduce_loss(self.gan_loss_r1, self.gan_r1_count)

        self.reset()
        return stats


class WandbDistillCallback(WandbCallback):
    """
    WandB callback for distillation training.

    Extends WandbCallback to log additional distillation-specific losses:
    - vsd_loss, dsm_loss, and GAN losses
    """

    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__(
            logging_iter_multipler=logging_iter_multipler,
            save_logging_iter_multipler=save_logging_iter_multipler,
            save_s3=save_s3,
        )
        # Override with distillation-specific loss records
        self.train_image_log = _DistillLossRecord()
        self.train_video_log = _DistillLossRecord()
        self.final_loss_log = _DistillLossRecord()
        self.name = "wandb_distill_log" + self.wandb_extra_tag

    def _accumulate_distill_losses(
        self,
        record: _DistillLossRecord,
        vsd_loss: torch.Tensor | None,
        dsm_loss: torch.Tensor | None,
        gan_loss_gen: torch.Tensor | None = None,
        gan_loss_disc: torch.Tensor | None = None,
        gan_loss_r1: torch.Tensor | None = None,
    ) -> None:
        """Accumulate distillation losses into the record.

        Each loss tracks its own count so averaging is correct regardless of
        which phase (student / fake_score / joint) produced it.
        """
        if vsd_loss is not None:
            record.vsd_loss += vsd_loss.detach().float()
            record.vsd_count += 1
        if dsm_loss is not None:
            record.dsm_loss += dsm_loss.detach().float()
            record.dsm_count += 1

        # GAN losses: each tracks its own count
        if gan_loss_gen is not None:
            record.gan_loss_gen += gan_loss_gen.detach().float()
            record.gan_gen_count += 1
        if gan_loss_disc is not None:
            record.gan_loss_disc += gan_loss_disc.detach().float()
            record.gan_disc_count += 1
        if gan_loss_r1 is not None:
            record.gan_loss_r1 += gan_loss_r1.detach().float()
            record.gan_r1_count += 1

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            # Handle distillation losses
            vsd_loss_value = output_batch.get("vsd_loss")
            dsm_loss_value = output_batch.get("dsm_loss")
            gan_loss_gen_value = output_batch.get("gan_loss_gen")
            gan_loss_disc_value = output_batch.get("gan_loss_disc")
            gan_loss_r1_value = output_batch.get("gan_loss_r1")

            if model.is_image_batch(data_batch):
                self.train_image_log.loss += loss.detach().float()
                self.train_image_log.iter_count += 1
                self._accumulate_distill_losses(
                    self.train_image_log,
                    vsd_loss_value,
                    dsm_loss_value,
                    gan_loss_gen_value,
                    gan_loss_disc_value,
                    gan_loss_r1_value,
                )
            if not model.is_image_batch(data_batch):
                self.train_video_log.loss += loss.detach().float()
                self.train_video_log.iter_count += 1
                self._accumulate_distill_losses(
                    self.train_video_log,
                    vsd_loss_value,
                    dsm_loss_value,
                    gan_loss_gen_value,
                    gan_loss_disc_value,
                    gan_loss_r1_value,
                )

            self.final_loss_log.loss += loss.detach().float()
            self.final_loss_log.iter_count += 1
            self._accumulate_distill_losses(
                self.final_loss_log,
                vsd_loss_value,
                dsm_loss_value,
                gan_loss_gen_value,
                gan_loss_disc_value,
                gan_loss_r1_value,
            )
        else:
            if model.is_image_batch(data_batch):
                self.img_unstable_count += 1
            if not model.is_image_batch(data_batch):
                self.video_unstable_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            if self.logging_iter_multipler > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()
            image_stats = self.train_image_log.get_stat()
            video_stats = self.train_video_log.get_stat()
            final_stats = self.final_loss_log.get_stat()

            dist.all_reduce(self.img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                info = {f"timer/{key}": value for key, value in timer_results.items()}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/video_loss": video_stats["loss"],
                        f"train{self.wandb_extra_tag}/loss": final_stats["loss"],
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )
                # Add distillation losses if present
                for loss_key in [
                    "vsd_loss",
                    "dsm_loss",
                    "gan_loss_gen",
                    "gan_loss_disc",
                    "gan_loss_r1",
                ]:
                    if loss_key in final_stats:
                        info[f"train{self.wandb_extra_tag}/{loss_key}"] = final_stats[loss_key]
                    if loss_key in image_stats:
                        info[f"train{self.wandb_extra_tag}/image_{loss_key}"] = image_stats[loss_key]
                    if loss_key in video_stats:
                        info[f"train{self.wandb_extra_tag}/video_{loss_key}"] = video_stats[loss_key]
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
            if self.logging_iter_multipler == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.img_unstable_count.zero_()
            self.video_unstable_count.zero_()
