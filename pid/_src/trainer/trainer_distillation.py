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
Distillation trainer for models with multiple optimizers.

This trainer is designed for distillation models (e.g., DMD) that manage their own
optimizer_dict and scheduler_dict. Key differences from the base ImaginaireTrainer:

1. Calls model.init_optimizer_scheduler() which returns primary optimizer but
   model stores all optimizers internally in optimizer_dict
2. Uses model.optimizer_dict and model.scheduler_dict for checkpointing
3. Calls model.optimizers_schedulers_step() and model.optimizers_zero_grad()
   instead of directly stepping the optimizer

Interface the model must implement:
- init_optimizer_scheduler(optimizer_config, scheduler_config) -> (primary_opt, primary_sched)
- optimizer_dict: Dict[str, torch.optim.Optimizer]
- scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler]
- model_dict() -> Dict[str, nn.Module]
- optimizers_zero_grad(iteration: int) -> None
- optimizers_schedulers_step(grad_scaler, iteration: int) -> None
- is_student_phase(iteration: int) -> bool
"""

import torch
import torch.utils.data

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.trainer import ImaginaireTrainer
from pid._ext.imaginaire.utils import distributed, log, misc
from pid._ext.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


class DistillationTrainer(ImaginaireTrainer):
    """
    Trainer for distillation models with multiple optimizers.

    This trainer expects the model to manage its own optimizer_dict and scheduler_dict.
    The checkpointer must be a DistillationCheckpointer that accepts these dicts.
    """

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """
        Training loop for distillation models.

        Key differences from base trainer:
        1. Model manages optimizer_dict/scheduler_dict internally
        2. Checkpointer uses model.optimizer_dict and model.scheduler_dict
        3. Model handles optimizer stepping via optimizers_schedulers_step()
        """
        # Move model to GPU and initialize
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize optimizers - model stores all in optimizer_dict/scheduler_dict
        self.callbacks.on_optimizer_init_start()
        primary_optimizer, _ = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        self.optimizer = primary_optimizer  # keep the base trainer/callback contract
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()

        # Load checkpoint using optimizer_dict and scheduler_dict
        iteration = self.checkpointer.load(
            model,
            model.optimizer_dict,
            model.scheduler_dict,
            grad_scaler,
        )

        grad_accum_iter = 0
        log.info(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        if self.config.trainer.distributed_parallelism == "ddp":
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting distillation training...")
        self.callbacks.on_train_start(model, iteration=iteration)

        # Initial validation
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)

        _end_training = False
        _smart_stop_triggered = False
        _oom_triggered = False

        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            while True:
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)

                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break

                    data_batch = misc.to(data_batch, device="cuda")

                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)

                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."

                    try:
                        output_batch, loss, grad_accum_iter = self.training_step(
                            model_ddp,
                            grad_scaler,
                            data_batch,
                            iteration=iteration,
                            grad_accum_iter=grad_accum_iter,
                        )
                    except torch.OutOfMemoryError as e:
                        _oom_triggered = True
                        log.error(f"CUDA Out of Memory error caught: {e}")
                        log.info("Saving checkpoint due to CUDA OOM error...")
                        self.checkpointer.save(
                            model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration
                        )
                        log.success("Checkpoint saved successfully. Exiting training gracefully due to OOM.")
                        _end_training = True
                        break

                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )

                    if grad_accum_iter != 0:
                        continue

                    iteration += 1

                    # Save checkpoint
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(
                            model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration
                        )

                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)

                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)

                    self.straggler_detector.generate_report(iteration)

                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()

                if _end_training:
                    break

        log.success("Done with distillation training.")
        if iteration % self.config.checkpoint.save_iter != 0 and not _smart_stop_triggered and not _oom_triggered:
            self.checkpointer.save(model, model.optimizer_dict, model.scheduler_dict, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """
        Training step for distillation models.

        Key differences from base trainer:
        1. Model handles its own optimizer selection via get_current_optimizer_scheduler()
        2. Model handles optimizer stepping via optimizers_schedulers_step()
        3. Model handles zero_grad via optimizers_zero_grad()
        """
        if self.config.trainer.distributed_parallelism == "ddp":
            _model = model_ddp.module
        else:
            _model = model_ddp

        # Get current optimizer for gradient accumulation sync
        current_optimizer, current_scheduler = _model.get_current_optimizer_scheduler(iteration)

        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                with self.straggler_detector.profile_section(
                    "fwd", self.config.trainer.straggler_detection.analyze_forward
                ):
                    output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)

            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                with self.straggler_detector.profile_section(
                    "bwd", self.config.trainer.straggler_detection.analyze_backward
                ):
                    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                    loss_scaled.backward()
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_after_backward()
                    else:
                        model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)

        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model_ddp, current_optimizer, current_scheduler, grad_scaler, iteration=iteration
                    )

                    # Let model handle optimizer step (it knows which optimizer to use)
                    _model.optimizers_schedulers_step(grad_scaler, iteration=iteration)

                    self.callbacks.on_before_zero_grad(
                        model_ddp, current_optimizer, current_scheduler, iteration=iteration
                    )

                    # Let model handle EMA update before zero_grad
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_before_zero_grad(current_optimizer, current_scheduler, iteration=iteration)
                    else:
                        model_ddp.on_before_zero_grad(current_optimizer, current_scheduler, iteration=iteration)

                    # Let model handle zero_grad
                    _model.optimizers_zero_grad(iteration=iteration)

            grad_accum_iter = 0

        return output_batch, loss, grad_accum_iter
