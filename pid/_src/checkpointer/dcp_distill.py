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
Distributed checkpoint (DCP) for distillation models with multiple optimizers.

This checkpointer is specifically designed for distillation models that manage
their own optimizer_dict and scheduler_dict. Each optimizer/scheduler is saved
to a separate directory to avoid parameter ID mismatch issues with FSDP2.

Key differences from the base DistributedCheckpointer:
1. Saves each optimizer to a separate directory (e.g., optim_student/, optim_fake_score/)
2. Uses model.model_dict() for optimizer-to-model mapping
3. Handles multiple optimizers/schedulers in GAN-like training

Checkpoint structure:
    iter_000000005/
    ├── model/                   # Model state shards
    ├── optim_student/           # Student optimizer state
    ├── optim_fake_score/        # Fake score optimizer state (if exists)
    ├── scheduler_student/       # Student scheduler state
    ├── scheduler_fake_score/    # Fake score scheduler state (if exists)
    └── trainer/                 # Training state (grad_scaler, iteration)
"""

import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

from pid._ext.imaginaire.checkpointer.dcp import (
    AsyncMode,
    DefaultLoadPlanner,
    ModelWrapper,
    OptimizerWrapper,
)

# Import from base dcp module
from pid._ext.imaginaire.checkpointer.dcp import (
    DistributedCheckpointer as _DistributedCheckpointer,
)
from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import log, misc


class DistillationCheckpointer(_DistributedCheckpointer):
    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer] | None = None,
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        resume_keys, checkpoint_path, _is_self_resume = self.keys_to_resume_during_load()

        # Consolidated checkpoints do not contain optimizer/scheduler state and
        # therefore do not need the distillation-specific multi-optimizer path.
        # Delegate to the base implementation, which already handles .pth
        # loading and optional EMA replication. Check the resolved path rather
        # than self.load_path so a local DCP self-resume
        # still takes precedence over an explicitly configured .pth initializer.
        if checkpoint_path is not None and checkpoint_path.endswith(".pth"):
            return super().load(model, grad_scaler=grad_scaler)

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        model_dict = model.model_dict()

        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0
        _state_dict: Dict[str, Any] | None = None

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                # Self-resume must be exact; partial loads can hide missing
                # optimizer/scheduler/trainer state and corrupt continuation.
                load_planner = DefaultLoadPlanner(allow_partial_load=not _is_self_resume)

                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {checkpoint_path}")
                torch.distributed.barrier()
                log.critical(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the model...")
                    should_replicate_ema = (
                        self.config_checkpoint.replicate_ema_to_reg_in_training
                        and not _is_self_resume
                        and not self.load_training_state
                    )
                    if should_replicate_ema:
                        ckpt_metadata = storage_reader.read_metadata()
                        ckpt_keys = set(ckpt_metadata.state_dict_metadata.keys())
                        if not any(k.startswith("net_ema.") for k in ckpt_keys):
                            raise ValueError(
                                f"replicate_ema_to_reg_in_training is True, but checkpoint at "
                                f"{cur_key_ckpt_full_path} does not contain EMA weights (no 'net_ema.*' keys). "
                                f"Set replicate_ema_to_reg_in_training=False or use a checkpoint with EMA weights."
                            )
                    _model_wrapper = ModelWrapper(model, load_ema_to_reg=should_replicate_ema)
                    _state_dict = _model_wrapper.state_dict()
                    # TODO: (qsh 2025-01-23) set flag `allow_partial_load`
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    for k, v in optimizer_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info(f"- Loading the optimizer {k}...")
                        _optim_wrapper = OptimizerWrapper(model_dict[k], v)
                        _state_dict = _optim_wrapper.state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=load_planner,
                        )
                        _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    for k, v in scheduler_dict.items():
                        storage_reader = self.get_storage_reader(f"{cur_key_ckpt_full_path}_{k}")
                        log.info(f"- Loading the scheduler {k}...")
                        _state_dict = scheduler_dict[k].state_dict()
                        dcp.load(
                            _state_dict,
                            storage_reader=storage_reader,
                            planner=load_planner,
                        )
                        scheduler_dict[k].load_state_dict(_state_dict)
                elif key == "trainer":
                    storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                    log.info("- Loading the trainer state...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None and _state_dict is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.critical(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    @staticmethod
    def _sanitize_state_dict_for_dcp(state_dict: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Remove non-picklable entries from a state dict before DCP save.

        DCP's DefaultSavePlanner serializes non-tensor values via torch.save (pickle).
        Some entries (e.g. optimizer param_group values that reference Python modules)
        can fail to pickle. This method detects and removes such entries, logging warnings.
        """
        import io as _io
        import types

        # Types that are always safely picklable — skip expensive torch.save test
        _SAFE_TYPES = (int, float, bool, str, bytes, type(None))

        bad_keys = []
        checked_types: Dict[type, bool] = {}  # cache: type -> is_safe

        for fqn, obj in state_dict.items():
            # Tensors (including DTensor) are handled natively by DCP, skip them
            if isinstance(obj, torch.Tensor):
                continue
            # Primitive scalars are always safe
            if isinstance(obj, _SAFE_TYPES):
                continue
            # Quick check for obvious module references
            if isinstance(obj, types.ModuleType):
                bad_keys.append((fqn, type(obj).__name__, repr(obj)[:120]))
                continue
            # For containers (tuple, list, dict) and other types, use cache by (type, value)
            # to avoid re-checking identical param_group values repeated across all FQNs.
            obj_type = type(obj)
            if obj_type in checked_types:
                if not checked_types[obj_type]:
                    bad_keys.append((fqn, obj_type.__name__, repr(obj)[:120]))
                continue
            # Try pickling to catch transitive module references
            try:
                buf = _io.BytesIO()
                torch.save(obj, buf)
                checked_types[obj_type] = True
            except TypeError as e:
                if "cannot pickle" in str(e):
                    checked_types[obj_type] = False
                    bad_keys.append((fqn, obj_type.__name__, repr(obj)[:120]))

        if bad_keys:
            for fqn, tname, preview in bad_keys:
                log.warning(
                    f"[DCP sanitize] Removing unpicklable key from {label}: fqn={fqn}, type={tname}, preview={preview}"
                )
                state_dict.pop(fqn, None)

        return state_dict

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            if k in ["optim", "scheduler"]:
                for key_net, state_dict in v.items():
                    state_dict = self._sanitize_state_dict_for_dcp(state_dict, label=f"{k}_{key_net}")
                    storage_writer = self.get_storage_writer(f"{full_checkpoint_path}_{key_net}")
                    dcp.save(
                        state_dict,
                        storage_writer=storage_writer,
                        planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                    )
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
                )

        self._write_latest_checkpoint_file(checkpoint_file)
        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: ImaginaireModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler],
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        model_dict = model.model_dict()
        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": {k: OptimizerWrapper(model_dict[k], v).state_dict() for k, v in optimizer_dict.items()},
            "scheduler": {k: v.state_dict() for k, v in scheduler_dict.items()},
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)
