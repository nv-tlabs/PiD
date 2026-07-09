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

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import wandb

from pid._ext.imaginaire.utils import distributed
from pid._ext.imaginaire.utils.callback import Callback

log = logging.getLogger(__name__)


@torch.jit.script
def _fused_nan_to_num(params: List[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        # Move to CPU and convert to float for accumulation
        self.state += cur_state.detach().float().cpu()
        self.iter_count += 1

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(Callback):
    """
    This callback is used to clip the gradient norm of the model.
    It also logs the average gradient norm of the model to wandb.
    """

    def __init__(
        self,
        clip_norm=1.0,
        force_finite: bool = True,
        skip_on_nonfinite: bool = True,
        skip_grad_norm_threshold: Optional[float] = None,
    ):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        # When the global grad norm is non-finite, the previous behaviour was for
        # clip_grad_norm_ to scale every grad by clip_norm/inf = 0 — silently
        # zeroing the whole step (and letting AdamW weight-decay drift the params),
        # which can self-lock training into never updating. Instead, detect a
        # non-finite (skip_on_nonfinite) or extreme-finite (skip_grad_norm_threshold,
        # disabled when None) global grad norm, log the offending params loudly, and
        # SKIP the optimizer step (drop grads so the optimizer is a true no-op).
        self.skip_on_nonfinite = skip_on_nonfinite
        self.skip_grad_norm_threshold = skip_grad_norm_threshold

        self.img_mag_log = _MagnitudeRecord()
        self.video_mag_log = _MagnitudeRecord()
        self.coupled_mag_log = _MagnitudeRecord()
        self._cur_state = None
        self._last_batch_info = {}
        self._batch_info_window = []

    def on_training_step_start(self, model, data_batch, iteration: int = 0) -> None:
        self._last_iteration = iteration
        data_type = model.return_data_type(data_batch)
        self._last_batch_info = {
            "dataset_name": data_batch.get("dataset_name"),
            "data_type": data_type,
            "__url__": data_batch.get("__url__"),
            "__key__": data_batch.get("__key__"),
        }
        self._batch_info_window.append(self._last_batch_info)
        if data_type == "image":
            self._cur_state = self.img_mag_log
        elif data_type == "video":
            self._cur_state = self.video_mag_log
        elif data_type == "coupled":
            self._cur_state = self.coupled_mag_log
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    @staticmethod
    def _global_grad_norm_and_offenders(model) -> Tuple[float, List[Tuple[float, str]]]:
        """Global grad L2 norm accumulated in fp64 (so a large-but-finite spike
        cannot overflow the way the fp32 sum-of-squares inside clip_grad_norm_
        can) plus the top-6 parameters by grad norm, for diagnostics. For DDP the
        grads are already all-reduced (identical across ranks) so this is the true
        global norm; for sharded setups it is a per-shard heuristic, which is still
        a safe trigger for skipping a non-finite step."""
        sq_sum = 0.0
        offenders: List[Tuple[float, str]] = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            pn = float(param.grad.detach().float().norm())
            sq_sum += pn * pn
            offenders.append((pn, name))
        total = math.sqrt(sq_sum) if math.isfinite(sq_sum) else float("inf")
        offenders.sort(reverse=True)
        return total, offenders[:6]

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp
        batch_info_window = list(self._batch_info_window) if self._batch_info_window else [self._last_batch_info]
        params = []

        if self.force_finite:
            for param in model.parameters():
                if param.grad is not None:
                    params.append(param.grad)
            _fused_nan_to_num(params)

        # Debug: check for inf/nan in grads AFTER force_finite but BEFORE clip
        if iteration <= 100:
            n_inf_params = 0
            inf_names = []
            for name, param in model.named_parameters():
                if param.grad is not None and (param.grad.isinf().any() or param.grad.isnan().any()):
                    n_inf_params += 1
                    if len(inf_names) < 5:
                        inf_names.append(f"{name}(inf={param.grad.isinf().sum()},nan={param.grad.isnan().sum()})")
            if n_inf_params > 0:
                log.warning(
                    f"[GradClip] iter={iteration} AFTER force_finite: {n_inf_params} params with inf/nan grads: {inf_names}"
                )
            else:
                log.info(f"[GradClip] iter={iteration} AFTER force_finite: all grads finite")

        # Robust non-finite / spike guard. Capture the offending params and an
        # fp64 global grad norm BEFORE clipping (clip_grad_norm_ would zero the
        # grads on overflow, losing the offender info). We then skip the optimizer
        # step when the grad norm is non-finite or exceeds the optional spike
        # threshold. Two overflow paths are covered: (1) the fp64 scan catches
        # element-level non-finite/huge grads; (2) clip_grad_norm_'s own fp32
        # sum-of-squares can overflow to inf even when each element is finite — the
        # exact failure that silently froze training by scaling every grad by
        # clip_norm/inf = 0. Skipping (grads -> None) makes the optimizer a true
        # no-op (no AdamW weight-decay drift) and surfaces the spike loudly.
        guard_on = self.skip_on_nonfinite or self.skip_grad_norm_threshold is not None
        offenders: List[Tuple[float, str]] = []
        robust_total = 0.0
        if guard_on:
            robust_total, offenders = self._global_grad_norm_and_offenders(model)

        total_norm = model.clip_grad_norm_(self.clip_norm)

        if guard_on:
            tn = float(total_norm) if torch.is_tensor(total_norm) else float(total_norm)
            nonfinite = (not math.isfinite(robust_total)) or (not math.isfinite(tn))
            worst = max(robust_total if math.isfinite(robust_total) else 0.0, tn if math.isfinite(tn) else 0.0)
            too_big = self.skip_grad_norm_threshold is not None and worst > self.skip_grad_norm_threshold
            if (self.skip_on_nonfinite and nonfinite) or too_big:
                log.warning(
                    f"[GradClip] iter={iteration}: SKIPPING optimizer step — "
                    f"{'non-finite' if nonfinite else 'spiking'} grad norm "
                    f"(fp64_total={robust_total:.4e}, clip_total={tn:.4e}, clip_norm={self.clip_norm}). "
                    f"Top grads: {[(f'{n:.3e}', nm) for n, nm in offenders]}. "
                    f"Accumulation window: {batch_info_window}"
                )
                for param in model.parameters():
                    param.grad = None  # AdamW skips params with grad=None -> true no-op step
                self._cur_state.update(torch.tensor(robust_total if math.isfinite(robust_total) else 0.0))
                self._batch_info_window = []
                return

        # Log warning when grad norm exceeds clip_norm, print data batch info for debugging.
        if total_norm > self.clip_norm:
            log.warning(
                f"[GradClip] iter={iteration}, grad_norm={total_norm:.4f} exceeds clip_norm={self.clip_norm}. "
                f"Accumulation window: {batch_info_window}"
            )

        self._cur_state.update(total_norm)
        if iteration % self.config.trainer.logging_iter == 0:
            avg_img_mag, avg_video_mag, avg_coupled_mag = (
                self.img_mag_log.get_stat(),
                self.video_mag_log.get_stat(),
                self.coupled_mag_log.get_stat(),
            )
            if wandb.run:
                wandb.log(
                    {
                        "clip_grad_norm/image": avg_img_mag,
                        "clip_grad_norm/video": avg_video_mag,
                        "clip_grad_norm/coupled": avg_coupled_mag,
                        "iteration": iteration,
                    },
                    step=iteration,
                )
        self._batch_info_window = []
