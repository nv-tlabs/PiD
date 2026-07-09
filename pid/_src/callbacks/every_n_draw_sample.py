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


import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange
from megatron.core import parallel_state

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import distributed, log, misc
from pid._ext.imaginaire.utils.easy_io import easy_io
from pid._ext.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from pid._src.callbacks.every_n import EveryN


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


def _parse_prompt_bank(txt_path: str) -> list[str]:
    prompts = []

    with open(txt_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            is_comment = line.startswith("#")
            if is_comment:
                continue
            prompts.append(line)

    return prompts


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 64,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        save_s3: bool = False,
        is_ema: bool = False,
        run_at_start: bool = False,
        name: Optional[str] = None,
        resize_wandb_image: bool = True,
    ):
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.fix_batch = fix_batch_fp
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.name = (
            self.__class__.__name__
            + ("FixBatch" if self.fix_batch is not None else "")
            + (f"{name}" if name is not None else "")
        )
        self.is_ema = is_ema
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.resize_wandb_image = resize_wandb_image

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.fix_batch is not None:
            if self.fix_batch.endswith(".txt"):
                # Prompt-only shortcut for pure T2I: fix_batch_fp points straight at a .txt prompt bank.
                txt_path = self.fix_batch
                prompts = _parse_prompt_bank(txt_path)
                if not prompts:
                    raise ValueError(f"No prompts found in fix_batch_fp txt: {txt_path}")
                line_idx = self.data_parallel_id % len(prompts)
                this_prompt = prompts[line_idx]
                self.fix_batch = {"caption": [this_prompt]}
                log.info(
                    f"loading fix_batch prompt[{line_idx}] from {txt_path} "
                    f"to rank {self.data_parallel_id}: {this_prompt[:80]}",
                    rank0_only=False,
                )
            else:
                # fix_batch_fp is a .pt path template, e.g. ".../fix_batch_{:04d}.pt".
                this_load_path = self.fix_batch.format(self.data_parallel_id % self.n_sample_to_save)
                with misc.timer(f"loading fix_batch {this_load_path}"):
                    # load_fix_batch handles bytes decode, uint8 normalize, LQ_latent batching,
                    # and image aliases.
                    from pid._src.inference.inference_utils import load_fix_batch

                    self.fix_batch = load_fix_batch(this_load_path, device="cpu")
                    log.info(f"loading fix_batch {this_load_path} to rank {self.data_parallel_id}", rank0_only=False)

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        with context():
            sample_img_fp = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            if self.fix_batch is not None:
                self.fix_batch = misc.to(self.fix_batch, "cpu")

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        torch.cuda.empty_cache()

        # Gather file paths from all data parallel ranks to rank0 for wandb upload
        # We use all_gather to collect info from all ranks, then rank 0 uploads
        # This works because we're using a shared filesystem (Lustre)
        sample_counter = getattr(trainer, "sample_counter", iteration)

        if parallel_state.is_initialized():
            is_tp_cp_pp_rank0_flag = is_tp_cp_pp_rank0()
        else:
            is_tp_cp_pp_rank0_flag = True

        # Prepare local file path info - all ranks do this
        # Only is_tp_cp_pp_rank0 processes have valid file paths
        local_file_info = {
            "dp_rank": self.data_parallel_id,
            "file_path": sample_img_fp if is_tp_cp_pp_rank0_flag else None,
        }

        # Use all_gather to collect from all ranks to all ranks
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_file_infos = [None] * world_size
            dist.all_gather_object(gathered_file_infos, local_file_info)
        else:
            gathered_file_infos = [local_file_info]

        # Only global rank 0 uploads to wandb
        if wandb.run and self.rank == 0:
            data_type_batch = self.fix_batch if self.fix_batch is not None else data_batch
            data_type = model.return_data_type(data_type_batch)
            tag = "ema" if self.is_ema else "reg"
            tag += f"_{data_type}"

            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }

            # Filter valid file infos (non-None file paths). `sample()` normally
            # returns a single path, but diagnostics may return `{group: path}`
            # to log multiple image groups from one callback invocation.
            # Files are on shared filesystem, so rank 0 can access them
            valid_file_infos = []

            for file_info in gathered_file_infos:
                if file_info and file_info["file_path"]:
                    file_paths = file_info["file_path"]
                    if isinstance(file_paths, dict):
                        path_items = file_paths.items()
                    elif isinstance(file_paths, (list, tuple)):
                        path_items = [(f"path_{idx:02d}", path) for idx, path in enumerate(file_paths)]
                    else:
                        path_items = [("samples", file_paths)]

                    for group_key, file_path in path_items:
                        if file_path and os.path.exists(file_path):
                            valid_file_infos.append(
                                {
                                    "dp_rank": file_info["dp_rank"],
                                    "group_key": str(group_key),
                                    "file_path": file_path,
                                }
                            )
                        else:
                            log.warning(f"File not found for dp_rank {file_info['dp_rank']}: {file_path}")

            if valid_file_infos:
                grouped_file_infos = {}
                for file_info in valid_file_infos:
                    grouped_file_infos.setdefault(file_info["group_key"], []).append(file_info)

                total_uploaded = 0
                for group_key, group_infos in sorted(grouped_file_infos.items()):
                    group_infos = sorted(group_infos, key=lambda x: x["dp_rank"])[: self.n_sample_to_save]
                    media_list = []
                    for file_info in group_infos:
                        dp_rank = file_info["dp_rank"]
                        file_path = file_info["file_path"]
                        caption = f"step{sample_counter}_dp{dp_rank}_{group_key}"

                        media_list.append(wandb.Image(file_path, caption=caption))

                    suffix = "samples" if group_key == "samples" else f"{group_key}_samples"
                    info[f"{self.name}/{tag}_{suffix}"] = media_list
                    total_uploaded += len(group_infos)

                wandb.log(info, step=iteration)
                log.info(f"Uploaded {total_uploaded} samples to wandb at step {sample_counter}")

        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(
        self,
        trainer,
        model,
        data_batch,
        output_batch,
        loss,
        iteration,
        image_size=None,
        prompt_only_sample: bool = False,
    ):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        is_pid = hasattr(model, "encode_lq_latent")
        if self.fix_batch is not None:
            # fix_batch was already processed by load_fix_batch in on_train_start
            # (bytes decoded, LQ_latent batched, image aliases added). Copy + move to device.
            data_batch = dict(self.fix_batch)
            data_batch = misc.to(data_batch, **model.tensor_kwargs)
        else:
            data_batch = dict(data_batch)  # shallow-copy to avoid mutating caller's dict
            # PiD training mutates LQ_latent with training noise. Re-encode the
            # existing LQ image here so callback inference uses a clean latent.
            if is_pid:
                lq_image = data_batch.get("LQ_video_or_image")
                if not isinstance(lq_image, torch.Tensor):
                    raise ValueError("PiD training-batch sampling requires LQ_video_or_image")
                data_batch["LQ_latent"] = model.encode_lq_latent(lq_image).contiguous().to(**model.tensor_kwargs)
                data_batch["degrade_sigma"] = torch.zeros(
                    data_batch["LQ_latent"].shape[0], device=data_batch["LQ_latent"].device, dtype=torch.float32
                )

        if prompt_only_sample:
            # Multi-resolution sampling should reuse the batch captions but not
            # let the current training image shape override the requested image_size.
            input_data_key = getattr(model.config, "input_data_key", None)
            if input_data_key is not None:
                data_batch.pop(input_data_key, None)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch, return_latent_state=False)

        to_show = []

        if is_pid:
            # LQ pixels are callback-only visualization data. They are never
            # included in the model-facing PiD inference batch below.
            lq_img = data_batch.get("LQ_video_or_image")
            if not isinstance(lq_img, torch.Tensor):
                raise ValueError("PiD visualization requires LQ_video_or_image")
            if lq_img.ndim == 5:
                if lq_img.shape[2] != 1:
                    raise ValueError(f"Expected single-frame LQ input, got {tuple(lq_img.shape)}")
                lq_img = lq_img[:, :, 0]
            if lq_img.ndim != 4:
                raise ValueError(f"Expected image LQ tensor [B, C, H, W], got {tuple(lq_img.shape)}")
            if raw_data is not None:
                target_h, target_w = raw_data.shape[-2:]
                lq_img = F.interpolate(lq_img.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)
            to_show.append(lq_img.unsqueeze(2).float().cpu())
        elif raw_data is not None and hasattr(condition, "LQ_video_or_image_upscaled"):
            to_show.append(condition.LQ_video_or_image_upscaled.float().cpu())
        elif (
            raw_data is not None and hasattr(condition, "lq_video_or_image") and condition.lq_video_or_image is not None
        ):
            # PixelDiT SR LQ input is [B, C, H_lq, W_lq]; upsample spatially to GT res.
            lq_img = condition.lq_video_or_image.float()
            if lq_img.ndim != 4:
                raise ValueError(f"Expected image LQ tensor [B, C, H, W], got {tuple(lq_img.shape)}")
            target_h, target_w = raw_data.shape[-2], raw_data.shape[-1]
            lq_img = F.interpolate(lq_img, size=(target_h, target_w), mode="bicubic", align_corners=False)
            to_show.append(lq_img.unsqueeze(2).cpu())  # [B, C, 1, H, W]

        # Passing shift=None lets models with dynamic_shift compute shift from the
        # actual inference H/W; models without dynamic_shift keep the legacy fixed shift.
        sample_shift = None if getattr(model.config, "dynamic_shift", None) is not None else model.config.shift

        sampling_batch = data_batch
        if is_pid:
            sampling_batch = {
                model.config.input_caption_key: data_batch[model.config.input_caption_key],
                "LQ_latent": data_batch["LQ_latent"],
                "degrade_sigma": data_batch["degrade_sigma"],
            }

        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                sampling_batch,
                cfg_scale=guidance,
                # make sure no mismatch and also works for cp
                # state_shape=x0.shape[1:],
                # n_sample=x0.shape[0],
                shift=sample_shift,
                num_steps=self.num_sampling_step,
                image_size=image_size,
            )
            # Clean up CUDA memory to avoid OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            to_show.append(sample.float().cpu())

        # raw_data is None for prompt-only fix batches: there is no GT
        # to show, so the grid holds only the generated samples.
        if raw_data is not None:
            to_show.append(raw_data.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = raw_data.shape[0] if raw_data is not None else to_show[0].shape[0]

        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        else:
            local_path = None

        return local_path

    @staticmethod
    def _stack_to_grid(samples: torch.Tensor) -> torch.Tensor:
        """Arrange [n, b, c, t, h, w] samples into a single [t, c, H, W] grid.

        Default layout stacks the n entries vertically and the b batch entries
        horizontally -> (n h) (b w). For the common single-sample case (b == 1) a
        tall single column is awkward to view, so we instead lay the n entries out
        in two columns (row-major: entry i -> row i//2, col i%2), padding with a
        black tile when n is odd so the grid stays rectangular.
        """
        n, b = samples.shape[0], samples.shape[1]
        if b == 1 and n > 1:
            if n % 2 == 1:
                pad = torch.zeros_like(samples[:1])
                samples = torch.cat([samples, pad], dim=0)  # [n+1, b, c, t, h, w]
            return rearrange(samples, "(rows col) b c t h w -> t c (rows h) (b col w)", col=2)
        return rearrange(samples, "n b c t h w -> t c (n h) (b w)")

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w], range in [0, 1]
        is_image_output = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        image_file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_image_path = f"{self.local_dir}/{image_file_base_fp}"

        if not is_image_output:
            raise ValueError(f"EveryNDrawSample only supports image outputs with T=1, got T={to_show.shape[3]}")

        image_tensor = self._stack_to_grid(to_show[:, :n_viz_sample])
        image_grid = torchvision.utils.make_grid(image_tensor, nrow=1, padding=0, normalize=False)
        if self.resize_wandb_image:
            image_grid = resize_image(image_grid, 2048)
        torchvision.utils.save_image(image_grid, local_image_path, nrow=1, scale_each=True)

        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            easy_io.copyfile(local_image_path, f"s3://rundir/{self.name}/{image_file_base_fp}")

        return local_image_path


class EveryNDrawSampleMultiResolution(EveryNDrawSample):
    def __init__(
        self,
        every_n: int,
        image_sizes: List[int],
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 64,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        save_s3: bool = False,
        is_ema: bool = False,
        run_at_start: bool = False,
        name: Optional[str] = None,
        resize_wandb_image: bool = True,
        prompt_only_sample: bool = True,
    ):
        if not image_sizes:
            raise ValueError("EveryNDrawSampleMultiResolution requires at least one image_size")
        self.image_sizes = [self._normalize_square_image_size(size) for size in image_sizes]
        self.prompt_only_sample = prompt_only_sample
        self._current_image_size = None
        super().__init__(
            every_n=every_n,
            step_size=step_size,
            fix_batch_fp=fix_batch_fp,
            n_viz_sample=n_viz_sample,
            n_sample_to_save=n_sample_to_save,
            num_sampling_step=num_sampling_step,
            guidance=guidance,
            save_s3=save_s3,
            is_ema=is_ema,
            run_at_start=run_at_start,
            name=name,
            resize_wandb_image=resize_wandb_image,
        )

    @staticmethod
    def _normalize_square_image_size(image_size) -> int:
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise ValueError(
                "EveryNDrawSampleMultiResolution currently supports square integer image sizes only, "
                f"got {image_size!r}"
            )
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        return int(image_size)

    @staticmethod
    def _image_size_tag(image_size: int) -> str:
        return f"res{image_size}"

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        base_name = self.name
        base_local_dir = self.local_dir
        try:
            for image_size in self.image_sizes:
                res_tag = self._image_size_tag(image_size)
                self._current_image_size = image_size
                self.name = f"{base_name}_{res_tag}"
                self.local_dir = f"{base_local_dir}/{res_tag}"
                os.makedirs(self.local_dir, exist_ok=True)
                super().every_n_impl(trainer, model, data_batch, output_batch, loss, iteration)
        finally:
            self._current_image_size = None
            self.name = base_name
            self.local_dir = base_local_dir

    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        return super().sample(
            trainer,
            model,
            data_batch,
            output_batch,
            loss,
            iteration,
            image_size=self._current_image_size,
            prompt_only_sample=self.prompt_only_sample,
        )
