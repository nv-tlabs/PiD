# Flow DPM-Solver wrapper for PixelDiT inference.
# Copied from pixeldits/pixeldit/t2i/diffusion/model/flow_dpm.py with updated imports.

import torch

from . import gaussian_diffusion as gd
from .dpm_solver import DPM_Solver, NoiseScheduleFlow, NoiseScheduleVP, model_wrapper


def DPMS(
    model,
    condition,
    uncondition,
    cfg_scale,
    model_type="noise",  # or "x_start" or "v" or "score", "flow"
    noise_schedule="linear",
    guidance_type="classifier-free",
    model_kwargs=None,
    diffusion_steps=1000,
    schedule="VP",
    interval_guidance=None,
):
    if model_kwargs is None:
        model_kwargs = {}
    if interval_guidance is None:
        interval_guidance = [0, 1.0]

    betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))

    # 1) Noise schedule
    if schedule == "VP":
        noise_schedule_obj = NoiseScheduleVP(schedule="discrete", betas=betas)
    elif schedule == "FLOW":
        noise_schedule_obj = NoiseScheduleFlow(schedule="discrete_flow")
    else:
        raise ValueError(f"Unsupported schedule {schedule}")

    # 2) Wrap model for continuous-time solver
    model_fn = model_wrapper(
        model,
        noise_schedule_obj,
        model_type=model_type,
        model_kwargs=model_kwargs,
        guidance_type=guidance_type,
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
        interval_guidance=interval_guidance,
    )

    # 3) Return solver
    return DPM_Solver(model_fn, noise_schedule_obj, algorithm_type="dpmsolver++")
