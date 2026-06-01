from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment_2kto4k.shared_config import (
    _common_model_overrides_2kto4k,
)


def _sdxl_2kto4k_distill_experiment(name: str) -> LazyDict:
    """SDXL needs explicit net override: the default `pid_sr4x` net is sized for
    16-ch VAEs (Flux1 / SD3), but the SDXL VAE is 4-ch. `state_ch=4` enforces
    the VAE/model channel-count assertion; `net.lq_latent_channels=4` resizes the
    LQ-latent projection input conv to match."""
    cfg = _common_model_overrides_2kto4k(state_ch=4)
    cfg["net"] = {**cfg["net"], "lq_latent_channels": 4}
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "ddp_distill_pid"},
                {"override /net": "pid_sr4x"},
                {"override /conditioner": "pid_caption_lq"},
                {"override /ckpt_type": "dcp"},
                {"override /ema": None},
                {"override /checkpoint": "local"},
                {"override /tokenizer": "sdxl_vae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            model=dict(config=cfg),
        ),
    )


PID_RES2KTO4K_SR4X_OFFICIAL_SDXL_DISTILL_4STEP = _sdxl_2kto4k_distill_experiment(
    "PiD_res2kto4k_sr4x_official_sdxl_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2KTO4K_SR4X_OFFICIAL_SDXL_DISTILL_4STEP["job"]["name"],
    node=PID_RES2KTO4K_SR4X_OFFICIAL_SDXL_DISTILL_4STEP,
)
