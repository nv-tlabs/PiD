from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict
from pid._src.configs.pid.experiment_2kto4k.shared_config import (
    _common_model_overrides_2kto4k,
)


def _qwenimage_2kto4k_distill_experiment(name: str) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "ddp_distill_pid"},
                {"override /net": "pid_sr4x"},
                {"override /conditioner": "pid_caption_lq"},
                {"override /ckpt_type": "dcp"},
                {"override /ema": None},
                {"override /checkpoint": "local"},
                {"override /tokenizer": "qwenimage_vae_tokenizer"},
                "_self_",
            ],
            job=dict(group="pid_official", name=name),
            # Qwen-Image latent has 16 channels (3D VAE; T=1 in image mode).
            model=dict(config=_common_model_overrides_2kto4k(state_ch=16)),
        ),
    )


# Single unified inference experiment for both Qwen-Image and Qwen-Image-2512:
# the PiD student is the same; only the upstream LDM transformer differs.
PID_RES2KTO4K_SR4X_OFFICIAL_QWENIMAGE_DISTILL_4STEP = _qwenimage_2kto4k_distill_experiment(
    "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step"
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=PID_RES2KTO4K_SR4X_OFFICIAL_QWENIMAGE_DISTILL_4STEP["job"]["name"],
    node=PID_RES2KTO4K_SR4X_OFFICIAL_QWENIMAGE_DISTILL_4STEP,
)
