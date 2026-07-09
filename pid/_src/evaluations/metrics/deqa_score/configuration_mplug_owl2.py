# Vendored from DeQA-Score/src/model/configuration_mplug_owl2.py with two
# transformers-4.57+ compatibility tweaks:
#   - LlamaConfig.__init__ accepts (and silently consumes) `mlp_bias`, which
#     newer LlamaConfig requires; without this, loading the HF checkpoint
#     fails with "unexpected keyword argument 'mlp_bias'".
#   - MPLUGOwl2Config defaults `_attn_implementation` to "eager" so the model
#     never reaches DeQA's broken flash/sdpa attention code paths. We're an
#     inference-only metric on small inputs (~256 vision tokens + ~30 text
#     tokens), so eager attention is fast enough.
import copy  # noqa: F401
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING  # noqa: F401
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # noqa: F401
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # New attrs introduced by transformers >=4.45 LlamaConfig. Stored so
        # transformers internals that read them don't AttributeError.
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) < 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with at least the fields `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", self.rope_scaling.get("rope_type"))
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "default"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic', 'default'], got {rope_scaling_type}"
            )
        if rope_scaling_type != "default" and (
            rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0
        ):
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class MplugOwlVisionConfig(PretrainedConfig):
    model_type = "mplug_owl_vision_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        use_flash_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MplugOwlVisualAbstractorConfig(PretrainedConfig):
    model_type = "mplug_owl_visual_abstract"

    def __init__(
        self,
        num_learnable_queries=64,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=2816,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        encoder_hidden_size=1024,
        grid_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_learnable_queries = num_learnable_queries
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.grid_size = grid_size if grid_size else 32

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["abstractor_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


DEFAULT_VISUAL_CONFIG = {
    "visual_model": MplugOwlVisionConfig().to_dict(),
    "visual_abstractor": MplugOwlVisualAbstractorConfig().to_dict(),
}


class MPLUGOwl2Config(LlamaConfig):
    model_type = "mplug_owl2"

    def __init__(self, visual_config=None, **kwargs):
        if visual_config is None:
            self.visual_config = DEFAULT_VISUAL_CONFIG
        else:
            self.visual_config = visual_config

        # Default to eager attention. DeQA's flash/sdpa code paths reference
        # symbols removed in transformers >=4.46, so flipping this away from
        # the transformers default ("sdpa") avoids loading them. Inference is
        # cheap on a single 448×448 image — eager is fine.
        kwargs.setdefault("attn_implementation", "eager")

        super().__init__(**kwargs)


if __name__ == "__main__":
    print(MplugOwlVisionConfig().to_dict())
