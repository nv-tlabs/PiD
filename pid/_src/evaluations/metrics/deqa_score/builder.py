# Vendored from DeQA-Score/src/model/builder.py, trimmed to the only loading
# path we need: the full-tuning fp16 checkpoint at
# `zhiyuanyou/DeQA-Score-Mix3` (no LoRA, no 4-bit / 8-bit quantization).
import torch
from transformers import AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from .modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name="mplug_owl2",
    device_map="auto",
    device="cuda",
    preprocessor_path=None,
):
    if preprocessor_path is None:
        preprocessor_path = model_path

    if device != "cuda":
        device_map = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(preprocessor_path, use_fast=False)
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    image_processor = CLIPImageProcessor.from_pretrained(preprocessor_path)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
