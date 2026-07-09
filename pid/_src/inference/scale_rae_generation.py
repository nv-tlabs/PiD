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
Scale-RAE T2I generation helpers for the from_ldm --backbone siglip demo.

Loads the Scale-RAE Qwen 1.5B + DiT 2.4B latent-diffusion model
(`nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B`) plus the SigLIP-2 ViT-XL
`MultimodalDecoder`, and provides a text-prompt → (latent, image) generation
function. Optional `--save_xt_steps K1 K2 …` snapshots the diffusion
trajectory by monkey-patching `model.diff_head.inference_flow.p_sample_loop`
(no upstream code is modified).

Requires the upstream Scale-RAE GitHub repo
(https://github.com/ZitengWangNYU/Scale-RAE) cloned somewhere on disk and
installed via ``pip install --no-deps -e .``. The location is resolved
from the ``SCALE_RAE_REPO_PATH`` environment variable (default:
``../Scale-RAE``, i.e. cloned as a sibling of the pid working tree); the
``--scale_rae_repo_path`` CLI flag falls back to that default.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

from pid._src.tokenizers.scale_rae_decoder import GeneralDecoder

logger = logging.getLogger(__name__)

# Default location: ``$SCALE_RAE_REPO_PATH`` if set, else ``../Scale-RAE``
# relative to CWD (the README convention is to clone Scale-RAE as a sibling
# of the pid working tree).
DEFAULT_SCALE_RAE_REPO_PATH = os.environ.get("SCALE_RAE_REPO_PATH", "../Scale-RAE")


class _LocalMultimodalDecoder(nn.Module):
    """Drop-in for `scale_rae.model.multimodal_decoder.MultimodalDecoder` that
    uses our local `GeneralDecoder` (which is transformers-4.57+ compatible,
    unlike the upstream copy that pins an older API).

    Wraps the same I/O contract: input (B, N+1, C) image features (with a
    leading placeholder CLS), output (B, 3, H, W) in [0, 1] after applying the
    encoder's image_std/mean denormalization.
    """

    def __init__(
        self,
        pretrained_encoder_path: str,
        general_decoder_config: str,
        num_patches: int,
        drop_cls_token: bool = True,
        decoder_path: Optional[str] = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor  # noqa: E402
        from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig  # noqa: E402

        with open(general_decoder_config) as f:
            cfg_dict = json.load(f)

        encoder_cfg = AutoConfig.from_pretrained(pretrained_encoder_path)
        if hasattr(encoder_cfg, "vision_config"):
            cfg_dict["hidden_size"] = encoder_cfg.vision_config.hidden_size
        else:
            cfg_dict["hidden_size"] = encoder_cfg.hidden_size

        cfg = ViTMAEConfig(**cfg_dict)
        self.decoder = GeneralDecoder(cfg, num_patches=num_patches)
        self.drop_cls_token = drop_cls_token

        proc = AutoImageProcessor.from_pretrained(pretrained_encoder_path)
        self.register_buffer("image_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(proc.image_std).view(1, 3, 1, 1))

        if decoder_path is not None:
            sd = torch.load(decoder_path, map_location="cpu")
            missing, unexpected = self.decoder.load_state_dict(sd, strict=False)
            if missing:
                print(f"[scale_rae decoder] missing keys: {len(missing)} (first 3): {missing[:3]}")
            if unexpected:
                print(f"[scale_rae decoder] unexpected keys: {len(unexpected)} (first 3): {unexpected[:3]}")

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        out = self.decoder(zs, drop_cls_token=self.drop_cls_token)
        logits = out.logits
        pixels = self.decoder.unpatchify(logits)
        return pixels * self.image_std.to(pixels) + self.image_mean.to(pixels)


def _ensure_scale_rae_on_path(repo_path: str) -> None:
    """Insert <repo_path> into sys.path so the scale_rae package resolves."""
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Scale-RAE repo not found: {repo_path}. Set SCALE_RAE_REPO_PATH or "
            f"pass --scale_rae_repo_path. See README for installation instructions."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


# Default HF repo holding the SigLIP-2 ViT-XL decoder weights + config.
# Mirrors `inference/cli.py:DEFAULT_DECODER_REPO` upstream.
_DEFAULT_DECODER_HF_REPO = "nyu-visionx/siglip2_decoder"


def _resolve_decoder_paths(
    decoder_config_path: Optional[str],
    decoder_ckpt: Optional[str],
    repo_path: str,
) -> tuple[str, str]:
    """Locate the decoder config + weights. Search order:

    1. Explicit user-supplied paths (must exist).
    2. ``<scale_rae_repo>/decoder/{XL_decoder_config.json, siglip2_sop14_i224_web73M_ganw3_decXL.pt}``.
    3. Download from HuggingFace ``nyu-visionx/siglip2_decoder`` (`config.json`, `model.pt`).
    """
    from huggingface_hub import hf_hub_download  # noqa: E402

    # 1. Explicit paths
    if decoder_config_path and decoder_ckpt:
        if not os.path.exists(decoder_config_path):
            raise FileNotFoundError(f"--scale_rae_decoder_config not found: {decoder_config_path}")
        if not os.path.exists(decoder_ckpt):
            raise FileNotFoundError(f"--scale_rae_decoder_ckpt not found: {decoder_ckpt}")
        return decoder_config_path, decoder_ckpt

    # 2. Co-located in repo
    repo_cfg = os.path.join(repo_path, "decoder", "XL_decoder_config.json")
    repo_pt = os.path.join(repo_path, "decoder", "siglip2_sop14_i224_web73M_ganw3_decXL.pt")
    if os.path.exists(repo_cfg) and os.path.exists(repo_pt):
        return repo_cfg, repo_pt

    # 3. HF fallback
    print(f"Downloading decoder from HuggingFace repo {_DEFAULT_DECODER_HF_REPO} ...")
    cfg = hf_hub_download(repo_id=_DEFAULT_DECODER_HF_REPO, filename="config.json")
    pt = hf_hub_download(repo_id=_DEFAULT_DECODER_HF_REPO, filename="model.pt")
    return cfg, pt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scale_rae_stack(
    repo_path: str,
    model_path: str,
    decoder_config_path: str,
    decoder_ckpt: str,
    pretrained_encoder_path: str = "google/siglip2-so400m-patch14-224",
    decoder_num_patches: int = 256,
    drop_cls_token: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load (tokenizer, model, decoder) for Scale-RAE T2I.

    `model` is the Qwen LM with the diffusion DiT head built in. `decoder` is
    the SigLIP-2 ViT-XL `MultimodalDecoder` (it bundles SigLIP image_std/mean
    denorm + unpatchify so its output is already in [0, 1]).
    """
    _ensure_scale_rae_on_path(repo_path)

    # Lazy imports — scale_rae is only on sys.path now.
    from scale_rae.mm_utils import get_model_name_from_path  # noqa: E402
    from scale_rae.model.builder import load_pretrained_model  # noqa: E402
    from scale_rae.utils import disable_torch_init  # noqa: E402

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device,
        device_map={"": device},
        torch_dtype=dtype,
    )

    # Compat shim 1: scale_rae_qwen2.py:237 reads `self._attn_implementation`,
    # which moved to `self.config._attn_implementation` in transformers 4.40+.
    inner = model.model if hasattr(model, "model") else model
    attn_impl = getattr(inner.config, "_attn_implementation", "sdpa") or "sdpa"
    for m in [model, inner]:
        try:
            object.__setattr__(m, "_attn_implementation", attn_impl)
        except Exception:
            pass

    # Compat shim 2: ScaleRAEQwenModel.forward (`scale_rae_qwen2.py:171`) is a
    # near-verbatim copy of the transformers-4.37 Qwen2Model.forward and breaks
    # against the modern (4.45+ / 5.x) Qwen2DecoderLayer API. The inner-model
    # call inside ScaleRAEQwenForCausalLM.forward (line 745) only ever passes
    # *vanilla* Qwen2Model kwargs at inference time — the scale_rae-only
    # vision_tower_aux_* path is gated by `if hasattr(self, 'vision_tower_aux_feature_list'): raise NotImplementedError`,
    # so we delegate `inner.forward` to the upstream Qwen2Model.forward bound
    # to the same instance. Tracks transformers API drift automatically.
    #
    # One pre-processing wart: ScaleRAEQwenForCausalLM.greedy_decode passes a
    # `(1, 1)` sentinel attention_mask each iteration (scale_rae_qwen2.py:1246)
    # regardless of how long inputs_embeds has grown. The legacy Qwen2Model
    # silently rebuilt a fresh causal mask in that case; the modern one feeds
    # the malformed mask straight into SDPA, which then trips
    # `(*bias): last dimension must be contiguous`. Drop the sentinel here so
    # the upstream mask-builder constructs the right 4D causal mask from scratch.
    import types

    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    _upstream_qwen2_model_forward = Qwen2Model.forward

    def _patched_forward(self, *args, attention_mask=None, inputs_embeds=None, input_ids=None, **kwargs):
        # Resolve sequence length without forcing the embed lookup early — the
        # upstream forward does that itself.
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[-1]
        past_seen = 0
        pkv = kwargs.get("past_key_values")
        if pkv is not None:
            try:
                past_seen = pkv.get_seq_length()
            except Exception:
                past_seen = 0
        if attention_mask is not None and attention_mask.shape[-1] != seq_len + past_seen:
            attention_mask = None
        return _upstream_qwen2_model_forward(
            self,
            *args,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            **kwargs,
        )

    inner.forward = types.MethodType(_patched_forward, inner)

    # `load_pretrained_model` doesn't call .eval() on the LM, leaving dropout
    # active during inference and producing noisy logits. Force eval mode.
    model.eval()
    inner.eval()
    if hasattr(model, "diff_head") and model.diff_head is not None:
        model.diff_head.eval()

    # Use our local _LocalMultimodalDecoder (transformers-4.57+ safe). Upstream
    # `scale_rae.model.multimodal_decoder.MultimodalDecoder` pulls in an
    # older `ViTMAELayer` clone that requires `config._attn_implementation`,
    # which isn't set on the published decoder config and trips a `KeyError`.
    decoder = (
        _LocalMultimodalDecoder(
            pretrained_encoder_path=pretrained_encoder_path,
            general_decoder_config=decoder_config_path,
            num_patches=decoder_num_patches,
            drop_cls_token=drop_cls_token,
            decoder_path=decoder_ckpt,
        )
        .to(device)
        .to(dtype)
    )
    decoder.eval()

    return tokenizer, model, decoder


# ---------------------------------------------------------------------------
# xt-trajectory capture (monkey-patch of inference_flow.p_sample_loop)
# ---------------------------------------------------------------------------


def _validate_diff_head_is_full_sequence(model) -> None:
    from scale_rae.model.diffusion_loss.diffloss import FullSequenceRectifiedFlowProjector  # noqa

    head = getattr(model, "diff_head", None)
    if head is None:
        raise RuntimeError("model has no `diff_head`; --save_xt_steps requires the diffusion-head LM variant")
    if not isinstance(head, FullSequenceRectifiedFlowProjector):
        raise NotImplementedError(
            f"--save_xt_steps only supports FullSequenceRectifiedFlowProjector, "
            f"got {type(head).__name__}. (Per-token diffusion would yield one trajectory per token.)"
        )


def _install_xt_capture(model, target_steps: list[int]):
    """Monkey-patch `model.diff_head.inference_flow.p_sample_loop` to record
    the trajectory state at each step in `target_steps`. Returns a `state` dict
    whose `state["captured"]` is reset to `{}` at the start of every call —
    callers should reset it themselves between prompts.

    Captured value at step K is `(xt_BCHW, t_value)`:
      * `xt_BCHW` — the raw `carry[0]` at step K, in patch-grid layout (B,C,H,W),
        in the diffusion's native (possibly normalized) space, possibly with
        CFG batch doubling. Post-processing (CFG chunk, permute, denorm) is
        applied on read by `_postprocess_captured_xt`.
      * `t_value` — `sampler_timesteps[used_timesteps[i_at_step_K]]` for the
        step that JUST landed at the snapshot (i.e. the new noise level).
    """
    flow = model.diff_head.inference_flow
    original = flow.p_sample_loop
    state = {"captured": {}, "_original": original}
    model.diff_head._xt_capture_state = state
    target_set = set(int(k) for k in target_steps)

    def patched_p_sample_loop(
        model_,
        shape,
        x_end=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        # Mirrors transport.py:RectifiedFlow.p_sample_loop verbatim, plus a
        # step counter that snapshots `carry[0]` at requested counts.
        if x_end is None:
            x_end = flow.get_x_end(shape, device)
        x_t = x_end.to(device)
        if flow.step_type == "ucgm":
            carry = (x_t, None, None)
        else:
            carry = (x_t,)

        step_counter = 0

        # Regular loop: i in [N-1, N-2, ..., 1].
        loop_iter = reversed(range(1, len(flow.used_timesteps) - 1))
        for i in loop_iter:
            t_curr_v = flow.sampler_timesteps[flow.used_timesteps[i]]
            t_next_v = flow.sampler_timesteps[flow.used_timesteps[i - 1]]
            t_curr = torch.tensor(t_curr_v).to(device).repeat(x_t.size(0)).to(x_t.dtype)
            t_next = torch.tensor(t_next_v).to(device).repeat(x_t.size(0)).to(x_t.dtype)
            carry = flow.step_fn(model_, *carry, t_curr, t_next, denoised_fn, model_kwargs)
            step_counter += 1
            if step_counter in target_set:
                state["captured"][step_counter] = (carry[0].detach().clone(), float(t_next_v))

        # Final step.
        t_curr_v = flow.sampler_timesteps[flow.used_timesteps[-2]]
        t_next_v = flow.sampler_timesteps[flow.used_timesteps[-1]]
        t_curr = torch.tensor(t_curr_v).to(device).repeat(x_t.size(0)).to(x_t.dtype)
        t_next = torch.tensor(t_next_v).to(device).repeat(x_t.size(0)).to(x_t.dtype)
        if flow.step_type == "ucgm":
            carry = flow.step_fn(model_, *carry, t_curr, t_next, denoised_fn, model_kwargs)
        else:
            carry = flow.euler_forward(model_, *carry, t_curr, t_next, denoised_fn, model_kwargs)
        step_counter += 1
        if step_counter in target_set:
            state["captured"][step_counter] = (carry[0].detach().clone(), float(t_next_v))

        x_t = carry[0]
        x_pred = x_t.clamp(-1, 1) if clip_denoised else x_t
        return x_pred

    flow.p_sample_loop = patched_p_sample_loop
    return state


def _postprocess_captured_xt(model, xt_bchw: torch.Tensor) -> torch.Tensor:
    """Apply the same post-processing `FullSequenceRectifiedFlowProjector.infer`
    does after `p_sample_loop` returns: optional CFG chunk, BCHW→BLD permute,
    optional `data_std/mean` denormalization. Result is decoder-ready (B, L, D).
    """
    head = model.diff_head
    if head.use_cfg:
        xt_bchw, _ = xt_bchw.chunk(2, dim=0)
    xt = xt_bchw.permute(0, 2, 3, 1).contiguous().view(xt_bchw.shape[0], -1, xt_bchw.shape[1])
    if head.normalize_data:
        data_mean = head.data_mean.to(xt.device).unsqueeze(0).expand(xt.shape[0], *xt.shape[1:])
        data_std = head.data_std.to(xt.device).unsqueeze(0).expand(xt.shape[0], *xt.shape[1:])
        xt = xt * data_std + data_mean
    return xt


# ---------------------------------------------------------------------------
# Single-prompt generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_scale_rae_image(
    prompt: str,
    tokenizer,
    model,
    decoder,
    *,
    guidance_level: float = 1.0,
    max_new_tokens: int = 512,
    final_pixel_size: int = 256,
):
    """Run T2I generation for one prompt. Mirrors Scale-RAE/inference.py:117-185.

    Returns (latent_2d, image_01) where:
      * latent_2d: (1152, 16, 16) bf16 cpu — clean image embeddings reshaped
        to spatial-grid layout, matching ScaleRAEVAEInterface.encode output.
      * image_01:  (3, final_pixel_size, final_pixel_size) f32 cpu in [0, 1].
        The decoder's native 224×224 output is bicubic-upsampled to
        ``final_pixel_size`` (default 256, the 14→16 multiple bridge that
        unifies the I/O with the rest of the pixel-diffusion pipeline).
    """
    # Lazy imports — scale_rae must already be on sys.path.
    from scale_rae.constants import IMAGE_TOKEN_INDEX  # noqa: E402
    from scale_rae.conversation import conv_templates  # noqa: E402
    from scale_rae.mm_utils import tokenizer_image_token  # noqa: E402

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    output_ids, image_embeds = model.generate(
        input_ids,
        images=None,
        output_image=True,
        do_sample=True,
        temperature=0.0,
        use_customize_greedy=True,
        top_p=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        start_image_token_id=start_image_token_id,
        end_image_token_id=end_image_token_id,
        eos_token_id=eos_token_id,
        guidance_level=guidance_level,
    )

    if image_embeds is None or image_embeds.ndim < 2 or image_embeds.shape[0] == 0:
        raise RuntimeError(f"Scale-RAE returned no image embeddings for prompt: {prompt!r}")

    # image_embeds: (num_image_tokens=256, 1152). Add batch + zero CLS for decoder.
    image_embeds = image_embeds.unsqueeze(0)  # (1, 256, 1152)
    cls = torch.zeros(
        (image_embeds.shape[0], 1, image_embeds.shape[-1]),
        device=image_embeds.device,
        dtype=image_embeds.dtype,
    )
    image_features = torch.cat([cls, image_embeds], dim=1)  # (1, 257, 1152)
    decoder_dtype = next(decoder.parameters()).dtype
    pixels = decoder(image_features.to(decoder_dtype))  # (1, 3, 224, 224) in [0, 1]
    pixels = pixels.clamp(0.0, 1.0).float()
    if pixels.shape[-1] != final_pixel_size:
        # Bicubic upsample 224 → 256 (the "14→16 multiple bridge").
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(final_pixel_size, final_pixel_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
    pixels = pixels.cpu().squeeze(0)

    # Reshape clean latent to (1152, 16, 16) — raster-order grid.
    n_tokens, dim = image_embeds.shape[1], image_embeds.shape[2]
    grid = int(n_tokens**0.5)
    assert grid * grid == n_tokens, f"Scale-RAE returned {n_tokens} tokens; not a perfect square"
    latent_2d = image_embeds[0].reshape(grid, grid, dim).permute(2, 0, 1).contiguous().to(torch.bfloat16).cpu()

    return latent_2d, pixels


@torch.inference_mode()
def decode_xt_to_image(model, decoder, xt_bld: torch.Tensor, final_pixel_size: int = 256) -> torch.Tensor:
    """Decode (1, 256, 1152) post-processed xt into (3, final_pixel_size, final_pixel_size) [0, 1]."""
    cls = torch.zeros(
        (xt_bld.shape[0], 1, xt_bld.shape[-1]),
        device=xt_bld.device,
        dtype=xt_bld.dtype,
    )
    image_features = torch.cat([cls, xt_bld], dim=1)
    decoder_dtype = next(decoder.parameters()).dtype
    pixels = decoder(image_features.to(decoder_dtype))
    pixels = pixels.clamp(0.0, 1.0).float()
    if pixels.shape[-1] != final_pixel_size:
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(final_pixel_size, final_pixel_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
    return pixels.cpu().squeeze(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_scale_rae_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scale_rae_repo_path",
        type=str,
        default=DEFAULT_SCALE_RAE_REPO_PATH,
        help=(
            "Path to the Scale-RAE repo (added to sys.path so `import scale_rae` resolves). "
            f"Defaults to $SCALE_RAE_REPO_PATH or '../Scale-RAE' (current: {DEFAULT_SCALE_RAE_REPO_PATH!r})."
        ),
    )
    p.add_argument(
        "--scale_rae_model_path",
        type=str,
        default="nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B",
        help="HuggingFace repo id (or local dir) for the Scale-RAE LM + diffusion DiT.",
    )
    p.add_argument(
        "--scale_rae_pretrained_encoder",
        type=str,
        default="google/siglip2-so400m-patch14-224",
        help="HF id of the SigLIP-2 encoder (used by the decoder for image_std/mean).",
    )
    p.add_argument(
        "--scale_rae_decoder_config",
        type=str,
        default=None,
        help="Path to the decoder JSON config (e.g. <repo>/decoder/XL_decoder_config.json). "
        "If None, defaults to <repo>/decoder/XL_decoder_config.json.",
    )
    p.add_argument(
        "--scale_rae_decoder_ckpt",
        type=str,
        default=None,
        help="Path to the decoder weights. If None, defaults to "
        "<repo>/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt.",
    )
    p.add_argument(
        "--scale_rae_guidance_level",
        type=float,
        default=1.0,
        help="Classifier-free guidance level (1.0 = no guidance).",
    )
    p.add_argument(
        "--scale_rae_max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens for the LM autoregressive loop.",
    )
    p.add_argument(
        "--scale_rae_prompt_prefix",
        type=str,
        default="Could you generate an image of ",
        help="String prepended to every prompt before tokenization (Scale-RAE was "
        "trained on request-style prompts and ignores plain captions otherwise; see "
        "docs/Inference.md). The ORIGINAL caption is what we feed into the PiD pixel "
        "decoder; the prefix only affects what the LM sees. Default is "
        '"Could you generate an image of " — confirmed to push <im_start> into '
        "the top-2 next-token logits for arbitrary descriptive captions (just "
        '"Could you generate " is too short and the model drifts to "As an AI..."). '
        "Pass an empty string to disable.",
    )


def run_scale_rae_demo(args):
    """from_ldm demo flow for the Scale-RAE (SigLIP-2 + Qwen LM + DiT) text-conditional backbone.

    Bypasses the diffusers pipeline: generates a 256px image per prompt, optionally
    snapshots the diffusion trajectory at `--save_xt_steps` via a monkey-patched
    p_sample_loop, decodes each snapshot (baseline), and runs the shared PiD decode/save
    step. Invoked from from_ldm.py when --backbone is "siglip".
    """
    from pid._src.inference.cli_utils import maybe_init_distributed
    from pid._src.inference.decoder import load_our_decoder, run_ours_and_save_step
    from pid._src.inference.inference_utils import (
        AsyncUploader,
        build_tag,
        get_rank_and_world_size,
        load_prompts,
    )

    rank, world_size = get_rank_and_world_size()
    maybe_init_distributed(world_size, rank)
    is_rank0 = rank == 0

    if args.resolution != 256:
        raise ValueError(f"siglip backbone only supports --resolution 256, got {args.resolution}")
    save_xt_set = sorted(set(args.save_xt_steps)) if args.save_xt_steps else []

    prompts = load_prompts(args)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tag = build_tag(args, "siglip")
    if is_rank0:
        logger.info(
            f"Backbone: siglip  resolution: 256  guidance: {args.scale_rae_guidance_level}  "
            f"pid_steps: {args.pid_inference_steps}"
        )
        logger.info(f"Tag: {tag}")
        logger.info(f"#Prompts: {len(prompts)}  save_xt_steps: {save_xt_set}  scale: {args.scale}")

    experiment_opts = list(args.extra_experiment_opts) if args.extra_experiment_opts else []
    if is_rank0 and experiment_opts:
        logger.info(f"Extra experiment options: {experiment_opts}")

    decoder_config_path, decoder_ckpt_path = _resolve_decoder_paths(
        args.scale_rae_decoder_config,
        args.scale_rae_decoder_ckpt,
        args.scale_rae_repo_path,
    )

    # ---- Load Scale-RAE stack (warm-cache pattern across ranks) ----
    if world_size > 1:
        import torch.distributed as dist

        tokenizer = sr_model = decoder = None
        for r in range(world_size):
            if rank == r:
                msg = "from disk" if r == 0 else "from OS cache"
                logger.info(f"[Rank {rank}] Loading Scale-RAE stack ({msg}) ...")
                tokenizer, sr_model, decoder = load_scale_rae_stack(
                    repo_path=args.scale_rae_repo_path,
                    model_path=args.scale_rae_model_path,
                    decoder_config_path=decoder_config_path,
                    decoder_ckpt=decoder_ckpt_path,
                    pretrained_encoder_path=args.scale_rae_pretrained_encoder,
                    device="cuda",
                    dtype=dtype,
                )
            dist.barrier()
    else:
        logger.info("Loading Scale-RAE stack ...")
        tokenizer, sr_model, decoder = load_scale_rae_stack(
            repo_path=args.scale_rae_repo_path,
            model_path=args.scale_rae_model_path,
            decoder_config_path=decoder_config_path,
            decoder_ckpt=decoder_ckpt_path,
            pretrained_encoder_path=args.scale_rae_pretrained_encoder,
            device="cuda",
            dtype=dtype,
        )

    capture_state = None
    if save_xt_set:
        _validate_diff_head_is_full_sequence(sr_model)
        capture_state = _install_xt_capture(sr_model, save_xt_set)

    model = load_our_decoder(args, experiment_opts, is_rank0)

    output_dir = args.output_dir or "./results/official_demo/siglip"
    os.makedirs(output_dir, exist_ok=True)
    if is_rank0:
        logger.info(f"Outputs -> {output_dir}")

    uploader = AsyncUploader(max_workers=8) if args.upload else None

    prompt_prefix = args.scale_rae_prompt_prefix or ""
    if is_rank0 and prompt_prefix:
        logger.info(f"Prepending {prompt_prefix!r} to every prompt before LM tokenization")

    indexed_prompts = list(enumerate(prompts))
    if world_size > 1:
        indexed_prompts = indexed_prompts[rank::world_size]
        logger.info(f"[Rank {rank}/{world_size}] Processing {len(indexed_prompts)} prompts")

    for prompt_idx, prompt in indexed_prompts:
        seed = args.seed + prompt_idx
        sample_id = f"{prompt_idx:08d}"
        # Scale-RAE's autoregressive LM uses torch.manual_seed for sampling determinism.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if capture_state is not None:
            capture_state["captured"] = {}

        lm_prompt = prompt_prefix + prompt if prompt_prefix else prompt
        logger.info(f"[{prompt_idx}] Generating siglip (seed={seed}): {prompt[:80]!r}")

        latent_2d, image_01 = generate_scale_rae_image(
            prompt=lm_prompt,
            tokenizer=tokenizer,
            model=sr_model,
            decoder=decoder,
            guidance_level=args.scale_rae_guidance_level,
            max_new_tokens=args.scale_rae_max_new_tokens,
            final_pixel_size=256,
        )
        # latent_2d: (1152, 16, 16) bf16 cpu; image_01: (3, 256, 256) f32 cpu

        # Build (label, latent_[1,1152,16,16], baseline_01_[1,3,256,256], sigma) per step.
        steps: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []
        if capture_state is not None:
            for K in save_xt_set:
                if K not in capture_state["captured"]:
                    raise RuntimeError(
                        f"xt capture for step {K} did not fire — check that "
                        f"model.diff_head.inference_flow.p_sample_loop is the entry point"
                    )
                xt_bchw, t_K = capture_state["captured"][K]
                xt_bld = _postprocess_captured_xt(sr_model, xt_bchw)  # (1, 256, 1152)
                grid = int(xt_bld.shape[1] ** 0.5)
                xt_latent = (
                    xt_bld[0].reshape(grid, grid, xt_bld.shape[2]).permute(2, 0, 1).contiguous().unsqueeze(0)
                )  # (1, 1152, 16, 16)
                xt_baseline = decode_xt_to_image(sr_model, decoder, xt_bld, final_pixel_size=256).unsqueeze(0)
                steps.append((f"{K:02d}xt", xt_latent, xt_baseline, float(t_K)))

        steps.append(("x0", latent_2d.unsqueeze(0), image_01.unsqueeze(0), 0.0))

        for step_label, latent, baseline_01, sigma in steps:
            run_ours_and_save_step(
                model=model,
                args=args,
                tag=tag,
                sample_id=sample_id,
                prompt_idx=prompt_idx,
                step_label=step_label,
                latent=latent,
                baseline_01=baseline_01,
                sigma=sigma,
                caption=prompt,  # original caption — not the LM-prefixed one
                output_dir=output_dir,
                uploader=uploader,
                baseline_subdir="siglip_decode",
                baseline_upload_tag_prefix="siglip_decode",
            )

    if uploader is not None:
        if is_rank0:
            logger.info("Waiting for background uploads to complete ...")
        uploader.wait()

    if is_rank0:
        logger.info(f"Done! Results saved under {output_dir}")


__all__ = [
    "DEFAULT_SCALE_RAE_REPO_PATH",
    "_install_xt_capture",
    "_postprocess_captured_xt",
    "_resolve_decoder_paths",
    "_validate_diff_head_is_full_sequence",
    "add_scale_rae_args",
    "decode_xt_to_image",
    "generate_scale_rae_image",
    "load_scale_rae_stack",
    "run_scale_rae_demo",
]
