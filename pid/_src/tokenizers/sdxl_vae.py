# SDXL VAE tokenizer — self-contained, loads local weights (no HF download).
#
# Architecture: identical to the Flux/SD3 conv AutoEncoder but with z_channels=4 and the
# `quant_conv` / `post_quant_conv` 1x1 convolutions that SDXL's AutoencoderKL keeps (Flux
# and SD3 dropped them). Reusing the quant-conv-free Flux AutoEncoder would silently drop
# those layers and produce a wrong latent (honeycomb decode artifacts) — so SDXL needs the
# dedicated SDAutoEncoder below.
#
# Normalization is affine scale/shift:
#   encode: z = scale_factor * (z_raw - shift_factor)     (SDXL: scale=0.13025, shift=0)
#   decode: z_raw = z / scale_factor + shift_factor
#
# Weights are loaded from a local file (default ./checkpoints/sdxl_vae.safetensors), the
# same diffusers-format SDXL VAE the PiD-SDXL student was trained with. Place it via:
#   cp <linear-vsr>/checkpoints/sdxl_vae.safetensors checkpoints/sdxl_vae.safetensors
# (or download the SDXL VAE once and convert; the loader auto-handles diffusers→LDM keys).
#
# Mirrors the internal linear-vsr tokenizer (pid/_src/tokenizers/sd_vae.py).

from contextlib import nullcontext

import torch
import torch.nn as nn

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.distributed import get_rank, sync_model_states
from pid._src.models.utils import load_state_dict
from pid._src.tokenizers.flux_vae import (
    AutoEncoderParams,
    Decoder,
    DiagonalGaussian,
    Encoder,
)
from pid._src.tokenizers.interface import VideoTokenizerInterface

__all__ = [
    "SDAutoEncoder",
    "SDXLVAE",
    "SDXLVAEInterface",
    "SDXLVAEConfig",
]

SDXL_VAE_PARAMS = AutoEncoderParams(
    z_channels=4,
    scale_factor=0.13025,
    shift_factor=0.0,
)


def _convert_diffusers_to_ldm(state_dict: dict) -> dict:
    """Auto-detect diffusers-format keys and convert to the LDM naming the Encoder/Decoder
    classes expect. If the state_dict is already in LDM format (no `down_blocks`), it is
    returned unchanged. `quant_conv` / `post_quant_conv` keys are identical in both formats.
    """
    if not any("down_blocks" in k for k in state_dict):
        return state_dict

    num_levels = 4  # ch_mult=[1,2,4,4] -> 4 levels

    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        if "encoder.down_blocks" in key:
            new_key = new_key.replace("down_blocks", "down")
            new_key = new_key.replace("resnets", "block")
            new_key = new_key.replace("downsamplers.0", "downsample")
        if "decoder.up_blocks" in key:
            for i in range(num_levels):
                if f"up_blocks.{i}" in new_key:
                    new_key = new_key.replace(f"up_blocks.{i}", f"up.{num_levels - 1 - i}")
                    break
            new_key = new_key.replace("resnets", "block")
            new_key = new_key.replace("upsamplers.0", "upsample")
        new_key = new_key.replace("mid_block.resnets.0", "mid.block_1")
        new_key = new_key.replace("mid_block.resnets.1", "mid.block_2")
        new_key = new_key.replace("mid_block.attentions.0.group_norm", "mid.attn_1.norm")
        new_key = new_key.replace("mid_block.attentions.0.to_q", "mid.attn_1.q")
        new_key = new_key.replace("mid_block.attentions.0.to_k", "mid.attn_1.k")
        new_key = new_key.replace("mid_block.attentions.0.to_v", "mid.attn_1.v")
        new_key = new_key.replace("mid_block.attentions.0.to_out.0", "mid.attn_1.proj_out")
        new_key = new_key.replace("conv_shortcut", "nin_shortcut")
        new_key = new_key.replace("conv_norm_out", "norm_out")
        # Reshape 2D attention weights (diffusers nn.Linear) to 4D (our nn.Conv2d k=1).
        if "attn_1" in new_key and value.ndim == 2:
            value = value.unsqueeze(-1).unsqueeze(-1).contiguous()
        new_sd[new_key] = value

    return new_sd


class SDAutoEncoder(nn.Module):
    """SDXL AutoEncoder. Same conv stack as Flux/SD3 but with quant_conv/post_quant_conv."""

    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian(sample=False)
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

        # SDXL has quant_conv and post_quant_conv (Flux/SD3 do not)
        self.quant_conv = nn.Conv2d(2 * params.z_channels, 2 * params.z_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(params.z_channels, params.z_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            assert x.shape[2] == 1, f"Image-only VAE requires T=1, got T={x.shape[2]}"
            x = x.squeeze(2)
            video_format_input = True
        else:
            video_format_input = False

        h = self.encoder(x)
        h = self.quant_conv(h)
        z = self.reg(h)
        z = self.scale_factor * (z - self.shift_factor)

        if video_format_input:
            z = z.unsqueeze(2)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 5:
            assert z.shape[2] == 1, f"Image-only VAE requires T=1, got T={z.shape[2]}"
            z = z.squeeze(2)
            video_format_input = True
        else:
            video_format_input = False

        z = z / self.scale_factor + self.shift_factor
        z = self.post_quant_conv(z)
        x = self.decoder(z)

        if video_format_input:
            x = x.unsqueeze(2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _sdxl_vae(
    pretrained_path: str = None,
    device: str = "cpu",
    s3_credential_path: str = "credentials/s3_training.secret",
) -> SDAutoEncoder:
    """Build SDAutoEncoder, loading local weights on rank 0 then broadcasting.

    Meta-device init + rank-0 load (assign=True) + sync_model_states, mirroring the
    Flux/SD3 VAE factories. Auto-converts diffusers-format checkpoints to LDM keys.
    """
    with torch.device("meta"):
        model = SDAutoEncoder(SDXL_VAE_PARAMS)

    if pretrained_path is None:
        model.to_empty(device=device)
    else:
        if get_rank() == 0:
            ckpt = load_state_dict(
                pretrained_path,
                s3_credential_path=s3_credential_path if pretrained_path.startswith("s3://") else None,
            )
            ckpt = _convert_diffusers_to_ldm(ckpt)
            model_keys = set(model.state_dict().keys())
            ckpt = {k: v for k, v in ckpt.items() if k in model_keys}
            log.info(f"Loading SDXL VAE from {pretrained_path}")
            model.load_state_dict(ckpt, assign=True)
            model.to(device)
        else:
            model.to_empty(device=device)
    sync_model_states(model)

    return model


class SDXLVAE:
    """Wrapper with dtype/AMP handling. All tensors are 4D (B, C, H, W)."""

    def __init__(
        self,
        vae_pth: str = "./checkpoints/sdxl_vae.safetensors",
        s3_credential_path: str = "credentials/s3_training.secret",
        dtype: torch.dtype = torch.float,
        device: str = "cuda",
        is_amp: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.scale = None

        self.model = _sdxl_vae(
            pretrained_path=vae_pth,
            device=device,
            s3_credential_path=s3_credential_path,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [-1, 1]. Returns (B, 4, H/8, W/8)."""
        in_dtype = images.dtype
        with self.context:
            if not self.is_amp:
                images = images.to(self.dtype)
            latent = self.model.encode(images)
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        """zs: (B, 4, h, w). Returns (B, 3, H, W)."""
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            recon = self.model.decode(zs)
        return recon.to(in_dtype)


class SDXLVAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible interface for SDXL VAE. Image-only (temporal_compression_factor=1)."""

    def __init__(self, chunk_duration: int = 1, **kwargs):
        self.model = SDXLVAE(
            dtype=torch.bfloat16,
            is_amp=False,
            vae_pth=kwargs.get("vae_pth", "./checkpoints/sdxl_vae.safetensors"),
            s3_credential_path=kwargs.get("s3_credential_path", "credentials/s3_training.secret"),
        )
        self.chunk_duration = chunk_duration

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if state.ndim == 5:
            assert state.shape[2] == 1, f"Image-only VAE requires T=1, got T={state.shape[2]}"
            x = state.squeeze(2)
        else:
            x = state
        latent = self.model.encode(x)
        return latent.unsqueeze(2)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Accept 5D (B,C,T,H,W) for pipeline compat. T must be 1. Returns 5D."""
        if latent.ndim == 5:
            assert latent.shape[2] == 1, f"Image-only VAE requires T=1, got T={latent.shape[2]}"
            z = latent.squeeze(2)
        else:
            z = latent
        recon = self.model.decode(z)
        return recon.unsqueeze(2)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return num_pixel_frames

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 1

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_ch(self):
        return 4

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "sdxl_vae_tokenizer"


SDXLVAEConfig: LazyDict = L(SDXLVAEInterface)(
    name="sdxl_vae_tokenizer",
)
