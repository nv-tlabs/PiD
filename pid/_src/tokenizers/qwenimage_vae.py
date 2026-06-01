# Qwen-Image VAE tokenizer — self-contained 2D path, loads local weights (no HF download).
#
# This is the 2D-stripped variant of AutoencoderKLQwenImage (a WanVAE2d_ over the 16-ch
# Qwen-Image latent) — the exact tokenizer the PiD-Qwen-Image student was trained with.
# For T=1 image inputs it is numerically equivalent to the 3D AutoencoderKLQwenImage but
# avoids any HuggingFace download: weights are loaded from a local pre-converted state
# dict (default ./checkpoints/QwenImage_VAE_2d.pth). Place it via:
#   cp <linear-vsr>/checkpoints/QwenImage_VAE_2d.pth checkpoints/QwenImage_VAE_2d.pth
#
# Per-channel normalization (latents_mean / latents_std) is byte-for-byte identical to
# AutoencoderKLQwenImage.config (and to Wan2.1's). 16 latent channels, 8x spatial
# compression. Architecture ported from the internal linear-vsr tokenizers
# (wan2pt1_img_only.py:WanVAE2d_ + qwenimage.py:QwenImageVAE2d).

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._ext.imaginaire.lazy_config import LazyDict
from pid._ext.imaginaire.utils import log
from pid._ext.imaginaire.utils.distributed import get_rank, sync_model_states
from pid._src.tokenizers.interface import VideoTokenizerInterface

__all__ = [
    "WanVAE2d_",
    "QwenImageVAE2d",
    "QwenImageVAEInterface",
    "QwenImageVAEConfig",
]

_DEFAULT_LOCAL_CACHE = "./checkpoints/QwenImage_VAE_2d.pth"

# Per-channel latent normalization — AutoencoderKLQwenImage config defaults.
_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


# ===========================================================================
# Layer 1 — WanVAE2d_ architecture (2D-only)
# ===========================================================================


class RMS_norm(nn.Module):
    """2D RMS_norm. gamma shape is (dim,1,1) for channel-first 4D tensors."""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """Fix bfloat16 support for nearest-neighbor interpolation."""
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """2D-only resampling: none / upsample2d / downsample2d."""

    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "downsample2d")
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Spatial self-attention, single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        x = self.proj(x)
        return x + identity


class Encoder2d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = nn.Conv2d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode="downsample2d"))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Decoder2d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, mode="upsample2d"))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Conv2d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class WanVAE2d_(nn.Module):
    """2D-only Wan VAE. No streaming, no caching, no batch-chunking."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        temporal_window=4,  # ignored, kept for config compat
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales

        self.encoder = Encoder2d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout)
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder2d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample[::-1], dropout)

    def encode(self, x, scale):
        """x: (B,3,H,W) or 5D (B,3,T,H,W). Returns same ndim, normalized latent."""
        is_5d = x.ndim == 5
        if is_5d:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
        out = self.encoder(x)
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        if is_5d:
            mu = rearrange(mu, "(b t) c h w -> b c t h w", b=B, t=T)
        return mu

    def decode(self, z, scale):
        """z: (B,z,h,w) or 5D. Returns same ndim, image in [-1,1]."""
        is_5d = z.ndim == 5
        if is_5d:
            B, C, T, H, W = z.shape
            z = rearrange(z, "b c t h w -> (b t) c h w")
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1) + scale[0].view(1, self.z_dim, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        out = self.decoder(x)
        if is_5d:
            out = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
        return out


# ===========================================================================
# Layer 2 — Factory (local-only load) + wrapper
# ===========================================================================


def _build_qwenimage_2d_vae(z_dim: int, vae_pth: str, device: str) -> WanVAE2d_:
    """Build a meta-device WanVAE2d_, then materialize from a local pre-converted
    state dict on rank 0 and broadcast. No HuggingFace download."""
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    with torch.device("meta"):
        model = WanVAE2d_(**cfg)

    if get_rank() == 0:
        if not os.path.exists(vae_pth):
            raise FileNotFoundError(
                f"Qwen-Image 2D VAE weights not found at {vae_pth!r}. Copy them from the "
                f"internal checkpoints, e.g.:\n"
                f"  cp <linear-vsr>/checkpoints/QwenImage_VAE_2d.pth {vae_pth}"
            )
        log.info(f"Loading Qwen-Image 2D VAE from {vae_pth}")
        ckpt = torch.load(vae_pth, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, assign=True, strict=True)
    else:
        model.to_empty(device=device)

    # sync_model_states needs contiguous tensors with matching strides across ranks.
    for p in model.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()
    for b in model.buffers():
        if not b.is_contiguous():
            b.data = b.data.contiguous()

    sync_model_states(model)
    return model


class QwenImageVAE2d:
    """2D image-only VAE with Qwen-Image weights.

    encode((B,3,H,W) in [-1,1])  ->  normalized latent (B,16,H/8,W/8)
    decode((B,16,h,w))            ->  recon (B,3,H,W) in [-1,1]
    """

    def __init__(
        self,
        z_dim: int = 16,
        vae_pth: str = _DEFAULT_LOCAL_CACHE,
        dtype=torch.float,
        device: str = "cuda",
        is_amp: bool = True,
    ):
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim

        mean = torch.tensor(_LATENTS_MEAN, dtype=dtype, device=device)
        std = torch.tensor(_LATENTS_STD, dtype=dtype, device=device)
        self.scale = [mean, 1.0 / std]

        self.model = _build_qwenimage_2d_vae(z_dim, vae_pth, device)
        self.model = self.model.eval().requires_grad_(False)

        self.is_amp = is_amp
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        in_dtype = images.dtype
        with self.context:
            if not self.is_amp:
                images = images.to(self.dtype)
            latent = self.model.encode(images, self.scale)
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        in_dtype = zs.dtype
        with self.context:
            if not self.is_amp:
                zs = zs.to(self.dtype)
            recon = self.model.decode(zs, self.scale)
        return recon.to(in_dtype)


# ===========================================================================
# Layer 3 — VideoTokenizerInterface
# ===========================================================================


class QwenImageVAEInterface(VideoTokenizerInterface):
    """Pipeline-compatible interface for Qwen-Image VAE (2D path). Image-only."""

    def __init__(self, chunk_duration: int = 1, **kwargs):
        self.model = QwenImageVAE2d(
            z_dim=kwargs.get("z_dim", 16),
            vae_pth=kwargs.get("vae_pth", _DEFAULT_LOCAL_CACHE),
            dtype=torch.bfloat16,
            is_amp=False,
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
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "qwenimage_vae_tokenizer"


QwenImageVAEConfig: LazyDict = L(QwenImageVAEInterface)(
    name="qwenimage_vae_tokenizer",
)
