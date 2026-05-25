# PiD — Pixel Diffusion Decoder

<p align="center">
  <img src="figures/teaser.jpg" alt="PiD teaser" width="100%">
</p>

https://github.com/user-attachments/assets/e8a3e882-7a49-4867-b3f3-c3dee3c54c4f


PiD reformulates the latent-to-pixel decoder as a conditional pixel-space diffusion
model, unifying decoding and upsampling into a single generative module.
It directly denoises in high-resolution pixel
space and produces a super-resolved image in one pass.

**[Paper](https://arxiv.org/abs/2605.23902), [Project Page](https://research.nvidia.com/labs/sil/projects/pid/), [Model Weights](https://huggingface.co/nvidia/PiD)**

[Yifan Lu](https://yifanlu0227.github.io/),
[Qi Wu](https://wilsoncernwq.github.io/),
[Jay Zhangjie Wu](https://zhangjiewu.github.io/),
[Zian Wang](https://www.cs.toronto.edu/~zianwang/),
[Huan Ling](https://www.cs.toronto.edu/~linghuan/),
[Sanja Fidler](https://www.cs.utoronto.ca/~fidler/),
[Xuanchi Ren](https://xuanchiren.com/) <br>

## Installation

> [!TIP]
> **Quick Start** — if your environment already has PyTorch (with CUDA), `transformers==4.57.x`, and `diffusers>=0.37`, you don't need to build a new conda env. Just install the small set of utility deps the inference code pulls eagerly and you're ready to run the diffusers backbones (`flux`/`flux2`/`sd3`/`zimage`):
>
> ```bash
> pip install hydra-core==1.3.2 omegaconf==2.3.0 \
>     attrs einops loguru termcolor fvcore iopath pynvml wandb \
>     imageio opencv-python-headless pandas \
>     safetensors "huggingface-hub>=1.0" sentencepiece boto3 botocore
> pip install -e .
> ```
>
> For the `dinov2` / `siglip` backbones you additionally need the upstream RAE / Scale-RAE repos plus a couple of extra packages — see [docs/dinov2_siglip.md](docs/dinov2_siglip.md).

Full conda-managed install (preferred if you're starting from scratch):

```bash
conda env create -f environment.yml
conda activate pid

# 2. Install this package in editable mode.
pip install -e .
```

## Checkpoints and assets

Pretrained PiD checkpoints live under `checkpoints/`. Each diffusers backbone ships
two variants — the original `2k` decoder (trained at 2048px) and a `2kto4k` decoder
(trained with multi-resolution data bucketing 2048→3840 + an SD3-style dynamic
shift, intended for 1024 LDM → 4K decoding). Pick the variant at the CLI via
`--pid_ckpt_type {2k,2kto4k}` (default: `2k`).

### Downloading

The released decoder weights and the encoder/decoder ("VAE") weights they
depend on are hosted at [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) on
the Hugging Face Hub. Pull just the `checkpoints/` tree into this repo:

```bash
hf download nvidia/PiD --local-dir . --include "checkpoints/*"
```

## Running inference

PiD ships two complementary entry points per backbone:

| Backbone | `from_clean_*` (image → encode → PiD) | `from_ldm_*` (text/class → LDM → PiD) |
|----------|---------------------------------------|---------------------------------------|
| flux     | `from_clean_flux.py`    | `from_ldm_flux.py`    |
| flux2    | `from_clean_flux2.py`   | `from_ldm_flux2.py`   |
| sd3      | `from_clean_sd3.py`     | `from_ldm_sd3.py`     |
| zimage   | reuses `flux`           | `from_ldm_zimage.py`  |
| dinov2   | `from_clean_dinov2.py`  | `from_ldm_dinov2.py`  |
| siglip   | `from_clean_siglip.py`  | `from_ldm_siglip.py`  |

All scripts live under `pid/_src/inference/` and decode each captured latent
twice — once with the backbone's native VAE (baseline) and once with PiD.

> [!IMPORTANT]
> Picking the checkpoint variant — `--pid_ckpt_type`
> Every entry point accepts `--pid_ckpt_type {2k,2kto4k}` (default `2k`):
>
> - **`2k`** — the original 2048px-trained decoder.
> - **`2kto4k`** — the up-to-4K-resolution decoder. > > Available for `flux` / `flux2` / `sd3` / `zimage` only. Worse than `2k` at 2048px resolution.
>
> For the exact checkpoint path for each backbone, see [docs/checkpoints.md](docs/checkpoints.md).
> A quick sanity check that the right variant loaded: when `2kto4k` is active you
should see `PixelDiT dynamic shift: base_shift=4.0 base_image_size=1024` in the
init log; for `2k` that line is absent. Both `2k` and `2kto4k` support non-square aspect ratios.

### 📕 `from_ldm_*`: text / class → latent diffusion → PiD decode

Runs the corresponding latent-diffusion backbone on a prompt (or class id for
the class-conditional `dinov2` backbone), captures the intermediate `x_t` at
user-specified denoising steps and the final clean `x_0`, then decodes
each captured latent with both the native VAE / RAE decoder (baseline) and PiD.

For `flux` / `flux2` / `sd3` / `zimage` the LDM is a HuggingFace `diffusers`
pipeline (`FluxPipeline`, `Flux2Pipeline`, `StableDiffusion3Pipeline`,
`ZImagePipeline`).

For `dinov2` and `siglip` the LDM is the upstream
[RAE](https://github.com/bytetriper/RAE) (class-conditional ImageNet-512) or
[Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE) (text-conditional
256px) repo — see the optional-deps section below for installation.

#### Example 1 — Single-GPU, single prompt (Flux, default `2k` decoder)

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux \
    --prompt "A photorealistic half-body portrait of a brown tabby cat with bold stripes sitting attentively on a rustic wooden kitchen table, soft morning light streaming sideways through a large window, fine fur detail and stripe patterns sharply visible, intense amber-green eyes in razor-sharp focus, warm farmhouse kitchen softly out of focus, cinematic shallow depth of field, ultra-detailed fur texture, photorealistic" \
    --ldm_inference_steps 28 --save_xt_steps 22 24 26 \
    --output_dir ./results/official_demo/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

#### Example 2 — Single-GPU, 4K decode (Flux, `2kto4k` decoder)

Same backbone as Example 1 but with `--resolution 1024 --pid_ckpt_type 2kto4k`,
so the LDM produces a 1024² latent and PiD decodes it to 4K.

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm_flux \
    --prompt "A photorealistic half-body portrait of a brown tabby cat with bold stripes sitting attentively on a rustic wooden kitchen table, soft morning light streaming sideways through a large window, fine fur detail and stripe patterns sharply visible, intense amber-green eyes in razor-sharp focus, warm farmhouse kitchen softly out of focus, cinematic shallow depth of field, ultra-detailed fur texture, photorealistic" \
    --resolution 1024 --pid_ckpt_type 2kto4k \
    --ldm_inference_steps 28 --save_xt_steps 22 24 26 \
    --output_dir ./results/official_demo/flux_4k \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

#### Example 3 — Multi-GPU with a prompt file (Z-Image)

`torchrun` shards `--prompt_file` across ranks; each rank writes to
`--output_dir` independently.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm_zimage \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 50 --save_xt_steps 44 46 48 \
    --output_dir ./results/official_demo/zimage \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

#### `dinov2` / `siglip` backbones

The upstream RAE / Scale-RAE LDMs don't live in `diffusers` — see
[`docs/dinov2_siglip.md`](docs/dinov2_siglip.md) for setup and end-to-end
examples.

#### Suggested step settings per diffusers backbone

(See each script's docstring for the exact recipe.)

| Backbone | LDM steps flag          | Default steps | `--save_xt_steps` (example) | Best LDM termination steps |
|----------|-------------------------|---------------|-----------------------------|----------------------|
| flux     | `--ldm_inference_steps` | 28            | `18 20 22 24 26`         | 24  |
| sd3      | `--ldm_inference_steps` | 28            | `18 20 22 24 26`         | 24  |
| flux2    | `--ldm_inference_steps` | 50            | `40 42 44 46 48`         | 46  |
| zimage   | `--ldm_inference_steps` | 50            | `40 42 44 46 48`         | 46  |

---
### 📗 `from_clean_*`: image → VAE encode → PiD decode

No latent diffusion model is run. The input image is encode by VAE,
optionally corrupted with Gaussian noise at each
sigma in `--degrade_sigmas`, then decoded by PiD at `--scale * input_resolution`.

Single-GPU example (Flux):

```bash
PYTHONPATH=. python -m pid._src.inference.from_clean_flux \
    --manifest assets/clean_image_manifest.jsonl \
    --input_resolution 512 \
    --degrade_sigmas 0.0 \
    --output_dir ./results/official_demo_from_clean/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

You can pass a single image with `--input_path` and a prompt with `--prompt`
instead of `--manifest`, and a sigma sweep such as `--degrade_sigmas 0.0 0.2 0.4 0.8`
to decode noise-corrupted latents.

The `dinov2` / `siglip` `from_clean_*` flows take the same flags but with
different default resolutions and scales —
see [`docs/dinov2_siglip.md`](docs/dinov2_siglip.md).

### Common arguments

| Flag | Meaning |
|------|---------|
| `--pid_inference_steps`| Number of denoising steps for PiD (4 for the released distilled checkpoints) |
| `--scale`              | PiD upscale factor (output = `baseline * scale`); 8 for Scale-RAE and 4 for other backbones |
| `--cfg_scale`          | Classifier-free guidance scale for PiD |
| `--output_dir`         | Where to write the side-by-side comparison images |
| `--seed`               | Base random seed |

Multi-GPU runs use `torchrun --nproc_per_node=N`; each rank processes a shard
of the prompts / manifest entries and writes to `--output_dir` independently.

## Repository layout

```
pid/_src/inference/
├── from_ldm_{flux,flux2,sd3,zimage,dinov2,siglip}.py  # text/class → LDM → PiD decode
├── from_clean_{flux,flux2,sd3,dinov2,siglip}.py       # image → encode → PiD decode
├── _demo_common.py                                    # shared CLI + run loop for from_ldm_*
├── _demo_from_clean_common.py                         # shared CLI + run loop for from_clean_*
├── checkpoint_registry.py                             # backbone → PiD checkpoint mapping
├── pipeline_registry.py                               # diffusers backbone → HF pipeline mapping
├── rae_generation.py                                  # DINOv2-RAE LDM helpers (from_ldm_dinov2)
├── scale_rae_generation.py                            # Scale-RAE LDM helpers (from_ldm_siglip)
└── prompts/                                           # prompt files for from_ldm_*
```

## License

PiD codebase is licensed under the [Apache License 2.0](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, code style,
and the DCO sign-off requirement.


## Citation

```bibtex
@article{lu2026pid,
    title={PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion},
    author={Lu, Yifan and Wu, Qi and Wu, Jay Zhangjie and Wang, Zian and Ling, Huan and Fidler, Sanja and Ren, Xuanchi},
    journal={arXiv preprint arXiv:2605.23902},
    year={2026}
}
```
