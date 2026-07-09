# PiD — Pixel Diffusion Decoder

> **TL;DR** — PiD is a plug-and-play diffusion decoder that replaces VAE/RAE decoders, turning latent representations directly into super-resolved pixels in a single pass.

<p align="center">
  <img src="figures/teaser.jpg" alt="PiD teaser" width="100%">
</p>

https://github.com/user-attachments/assets/a556e2d4-5de5-4bcf-9daa-80f7ea6b2124

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

## News
- 🚀 [July 9, 2026] PiD Training code released, with [PiD v1.5 and PixelDiT (2kto4k)](https://huggingface.co/nvidia/PiD/commit/3348c59bb545d9d0e29c2dec4c79b94592b83e8c) distilled and undistilled checkpoints!
- 🚀 [July 9, 2026] PiD **v1.5** checkpoints for **FLUX**, **FLUX.2**, and **Qwen-Image** are released. Check [release page](https://research.nvidia.com/labs/sil/projects/pid/comparison.html) to see improvements!
- 🔥 [June 2, 2026] PiD checkpoints for **SDXL**, **Qwen-Image** and **Qwen-Image-2512** are released. Check [HuggingFace](https://huggingface.co/nvidia/PiD).
- 🔥 [June 2, 2026] We clean up the codebase and remove useless code. Torch.compile mode is also available now.
- 🚀 [May 27, 2026] PiD is now in [ComfyUI](https://github.com/Comfy-Org/ComfyUI/pull/14103)!
- 🚀 [May 25, 2026] Paper, code, and model weights released, with PiD options for **FLUX**, **FLUX.2**, **Z-Image**, **Z-Image-Turbo**, **SD3**, **DINOv2**, and **SigLIP**.

## Table of Contents

- [Installation](#installation)
- [Download Checkpoints](#download-checkpoints)
- [Running inference with released checkpoints](#running-inference-with-released-checkpoints)
  - [LDM → PiD decode](#from-ldm)
  - [image → PiD decode](#from-clean)
- [Training](#training)
- [Repository layout](docs/repository_layout.md)

## Installation

> [!TIP]
> **Quick Start** — if your existing Python environment already has PyTorch (with CUDA), `transformers>=4.57.x`, and `diffusers>=0.37`, you can use it directly. Just install the small set of utility dependencies the inference code imports eagerly, and you're ready to run the diffusers backbones (`flux`/`flux2`/`flux2-klein-4b`/`flux2-klein-9b`/`sd3`/`zimage`/`zimage-turbo`):
>
> ```bash
> pip install hydra-core omegaconf pyyaml \
>     attrs einops loguru termcolor fvcore iopath wandb \
>     imageio opencv-python-headless pandas \
>     safetensors sentencepiece boto3 botocore
> ```
>
> Run commands from the repository root with `PYTHONPATH=.`.
> To validate the environment, run `PYTHONPATH=. python verify_env.py`. If you see `[PASS] Environment OK — all required imports and CUDA checks passed.`, the environment is ready to use.


If you want to create a new inference environment, `uv` is fast and easy to use.
It create a project-local `.venv` with the locked base dependencies:

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
# You can simply run `pip install uv`
uv python install 3.12
uv sync --frozen
source .venv/bin/activate
PYTHONPATH=. python verify_env.py
```

## Download Checkpoints

Checkpoints are hosted at [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) on the HuggingFace.
Pull the `checkpoints/` folder into this repo:

```bash
hf download nvidia/PiD --local-dir . --include "checkpoints/*"
```

## Running inference with released checkpoints

PiD ships two complementary entry points, each selecting a backbone with `--backbone`:

- `from_ldm.py`  — text/class → latent diffusion → PiD decode
- `from_clean.py` — image → VAE encode → PiD decode

> [!IMPORTANT]
> Picking the checkpoint variant — `--pid_ckpt_type`
> Every entry point accepts `--pid_ckpt_type {2k,2kto4k,2kto4k_v1pt5}` (default `2k`):
>
> - **`2k`** — the original 2048px-trained decoder, trained with 2K resolution only. Multiple aspect ratios are supported, typically 2048 × 2048 (1:1), 2304 × 1728 (4:3), 1728 × 2304 (3:4), 2688 × 1536 (16:9), and 1536 × 2688 (9:16).
> - **`2kto4k`** — the v1 up-to-4K-resolution decoder, trained with varying resolution (range from 2K to 4K). Multiple aspect ratios are supported. Less sharp than `2k` at 2048px resolution.
> - **`2kto4k_v1pt5`** — the v1.5 up-to-4K-resolution decoder for **FLUX**, **FLUX.2**, **Qwen-Image (WAN2.1)** VAE (range from 2K to 4K). Better color accuracy, no grid artifacts in the corners, trained with more anime data and small-face data. Better than `2kto4k` overall but less sharp than `2k` at 2048px resolution.
>
> For the exact checkpoint path for each backbone, see [docs/checkpoints.md](docs/checkpoints.md) and [checkpoint registry](pid/_src/inference/checkpoint_registry.py).


| `--backbone`   | Currently available `--pid_ckpt_type` |
|----------------|:-------------------------------------:|
| flux           | `2k`, `2kto4k_v1pt5` |
| flux2          | `2k`, `2kto4k_v1pt5` |
| flux2-klein-4b | `2k`, `2kto4k_v1pt5` |
| flux2-klein-9b | `2k`, `2kto4k_v1pt5` |
| zimage         | `2k`, `2kto4k_v1pt5` |
| zimage-turbo   | `2k`, `2kto4k_v1pt5` |
| sd3            | `2k`, `2kto4k` |
| qwenimage      | `2kto4k_v1pt5` |
| qwenimage-2512 | `2kto4k_v1pt5` |
| sdxl           | `2kto4k` |
| dinov2 (RAE)   | `2k` |
| siglip (Scale-RAE) | `2k` |

For the exact checkpoint path behind each `(backbone, --pid_ckpt_type)`, see [docs/checkpoints.md](docs/checkpoints.md).

<a id="from-ldm"></a>

### 📕 `from_ldm`: text / class → latent diffusion → PiD decode

Runs the chosen `--backbone` on a prompt, captures the intermediate `x_t` at user-specified denoising steps (early LDM
termination) and the final clean `x_0`, then decodes each captured latent with both the
native VAE / RAE decoder (baseline) and PiD.

#### Example 1 — Single-GPU, single prompt (Flux, default `2k` decoder)
Generating a 2048px image with Flux + PiD decode. Decoding latent from 24 and 28 (full) LDM steps.

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone flux \
    --prompt "A photorealistic half-body portrait of a brown tabby cat with bold stripes sitting attentively on a rustic wooden kitchen table, soft morning light streaming sideways through a large window, fine fur detail and stripe patterns sharply visible, intense amber-green eyes in razor-sharp focus, warm farmhouse kitchen softly out of focus, cinematic shallow depth of field, ultra-detailed fur texture, photorealistic" \
    --ldm_inference_steps 28 --save_xt_steps 24 \
    --output_dir ./results/official_demo/flux \
    --pid_inference_steps 4
```

#### Example 2 — Single-GPU, 4K decode with 4:3 aspect ratio (Flux, `2kto4k_v1pt5` decoder)

Same backbone as Example 1 but with `--resolution 4096,3072 --pid_ckpt_type 2kto4k_v1pt5`.
`--resolution` is the final output size, so the LDM runs at `1024,768` and
PiD decodes it to 4K.

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone flux \
    --prompt "A close photograph of a cat looking through frosted glass beside a small pine branch, winter light, soft condensation, simple cozy composition, expressive eyes." \
    --resolution 4096,3072 --pid_ckpt_type 2kto4k_v1pt5 \
    --ldm_inference_steps 28 --save_xt_steps 24 26 \
    --output_dir ./results/official_demo/flux_4k_ar4_3
```

#### Example 3 — Multi-GPU with a prompt file (Z-Image) with torch.compile

`torchrun` shards `--prompt_file` across ranks; each rank writes to
`--output_dir` independently. We use `--compile` to enable torch.compile for faster inference,
the first call will be slow due to the compilation. We use `default` compilation mode, to get further speedup, change to the `max-autotune` mode in `_maybe_compile_net (pid/_src/models/pixeldit_model.py)`. Note that extra cudatoolkit like nvcc is required for the compilation.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm --backbone zimage \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 50 --save_xt_steps 46 \
    --compile \
    --output_dir ./results/official_demo/zimage
```

#### Example 4 — Multi-GPU, 4K decode (Z-Image-Turbo, `2kto4k_v1pt5` decoder)

Z-Image-Turbo defaults to 9 diffusers steps with `guidance_scale=0.0`. The final
clean latent `x0` is always saved and is the recommended Turbo output to inspect.
`--save_xt_steps 7` is optional; it saves an additional near-final `x_t` sample
for comparison. `--resolution 4096` means `H=4096, W=4096` and the LDM runs at `1024,1024`.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm --backbone zimage-turbo \
    --prompt_file pid/_src/inference/prompts/prompt_zimage_turbo.txt \
    --resolution 4096 --pid_ckpt_type 2kto4k_v1pt5 \
    --output_dir ./results/official_demo/zimage_turbo_4k
```

#### `dinov2` / `siglip` backbones

The upstream RAE / Scale-RAE LDMs don't live in `diffusers` — see
[`docs/dinov2_siglip.md`](docs/dinov2_siglip.md) for setup and end-to-end
examples.

#### Suggested step settings per diffusers backbone

(See each script's docstring for the exact recipe.)

| Backbone | LDM steps flag          | Default steps | Optional `--save_xt_steps` | Recommended latent |
|----------|-------------------------|---------------|----------------------------|--------------------|
| flux     | `--ldm_inference_steps` | 28            | `22 24 26`                 | step `24`          |
| sd3      | `--ldm_inference_steps` | 28            | `22 24 26`                 | step `24`          |
| sdxl     | `--ldm_inference_steps` | 30            | `24 26 28`                 | step `26`          |
| flux2    | `--ldm_inference_steps` | 50            | `44 46 48`                 | step `46`          |
| flux2-klein-4b | `--ldm_inference_steps` | 4      | `3`                      | `x0`               |
| flux2-klein-9b | `--ldm_inference_steps` | 4      | `3`                      | `x0`               |
| qwenimage | `--ldm_inference_steps` | 50 | `44 46 48`             | step `44`          |
| qwenimage-2512 | `--ldm_inference_steps` | 50 | `44 46 48`             | step `44`          |
| zimage   | `--ldm_inference_steps` | 50            | `44 46 48`                 | step `46`          |
| zimage-turbo | `--ldm_inference_steps` | 9         | `7`                        | `x0`               |

---
<a id="from-clean"></a>

### 📗 `from_clean`: image → VAE encode → PiD decode

No latent diffusion model is run. The input image is fed at its native resolution
(only center-cropped so each side is a multiple of 16), encoded by VAE, optionally
corrupted with Gaussian noise at each sigma in `--degrade_sigmas`, then decoded by PiD
at `--scale * vae_native_resolution`.

Single-GPU example (Flux):

```bash
PYTHONPATH=. python -m pid._src.inference.from_clean --backbone flux \
    --manifest assets/clean_image_manifest.jsonl \
    --degrade_sigmas 0.0 \
    --output_dir ./results/official_demo_from_clean/flux \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

You can pass a single image with `--input_path` and a prompt with `--prompt`
instead of `--manifest`, and a sigma sweep such as `--degrade_sigmas 0.0 0.2 0.4 0.8`
to decode noise-corrupted latents. Swap `--backbone` to use a different VAE
(`flux2` / `sd3` / `sdxl` / `qwenimage`); `sdxl` automatically uses its
variance-preserving noising form.

The `dinov2` / `siglip` `from_clean` flows take the same flags but with a different
`--scale` (8 for `siglip`); their encoders resize internally to their fixed native
interface (512 / 256) regardless of the input image size — see
[`docs/dinov2_siglip.md`](docs/dinov2_siglip.md).

## Training

### Create the training environment
We still use conda since it is easy to install CUDA toolkit.

```bash
conda env create -f environment.yml
conda activate pid
python -m pip install -e . --group full \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

Check [Docker](docker/README.md) to create a training docker image.

### Training Tutorial
See the [training guide](docs/training.md) for dataset preparation and commands to finetune PixelDiT, train a PiD teacher, and distill a PiD student.

To evaluate training experiment checkpoints, see [inference with training checkpoints](docs/inference.md) for PixelDiT, PiD teacher, and PiD student examples.


## License

PiD codebase is licensed under the [Apache License 2.0](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, code style,
and the DCO sign-off requirement.

## Acknowledgments

The authors would like to acknowledge [Yongsheng Yu](https://www.yongshengyu.com/) and [Wei Xiong](https://wxiong.me/) for open-sourcing [PixelDiT](https://pixeldit.github.io/)'s model and weights, and thank Product Managers [Aditya Mahajan](https://www.linkedin.com/in/aditya-mahajan1) and [Matt Cragun](https://www.linkedin.com/in/mcragun/) for their valuable support and guidance.


## Citation

```bibtex
@article{lu2026pid,
    title={PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion},
    author={Lu, Yifan and Wu, Qi and Wu, Jay Zhangjie and Wang, Zian and Ling, Huan and Fidler, Sanja and Ren, Xuanchi},
    journal={arXiv preprint arXiv:2605.23902},
    year={2026}
}
```
