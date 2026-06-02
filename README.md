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
- 🔥 [June 2, 2026] PiD checkpoints for **SDXL**, **Qwen-Image** and **Qwen-Image-2512** are released. Check [HuggingFace](https://huggingface.co/nvidia/PiD).
- 🔥 [June 2, 2026] A new checkpoint for **FLUX.2 (2kto4k)** (with `_2606` suffix) that has no color drifting issue. See [here](docs/FLUX2_2kto4k_new_ckpt_compare.md) for comparison with the old one.
- 🔥 [June 2, 2026] We clean up the codebase and remove useless code. Torch.compile mode is also available now.
- 🚀 [May 27, 2026] PiD is now in [ComfyUI](https://github.com/Comfy-Org/ComfyUI/pull/14103)!
- 🚀 [May 25, 2026] Paper, code, and model weights released, with PiD options for **FLUX**, **FLUX.2**, **Z-Image**, **Z-Image-Turbo**, **SD3**, **DINOv2**, and **SigLIP**.
- 🔜 [Coming Soon] PiD undistilled checkpoints.
- ⏳ [Planned] Training scripts.

## Installation

> [!TIP]
> **Quick Start** — if your environment already has PyTorch (with CUDA), `transformers>=4.57.x`, and `diffusers>=0.37`, you don't need to build a new conda env. Just install the small set of utility deps the inference code pulls eagerly and you're ready to run the diffusers backbones (`flux`/`flux2`/`flux2-klein-4b`/`flux2-klein-9b`/`sd3`/`zimage`/`zimage-turbo`):
>
> ```bash
> pip install hydra-core omegaconf pyyaml \
>     attrs einops loguru termcolor fvcore iopath wandb \
>     imageio opencv-python-headless pandas \
>     safetensors sentencepiece boto3 botocore
> pip install -e .
> ```
> To validate your environment is ready for inference, run `python verify_env.py`.


Full conda-managed install (preferred if you're starting from scratch):

```bash
conda env create -f environment.yml
conda activate pid

# 2. Install this package in editable mode.
pip install -e .
```

### Download Checkpoints

Checkpoints are hosted at [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) on the HuggingFace.
Pull the `checkpoints/` folder into this repo:

```bash
hf download nvidia/PiD --local-dir . --include "checkpoints/*"
```

## Running inference

PiD ships two complementary entry points, each selecting a backbone with `--backbone`:

- `from_ldm.py`  — text/class → latent diffusion → PiD decode
- `from_clean.py` — image → VAE encode → PiD decode

> [!IMPORTANT]
> Picking the checkpoint variant — `--pid_ckpt_type`
> Every entry point accepts `--pid_ckpt_type {2k,2kto4k}` (default `2k`):
>
> - **`2k`** — the original 2048px-trained decoder, trained with 2K resolution only. Multiple aspect ratios are supported, typically 2048 × 2048 (1:1), 2304 × 1728 (4:3), 1728 × 2304 (3:4), 2688 × 1536 (16:9), and 1536 × 2688 (9:16).
> - **`2kto4k`** — the up-to-4K-resolution decoder, trained with varying resolution (from 2K to 4K). Multiple aspect ratios are supported. Worse than `2k` at 2048px resolution.
>
> For the exact checkpoint path for each backbone, see [docs/checkpoints.md](docs/checkpoints.md).


| `--backbone`   | Currently available `--pid_ckpt_type` |
|----------------|:-------------------------------------:|
| flux           | `2k`, `2kto4k` |
| flux2          | `2k`, `2kto4k` |
| flux2-klein-4b | `2k`, `2kto4k` |
| flux2-klein-9b | `2k`, `2kto4k` |
| sd3            | `2k`, `2kto4k` |
| zimage         | `2k`, `2kto4k` |
| zimage-turbo   | `2k`, `2kto4k` |
| sdxl           | `2kto4k` |
| qwenimage      | `2kto4k` |
| qwenimage-2512 | `2kto4k` |
| dinov2 (RAE)   | `2k` |
| siglip (Scale-RAE) | `2k` |

For the exact checkpoint path behind each `(backbone, --pid_ckpt_type)`, see [docs/checkpoints.md](docs/checkpoints.md).

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

#### Example 2 — Single-GPU, 4K decode with 4:3 aspect ratio (Flux, `2kto4k` decoder)

Same backbone as Example 1 but with `--resolution 4096,3072 --pid_ckpt_type 2kto4k`.
`--resolution` is the final output size, so the LDM runs at `1024,768` and
PiD decodes it to 4K.

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone flux \
    --prompt "A close photograph of a cat looking through frosted glass beside a small pine branch, winter light, soft condensation, simple cozy composition, expressive eyes." \
    --resolution 4096,3072 --pid_ckpt_type 2kto4k \
    --ldm_inference_steps 28 --save_xt_steps 24 26 \
    --output_dir ./results/official_demo/flux_4k_ar4_3
```

#### Example 3 — Multi-GPU with a prompt file (Z-Image) with torch.compile

`torchrun` shards `--prompt_file` across ranks; each rank writes to
`--output_dir` independently. We use `--compile` to enable torch.compile for faster inference,
the first call will be slow due to the compilation. We use `default` compilation mode, to get further speedup, change to the `max-autotune` mode in `_maybe_compile_net (pid/_src/models/pixeldit_model.py:210)`.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm --backbone zimage \
    --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 50 --save_xt_steps 46 \
    --compile \
    --output_dir ./results/official_demo/zimage
```

#### Example 4 — Multi-GPU, 4K decode (Z-Image-Turbo, `2kto4k` decoder)

Z-Image-Turbo defaults to 9 diffusers steps with `guidance_scale=0.0`. The final
clean latent `x0` is always saved and is the recommended Turbo output to inspect.
`--save_xt_steps 7` is optional; it saves an additional near-final `x_t` sample
for comparison. `--resolution 4096` means `H=4096, W=4096` and the LDM runs at `1024,1024`.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference.from_ldm --backbone zimage-turbo \
    --prompt_file pid/_src/inference/prompts/prompt_zimage_turbo.txt \
    --resolution 4096 --pid_ckpt_type 2kto4k \
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
| flux2-klein-4b | `--ldm_inference_steps` | 4      | `2 3`                      | `x0`               |
| flux2-klein-9b | `--ldm_inference_steps` | 4      | `2 3`                      | `x0`               |
| qwenimage | `--ldm_inference_steps` | 50 | `44 46 48`             | step `46`          |
| qwenimage-2512 | `--ldm_inference_steps` | 50 | `44 46 48`             | step `46`          |
| zimage   | `--ldm_inference_steps` | 50            | `44 46 48`                 | step `46`          |
| zimage-turbo | `--ldm_inference_steps` | 9         | `7`                        | `x0`               |

---
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

## Repository layout

```
pid/_src/inference/
├── from_ldm.py            # entrypoint: text/class → LDM → PiD decode (--backbone …)
├── from_clean.py          # entrypoint: image → VAE encode → PiD decode (--backbone …)
├── cli_utils.py           # argument parsers + backbone aliases for both entrypoints
├── decoder.py             # shared PiD decode/save core (+ from_clean VAE round-trip & noising)
├── step_capture.py        # diffusers callbacks: XtCaptureCallback / X0CaptureCallback
├── inference_utils.py     # image/prompt/manifest IO, save_image, tags, AsyncUploader, S3 helpers
├── checkpoint_registry.py # backbone → PiD checkpoint mapping
├── pipeline_registry.py   # diffusers backbone → HF pipeline mapping
├── rae_generation.py      # DINOv2-RAE backend + run_rae_demo (--backbone dinov2)
├── scale_rae_generation.py# Scale-RAE backend + run_scale_rae_demo (--backbone siglip)
└── prompts/               # prompt files
```

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
