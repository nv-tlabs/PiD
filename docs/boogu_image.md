# Boogu-Image Optional Setup

`from_boogu.py` is an optional integration for
[Boogu-Image](https://github.com/boogu-project/Boogu-Image). It is intentionally
not part of PiD's base environment because Boogu brings a separate custom
pipeline and a heavier dependency stack.

You do not need to update `environment.yml` or the base `pyproject.toml`
dependencies to use the regular PiD entry points. Install the Boogu overlay only
in environments where you plan to run Boogu.

## Recommended Setup

Clone Boogu under `third_party/` and install it as an editable package:

```bash
git clone https://github.com/boogu-project/Boogu-Image third_party/Boogu-Image

python -m pip install -e third_party/Boogu-Image --no-deps
python -m pip install \
    "cache-dit>=1.3,<2" \
    "kernels>=0.14,<0.15" \
    "torchao>=0.15,<0.18" \
    "python-dotenv>=1.0,<2" \
    "scipy>=1.11" \
    "webdataset>=1.0,<2"
```

Boogu's dependency metadata also lists `torchaudio`, but it is not required for
the text-to-image path tested here. Install `torchaudio` only when you can match
the exact CUDA build of your PyTorch wheel; otherwise transformers may fail
while importing audio helpers.

If pip wants to upgrade `transformers` / `huggingface-hub`, prefer doing that in
a Boogu-specific environment rather than in the shared PiD environment. The local
smoke test was run with:

```text
torch 2.8.0+cu129
diffusers 0.38.0
transformers 5.13.1
huggingface-hub 1.23.0
boogu-image 0.1.0 editable
cache-dit 1.5.0
kernels 0.14.1
torchao 0.17.0
```

`torchao` may warn that its C++ extension prefers a newer torch version. The
standard bf16 Boogu Turbo checkpoint still ran successfully; fp8 checkpoints may
need a cleaner Boogu-native torch stack.

## Run Boogu Native T2I

Run from the PiD repository root:

```bash
export PYTHONPATH=.:third_party/Boogu-Image

python -m pid._src.inference.from_boogu \
    --variant turbo \
    --prompt "A tiny ceramic robot watering a bonsai tree" \
    --resolution 1024 \
    --enable_model_cpu_offload \
    --output_dir ./results/boogu/turbo
```

Useful presets:

- `--variant base`: `Boogu/Boogu-Image-0.1-Base`, 50 steps, text guidance `4.0`.
- `--variant turbo`: `Boogu/Boogu-Image-0.1-Turbo`, 4 steps, text guidance `1.0`.

For prompt files, `torchrun` shards prompts round-robin across ranks:

```bash
PYTHONPATH=.:third_party/Boogu-Image torchrun --nproc_per_node=8 \
    -m pid._src.inference.from_boogu \
    --variant turbo \
    --prompt_file pid/_src/inference/prompts/prompt_zimage_turbo.txt \
    --resolution 1024 \
    --enable_model_cpu_offload \
    --output_dir ./results/boogu/turbo_batch
```

## Run Boogu with Flux1 PiD Decode

Boogu uses a Flux1-style 16-channel VAE latent. `--pid_decode` captures Boogu's
final normalized latent immediately before native VAE decode, then feeds it to
the Flux1 PiD pixel decoder.

```bash
PYTHONPATH=.:third_party/Boogu-Image python -m pid._src.inference.from_boogu \
    --variant turbo \
    --prompt "A tiny ceramic robot watering a bonsai tree" \
    --resolution 512 \
    --enable_model_cpu_offload \
    --pid_decode \
    --output_dir ./results/boogu/turbo_pid
```

With the default `--scale 4`, a `512x512` Boogu latent decode produces a `2048x2048`
PiD image. For 4K output, use a `1024x1024` Boogu resolution and select the v1.5
multi-resolution Flux checkpoint:

```bash
PYTHONPATH=.:third_party/Boogu-Image python -m pid._src.inference.from_boogu \
    --variant turbo \
    --prompt "A tiny ceramic robot watering a bonsai tree" \
    --resolution 1024 \
    --enable_model_cpu_offload \
    --pid_decode \
    --pid_ckpt_type 2kto4k_v1pt5 \
    --output_dir ./results/boogu/turbo_pid_4k
```

The smoke test captured a Boogu latent with shape `(1, 16, 64, 64)` for a
`512x512` native output, matching the Flux1 PiD input convention.

## Troubleshooting

- Use `--enable_model_cpu_offload` for single-GPU runs, especially when combining
  Boogu with PiD decode.
- If `transformers` fails while importing `torchaudio`, remove the mismatched
  `torchaudio` wheel or install one built for the same CUDA version as PyTorch.
- If Hugging Face rate-limits model metadata requests, retry with a configured
  `HF_TOKEN` and keep `HF_HOME` / `HF_HUB_CACHE` on shared scratch.
- If you only need regular PiD `from_ldm.py` / `from_clean.py`, skip this setup.
