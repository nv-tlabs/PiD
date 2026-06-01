# DINOv2 / SigLIP-2 from Clean Image
The `from_clean --backbone dinov2` and `--backbone siglip` flows accept the same flags as
the diffusers backbones (see the main README). The input image is fed at its native
resolution (only center-cropped to a 16-multiple); each encoder then resizes internally to
its own fixed native interface, so only `--scale` needs to match. The `dinov2` backbone is
the upstream **RAE** (DINOv2 encoder); the `siglip` backbone is the upstream **Scale-RAE**
(SigLIP-2 encoder):

- `dinov2` (RAE)       → `--scale 4` (native 512 → 2048)
- `siglip` (Scale-RAE) → `--scale 8` (native 256 → 2048)


# DINOv2 RAE / SigLIP-2 from Latent Diffusion Model

`from_ldm --backbone dinov2` and `--backbone siglip` wrap two latent-diffusion models that
are **not** distributed through `diffusers` — the `dinov2` backbone is the upstream
class-conditional ImageNet-512 [RAE](https://github.com/bytetriper/RAE) and the `siglip`
backbone is the text-conditional 256px [Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE).
These backbones additionally need the upstream LDM repos on `sys.path`.

> [!NOTE]
> LDM in vision encoder space is hard to train. When the latent itself is
> highly unstructured and unreasonable, RAE decoder produces unsatisfactory
> results, and PiD cannot correct it as well.

## Installation

```bash
# 1) Clone the upstream repos NEXT TO the pid repo (as siblings, not inside).
#    Run these from the directory that *contains* your pid checkout so the
#    repos land at ../RAE and ../Scale-RAE relative to the pid working tree —
#    this is the default the inference scripts look for, and keeps the pid
#    working tree clean. Any other location works as long as you point the
#    RAE_REPO_PATH / SCALE_RAE_REPO_PATH env vars (or CLI flags) at it.
cd ..
git clone https://github.com/bytetriper/RAE.git
git clone https://github.com/ZitengWangNYU/Scale-RAE.git
cd pid

# 2) Install Scale-RAE (--no-deps because its pyproject pins torch/torchvision/
#    transformers/tokenizers that would clobber the rest of the env). Then add
#    the runtime deps the upstream code actually needs.
#
#    IMPORTANT — downgrade transformers to <5: Scale-RAE's custom Qwen LM is
#    tightly coupled to the transformers 4.x API. On transformers 5.x the LM
#    silently emits degenerate image embeddings (16×16 tile pattern) without
#    raising. The base environment.yml leaves transformers unpinned (the core
#    diffusers backbones work on both 4.x and 5.x); this pin is *only* needed
#    if you use the siglip / Scale-RAE backbone.
#
#    Expected: the second `pip install` prints ~10 lines of
#      "scale-rae 1.0.0 requires <pkg>==<old-version>, but you have …"
#    These warnings are by design — Scale-RAE's pyproject pins ancient
#    versions of peft / torchtext / accelerate / transformers / tokenizers
#    that we deliberately ignore via `--no-deps` above. The PiD inference
#    code only exercises the parts of Scale-RAE that work with the newer
#    versions; the siglip (Scale-RAE) from_ldm / from_clean flows have been
#    verified end-to-end against the versions installed below.
pip install --no-deps -e ../Scale-RAE
pip install torchdiffeq timm omegaconf ezcolorlog shortuuid open_clip_torch accelerate
pip install "transformers==4.57.1"

# 3) Point the demos at the repos. The CLI flags fall back to these env vars,
#    which default to ../RAE and ../Scale-RAE (sibling of the pid working tree).
export RAE_REPO_PATH=$(realpath ../RAE)
export SCALE_RAE_REPO_PATH=$(realpath ../Scale-RAE)

# 4) Download the RAE weights. Scale-RAE weights are downloaded automatically.
cd $RAE_REPO_PATH
hf download nyu-visionx/RAE-collections \
    --local-dir models
```

## `from_ldm`: class / text → upstream LDM → PiD decode

Class-conditional example (DINOv2-RAE, ImageNet-512):

```bash
export RAE_REPO_PATH=$(realpath ../RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone dinov2 \
    --load_ema_to_reg \
    --rae_class_ids 207 281 387 \
    --num_inference_steps 50 --save_xt_steps 44 46 48 \
    --output_dir ./results/official_demo/dinov2 \
    --pid_inference_steps 4 --scale 4
```

Text-conditional example (Scale-RAE, 256 → 2048 at 8×):

```bash
export SCALE_RAE_REPO_PATH=$(realpath ../Scale-RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone siglip \
    --load_ema_to_reg \
    --prompt "A cat sitting on a windowsill at sunset" \
    --save_xt_steps 44 46 48 \
    --output_dir ./results/official_demo/siglip \
    --pid_inference_steps 4 --scale 8
```

Suggested step counts (see each entrypoint's docstring for the exact recipe):

| `--backbone` | LDM steps flag          | Default steps | `--save_xt_steps` (example) |
|--------------|-------------------------|---------------|-----------------------------|
| dinov2 (RAE) | `--num_inference_steps` | 50            | `44 46 48`                  |
| siglip (Scale-RAE) | (no flag; LM-driven) | —          | `44 46 48`                  |
