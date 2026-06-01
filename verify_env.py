"""Verify the PiD inference env is correctly set up.

Imports every third-party package the inference code touches plus the
`from_clean_*` / `from_ldm_*` entry-point modules. Successful imports mean the
env is ready to run smoke tests — no model weights are downloaded and no
inference is performed.

The dinov2 / siglip backbones depend on upstream RAE / Scale-RAE repos (see
docs/dinov2_siglip.md) and are reported as optional: failures there are
printed as [SKIP] and do not fail the script.

Usage:
    python verify_env.py
"""

import importlib
import sys
import traceback

THIRD_PARTY_PACKAGES = [
    # Core compute.
    "torch",
    "torchvision",
    # HuggingFace stack.
    "diffusers",
    "transformers",
    "safetensors",
    "huggingface_hub",
    "sentencepiece",
    # Imaging / IO.
    "numpy",
    "pandas",
    "PIL",
    "imageio",
    "cv2",
    "einops",
    # Config / experiment plumbing.
    "hydra",
    "omegaconf",
    "yaml",
    "attrs",
    "attr",
    # Logging / utils.
    "loguru",
    "termcolor",
    "fvcore",
    "iopath",
    "wandb",
    "packaging",
    # Optional outputs (lazy boto3 import inside the run loop).
    "boto3",
    "botocore",
]

DIFFUSERS_PIPELINES = [
    "FluxPipeline",
    "Flux2Pipeline",
    "StableDiffusion3Pipeline",
    "ZImagePipeline",
]

# The two unified demo dispatchers + the dataset generator. Each backbone is selected
# at runtime via --backbone (see from_ldm.py / from_clean.py).
REQUIRED_INFERENCE_MODULES = [
    "pid._src.inference.from_ldm",
    "pid._src.inference.from_clean",
]

# Non-diffusers backends. They import cleanly without the upstream RAE / Scale-RAE
# repos (those are imported lazily inside the load/sample helpers).
OPTIONAL_INFERENCE_MODULES = [
    "pid._src.inference.rae_generation",
    "pid._src.inference.scale_rae_generation",
]


def try_import(name):
    try:
        importlib.import_module(name)
        return True, None
    except Exception:
        return False, traceback.format_exc()


def section(title):
    print(f"\n=== {title} ===")


def main():
    failures = []
    optional_failures = []

    section("Third-party packages")
    for pkg in THIRD_PARTY_PACKAGES:
        ok, err = try_import(pkg)
        print(f"  [{'PASS' if ok else 'FAIL'}] {pkg}")
        if not ok:
            failures.append((pkg, err))

    section("Diffusers pipelines (need diffusers >= 0.37)")
    try:
        import diffusers

        for cls in DIFFUSERS_PIPELINES:
            present = hasattr(diffusers, cls)
            print(f"  [{'PASS' if present else 'FAIL'}] diffusers.{cls}")
            if not present:
                failures.append((f"diffusers.{cls}", "attribute not present on diffusers module"))
    except Exception:
        # `diffusers` import itself failed; already counted above.
        pass

    section("Inference entry points (diffusers backbones)")
    for mod in REQUIRED_INFERENCE_MODULES:
        ok, err = try_import(mod)
        print(f"  [{'PASS' if ok else 'FAIL'}] {mod}")
        if not ok:
            failures.append((mod, err))

    section("Optional inference entry points (dinov2 / siglip)")
    for mod in OPTIONAL_INFERENCE_MODULES:
        ok, err = try_import(mod)
        print(f"  [{'PASS' if ok else 'SKIP'}] {mod}")
        if not ok:
            optional_failures.append((mod, err))

    section("torch + CUDA runtime")
    try:
        import torch

        print(f"  torch:                {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_ok}")
        if cuda_ok:
            print(f"  torch.version.cuda:   {torch.version.cuda}")
            print(f"  device count:         {torch.cuda.device_count()}")
            print(f"  device[0]:            {torch.cuda.get_device_name(0)}")
            t = torch.zeros(1, device="cuda")
            _ = (t + 1).cpu()
            print(f"  [PASS] cuda kernel + d2h copy")
        else:
            print(f"  [FAIL] torch.cuda.is_available() returned False")
            failures.append(("torch.cuda", "torch.cuda.is_available() returned False"))
    except Exception:
        failures.append(("torch.cuda", traceback.format_exc()))

    print()
    if failures:
        print(f"[FAIL] {len(failures)} required check(s) failed:\n")
        for name, err in failures:
            print(f"--- {name} ---")
            if err:
                print(err)
            print()
        sys.exit(1)

    print("[PASS] Environment OK — all required imports and CUDA checks passed.")
    if optional_failures:
        print(
            f"  ({len(optional_failures)} optional dinov2/siglip module(s) skipped — "
            "install upstream RAE / Scale-RAE to enable them.)"
        )


if __name__ == "__main__":
    main()
