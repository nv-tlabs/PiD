"""Official demo: Z-Image-Turbo latent diffusion vs ours pixel-diffusion decoder.

Z-Image-Turbo uses the same diffusers ZImagePipeline/VAE convention as Z-Image,
but is distilled for 8 NFE. The default registry preset uses the model-card
recipe: 9 diffusers inference steps and guidance_scale=0. The final clean latent
`x0` is always saved and is the recommended Turbo output to inspect. For optional
near-final xt comparison, `--save_xt_steps 7` captures a scheduler sigma of 0.3
at 512/1024-style resolutions, close to the base Z-Image `--save_xt_steps 46`
sigma of 0.28125.

PYTHONPATH=. python -m pid._src.inference.from_ldm_zimage_turbo \
    --prompt "A cinematic photo of a neon ramen shop in heavy rain" \
    --output_dir ./results/official_demo/zimage_turbo \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
"""

from pid._src.inference._demo_common import run_demo

if __name__ == "__main__":
    run_demo(backbone="zimage_turbo")
