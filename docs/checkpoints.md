# Checkpoint Reference

| Backbone | `--pid_ckpt_type 2k` (default) | `--pid_ckpt_type 2kto4k` |
|----------|--------------------------------|--------------------------|
| flux     | `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step`      | `checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step`  |
| flux2    | `checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step`     | `checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606` |
| sd3      | `checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step`       | `checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step`   |
| zimage   | `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step`      | `checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step`  |
| zimage_turbo | `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step`  | `checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step`  |
| sdxl     | -                                                            | `checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step`  |
| qwenimage | -                                                           | `checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step` |
| qwenimage_2512 | -                                                      | `checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step` |
| dinov2   | `checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step`    | -                                                              |
| siglip   | `checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step`    | -                                                              |

All released checkpoints are 4-step distilled. The `flux` / `flux2` / `sd3` /
`sdxl` / `qwenimage` / `zimage` / `zimage_turbo` / `dinov2` checkpoints decode at
4x upscale; the `siglip` checkpoint decodes at 8x (256 -> 2048, Scale-RAE's native
interface). `sdxl` and `qwenimage` (incl. `qwenimage_2512`) ship only the `2kto4k`
decoder.

## VAE encoder weights

Each backbone's pixel decoder loads a frozen VAE encoder/decoder. These weights are
read **locally** from `checkpoints/` (no HuggingFace download at runtime):

| Backbone | VAE weight file (default) |
|----------|---------------------------|
| flux / zimage / zimage_turbo | `checkpoints/ae.safetensors` |
| sd3      | `checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors` |
| flux2    | `checkpoints/flux2_ae.safetensors` |
| sdxl     | `checkpoints/sdxl_vae.safetensors` |
| qwenimage / qwenimage_2512 | `checkpoints/QwenImage_VAE_2d.pth` (2D-stripped AutoencoderKLQwenImage) |
| dinov2 | DINOv2-B encoder weights + decoder in `checkpoints/rae` |
| siglip | SigLIP-2 encoder weights + decoder in `checkpoints/scale_rae` |
