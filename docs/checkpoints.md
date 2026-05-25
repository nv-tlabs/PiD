# Checkpoint Reference

| Backbone | `--pid_ckpt_type 2k` (default) | `--pid_ckpt_type 2kto4k` |
|----------|--------------------------------|--------------------------|
| flux     | `checkpoints/PiD_res2k_sr4x_official_flux_distill_4step`      | `checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step`  |
| flux2    | `checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step`     | `checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step` |
| sd3      | `checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step`       | `checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step`   |
| zimage   | reuses `flux` (Z-Image shares Flux's VAE)                     | reuses `flux` 2kto4k                                           |
| zimage_turbo | reuses `flux` (Z-Image-Turbo shares Flux's VAE)            | reuses `flux` 2kto4k                                           |
| dinov2   | `checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step`    | -                                                              |
| siglip   | `checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step`    | -                                                              |

All released checkpoints are 4-step distilled. The `flux` / `flux2` / `sd3` /
`zimage` / `zimage_turbo` / `dinov2` checkpoints decode at 4x upscale; the
`siglip` checkpoint decodes at 8x (256 -> 2048, Scale-RAE's native interface).
