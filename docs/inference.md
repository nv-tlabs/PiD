# Inference with Training Checkpoints

The entry points in `pid/_src/inference_internal` load checkpoints produced by
the experiments in `pid/_src/configs/pid_training`. They are separate from the
released-checkpoint inference interfaces documented in the main README. Run all
commands below from the repository root with `PYTHONPATH=.`.

| Training workflow | Entry point | Typical sampling settings |
| --- | --- | --- |
| PixelDiT finetuning | `python -m pid._src.inference_internal.pixeldit` | CFG `2.75`, 50 steps |
| PiD teacher training | `python -m pid._src.inference_internal.pid_inference` | CFG `5`, 25 steps |
| PiD student distillation | `python -m pid._src.inference_internal.pid_inference` | 4 distilled steps |

Always use the same experiment name that created the checkpoint. The registered
families are:

- PixelDiT: `pixeldit_text_to_image_finetune_res_2048` and
  `pixeldit_text_to_image_finetune_res_2048_to_3840`.
- PiD teacher:
  `pid_v1pt5_teacher_<family>_h1024_d4_fix_backbone_res_2048`, with an optional
  `_to_3840` suffix after `2048`. `<family>` is `flux`, `flux2`, or
  `qwenimage`.
- PiD student:
  `pid_v1pt5_student_<family>_h1024_d4_res_2048_distill`, or
  `pid_v1pt5_student_<family>_h1024_d4_res_2048_to_3840_distill`.

## Checkpoint Paths

For a distributed checkpoint (DCP), pass the iteration directory itself:

```text
/path/to/run/checkpoints/iter_XXXXXXXXX
```

Do not pass the parent `checkpoints/` directory or the nested `model/`
directory. Add `--load_ema_to_reg` when you want the EMA weights from a DCP
loaded into the inference model.

A consolidated checkpoint can be passed directly, for example:

```text
/path/to/run/model_ema_bf16.pth
```

In that case, select the regular or EMA `.pth` file explicitly; the
`--load_ema_to_reg` flag is not needed. See [Checkpointing](checkpointing_EN.md)
for DCP layout and conversion instructions.

## PixelDiT Inference

PixelDiT maps a text prompt directly to an image. This single-GPU example loads
EMA weights from a 2K DCP checkpoint:

```bash
EXP=pixeldit_text_to_image_finetune_res_2048
CKPT=/path/to/pixeldit/run/checkpoints/iter_XXXXXXXXX

PYTHONPATH=. python -m pid._src.inference_internal.pixeldit \
    --experiment "$EXP" \
    --checkpoint_path "$CKPT" \
    --prompt "A majestic snow-capped mountain range at golden hour" \
    --output_dir ./results/training_checkpoints/pixeldit \
    --cfg_scale 2.75 --num_steps 50 \
    --load_ema_to_reg
```

For a prompt file, `torchrun` distributes prompts across workers. The
multi-resolution experiment also accepts `--image_size`:

```bash
EXP=pixeldit_text_to_image_finetune_res_2048_to_3840
CKPT=/path/to/pixeldit_multires/run/checkpoints/iter_XXXXXXXXX

PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference_internal.pixeldit \
    --experiment "$EXP" \
    --checkpoint_path "$CKPT" \
    --prompts_file pid/_src/dataprep/prompts/prompts_example.txt \
    --image_size 3840 --batch_size 1 \
    --output_dir ./results/training_checkpoints/pixeldit_3840 \
    --cfg_scale 2.75 --num_steps 50 \
    --load_ema_to_reg
```

The script appends a checkpoint/sampling tag to `--output_dir` and saves PNGs
with stable prompt indices such as `0000.png`.

## PiD Input Modes

The teacher and student share the same PiD inference entry point and support two
mutually exclusive input modes:

- `--input_path <image> --caption <text>` encodes one LQ image with the VAE from
  the selected experiment. Use one process for this mode.
- `--fix_batch_dir <directory>` reads precomputed `.pt` files containing
  `caption`, `LQ_latent`, `LQ_video_or_image`, and `degrade_sigma`. This mode can
  be sharded across several GPUs with `torchrun`.

Match the experiment family and resolution to the callback assets. The default
training configurations use:

| Experiment family | 2K fix-batch example | 2K-to-4K fix-batch example |
| --- | --- | --- |
| `flux` | `assets/pid_callback_assets/zimage/full_step/2048` | `assets/pid_callback_assets/zimage/full_step/4096` |
| `flux2` | `assets/pid_callback_assets/flux2/full_step/2048` | `assets/pid_callback_assets/flux2/full_step/4096` |
| `qwenimage` | `assets/pid_callback_assets/qwenimage/full_step/2048` | `assets/pid_callback_assets/qwenimage/full_step/4096` |

The FLUX PiD configuration uses Z-Image callback assets because they share the
compatible FLUX VAE latent space. Intermediate LDM latents, such as `46step`,
can be used in place of `full_step` when they match the training setup.

## PiD Teacher Inference

The following example performs 4x decoding/super-resolution from one image with
a Qwen-Image teacher checkpoint:

```bash
EXP=pid_v1pt5_teacher_qwenimage_h1024_d4_fix_backbone_res_2048
CKPT=/path/to/teacher/run/checkpoints/iter_XXXXXXXXX

PYTHONPATH=. python -m pid._src.inference_internal.pid_inference \
    --experiment "$EXP" \
    --checkpoint_path "$CKPT" \
    --input_path assets/0072.jpg \
    --caption "A tranquil alpine lakeside scene framed by forested mountains" \
    --output_dir ./results/training_checkpoints/pid_teacher \
    --save_format png \
    --cfg_scale 5 --num_steps 25 \
    --load_ema_to_reg
```

To evaluate the precomputed callback set instead, replace the single-image
arguments with a fix-batch directory:

```bash
PYTHONPATH=. python -m pid._src.inference_internal.pid_inference \
    --experiment "$EXP" \
    --checkpoint_path "$CKPT" \
    --fix_batch_dir assets/pid_callback_assets/qwenimage/full_step/2048 \
    --max_samples 16 --batch_size 1 \
    --output_dir ./results/training_checkpoints/pid_teacher_fix_batch \
    --save_format png \
    --cfg_scale 5 --num_steps 25 \
    --load_ema_to_reg
```

## PiD Student Inference

Use the student experiment that matches the distilled checkpoint. This example
shards a fix-batch directory across four GPUs:

```bash
EXP=pid_v1pt5_student_qwenimage_h1024_d4_res_2048_distill
CKPT=/path/to/student/run/checkpoints/iter_XXXXXXXXX

PYTHONPATH=. torchrun --nproc_per_node=4 \
    -m pid._src.inference_internal.pid_inference \
    --experiment "$EXP" \
    --checkpoint_path "$CKPT" \
    --fix_batch_dir assets/pid_callback_assets/qwenimage/full_step/2048 \
    --batch_size 1 \
    --output_dir ./results/training_checkpoints/pid_student \
    --save_format png \
    --cfg_scale 1 --num_steps 4 \
    --load_ema_to_reg
```

The current distilled sampler uses its fixed timestep schedule. `--cfg_scale` and `--shift`
do not alter the current student sampler.

The student experiment instantiates the training-state `PidDistillModel`, which
also constructs its teacher, fake-score network, and discriminator. The frozen
teacher is not stored in the distillation DCP, so the configured
`model.config.pretrained_teacher_path` still needs to exist.

These extra networks also make student inference more memory intensive than a
student-only deployment model.

## Common Options and Outputs

| Argument | Meaning |
| --- | --- |
| `--experiment` | Registered experiment matching the checkpoint architecture and resolution. |
| `--config_file` | Config entry point; defaults to `pid/_src/configs/pid_training/config.py`. |
| `--checkpoint_path` | DCP iteration directory or consolidated `.pth` file. |
| `--seed` | Sampling seed. |
| `--batch_size` | Per-process batch size; PixelDiT defaults to `4`, PiD to `1`. |
| `--output_dir` | Output root; the scripts append a checkpoint/sampling tag. |
| `--save_format` | PiD output format, `jpg` or `png`. |
| `--max_samples` | Reproducible cap applied before fix-batch rank sharding. |
| `--degrade_sigma` | Override the sigma stored in fix-batch assets; single-image mode defaults to `0.0`. |

PiD writes model outputs under `<output_dir>/<tag>/`, saves bicubic LQ references
under `<output_dir>/LQ_input/`, and saves non-placeholder ground truth under
`<output_dir>/GT/` when available.

Unrecognized trailing arguments are forwarded as Hydra experiment overrides.
Append them directly, without another `--` separator:

```bash
... model.config.dynamic_shift.base_shift=7.0
```

For multi-GPU runs, PixelDiT shards prompts and PiD shards fix-batch files. PiD
single-image mode is not sharded, so launching it with several ranks would
duplicate work and target the same output path. This is data-parallel work
sharding: every rank loads a complete model, so it does not reduce per-GPU model
memory. Standalone inference does not initialize the context-parallel groups
used during training, so `--nproc_per_node` does not need to match the
experiment's training-time context-parallel size.
