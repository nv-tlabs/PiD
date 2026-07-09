# Training

This guide covers the three training workflows included in this repository:
PixelDiT finetuning, PiD teacher training, and PiD student distillation. Run all
commands from the repository root. To sample checkpoints produced by these
workflows, see [Inference with Training Checkpoints](inference.md).

## Dataset preparation

Taking [MultiAspect-4K-1M](https://w2genai-lab.github.io/UltraFlux/) as an
example, first download the metadata and extract it:

```bash
mkdir -p raw_data && cd raw_data
wget https://huggingface.co/Owen777/UltraFlux-v1/resolve/main/MultiAspect-4K-1M.tar.gz
tar -xzf MultiAspect-4K-1M.tar.gz
```

Then download the images and reorganize the data into WebDataset format:

```bash
cd ..
# For illustration, download 10 of the 1008 metadata files. Remove
# --max-json-files to download the complete dataset.
python scripts/download_multiaspect_4k_1m.py --max-json-files 10

# Convert the downloaded data to WebDataset shards.
python scripts/sharding_wds.py --input-dir raw_data/MultiAspect-4K-1M-download --output-dir data/image_MultiAspect_4K_1M_webdataset
```

To add your own training data, use a directory structure like
`raw_data/MultiAspect-4K-1M-download`, convert it with
`scripts/sharding_wds.py`, register the source in
`pid/_src/datasets/data_sources/data_source_local.py`, and include it in
`pid/_src/datasets/data_sources/dataset_definition.py`.

The data structure is described in [WebDataset (Chinese)](webdataset_CN.md) and
[WebDataset (English)](webdataset_EN.md).


## PixelDiT Finetuning

We provide a finetuned PixelDiT checkpoint in `checkpoints/PixelDiT_finetune_2kto4k/model_ema_bf16.pth`. If you want to further finetune it, you can use the following command:

### Set up environment variables

```bash
# Create output and cache directories, then set their absolute paths.
mkdir -p imaginaire4/imaginaire4-output imaginaire4/imaginaire4-cache
export IMAGINAIRE_OUTPUT_ROOT=$(realpath "./imaginaire4/imaginaire4-output")
export IMAGINAIRE_CACHE_DIR=$(realpath "./imaginaire4/imaginaire4-cache")
export WANDB_API_KEY=your_wandb_api_key
```

### Training Command

```bash
# finetune on 2k resolution only. The dataloader only ships 2k resolution images.
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048"

# finetune on 2k to 4k resolution. The dataloader provides various resolution images.
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048_to_3840"
```
The configuration code lives in `pid/_src/configs/pid_training/experiment_pixeldit_finetune/finetune.py`.

> [!TIP]
> To learn more about the configuration system, please refer to [hydra_CN.md](hydra_CN.md) or [hydra_EN.md](hydra_EN.md). To learn more about dataloader registration, please refer to [dataloader_CN.md](dataloader_CN.md) or [dataloader_EN.md](dataloader_EN.md).

In `pixeldit_text_to_image_finetune_res_2048_to_3840` experiment, we assume your data source contains images in resolutions ranging from 2k to 4k. If you only use MultiAspect-4K-1M, the images from dataloader will be mostly 4k resolution.

See [PixelDiT inference](inference.md#pixeldit-inference) to sample a finetuned
checkpoint.

### Output Structure
All training artifacts are saved in `IMAGINAIRE_OUTPUT_ROOT` (set above to
`imaginaire4/imaginaire4-output`).

Each job is written to
`IMAGINAIRE_OUTPUT_ROOT/<project_name>/<group_name>/<job_name>`. The project is
`pid_training`; the experiment configuration overrides the group and job names.

```python
        ...
        job=dict(
            group="pixeldit_finetune",
            name="pixeldit_text_to_image_finetune_res_2048",
        ),
        ...
```

The resulting directory has the following structure:

```bash
imaginaire4/imaginaire4-output/pid_training/pixeldit_finetune/pixeldit_text_to_image_finetune_res_2048
├── DeviceMonitor
├── checkpoints
├── EveryNDrawSample
├── config.pkl
├── config.yaml
├── job_env.yaml
├── launch_info.yaml
├── stdout.log
├── wandb
└── wandb_id.txt
```

### Debug Mode

Add the `_debug` suffix to an experiment name to run fewer iterations with
Weights & Biases logging disabled. For example:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pixeldit_text_to_image_finetune_res_2048_debug"
```

## PiD Teacher Training

We provide PiD v1.5 teacher examples for FLUX, FLUX.2, and the Qwen-Image
(Wan 2.1) VAE. Other VAE or vision-encoder families can follow the same pattern.

### Prepare callback assets

First generate the latent assets used by the visualization and evaluation
callbacks. These are the default asset families referenced by the three teacher
configs:

```bash
bash pid/_src/dataprep/fix_batch_generation/generate_callback_assets.sh zimage flux2 qwenimage
```

This runs the corresponding Diffusers pipelines and saves their latents and
sigmas under `assets/pid_callback_assets`. The FLUX PiD configuration can use
Z-Image assets because both use the compatible FLUX VAE latent space; the other
configurations use their matching `flux2` or `qwenimage` assets.

By default, the script uses prompts from
`pid/_src/dataprep/prompts/prompts_harder_cases.txt` and generates 512 x 512 LDM
samples on four GPUs. Override the defaults with environment variables:

```bash
# Generate 512 x 512 and 1024 x 1024 assets on eight GPUs.
RESOLUTIONS="512 1024" NPROC=8 bash pid/_src/dataprep/fix_batch_generation/generate_callback_assets.sh zimage flux2 qwenimage
```
See `pid/_src/dataprep/fix_batch_generation/generate_callback_assets.sh` for
all available overrides.

> [!TIP]
> Callback assets are `.pt` files containing `HQ_video_or_image`, `caption`,
> `LQ_video_or_image`, `degrade_sigma`, and `LQ_latent`. Generated assets use a
> placeholder for `HQ_video_or_image`; paired LQ/HQ data can instead be used to
> evaluate reconstruction quality.

### Training Command
```bash
# Flux. 2k resolution training.
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_flux/teacher.py
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_teacher_flux_h1024_d4_fix_backbone_res_2048"

# Flux2. 2k resolution training.  
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_flux2/teacher.py
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_teacher_flux2_h1024_d4_fix_backbone_res_2048"

# QwenImage (wan 2.1) VAE. 2k resolution training.
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_qwenimage/teacher.py
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_teacher_qwenimage_h1024_d4_fix_backbone_res_2048"
```

PiD v1.5 makes the following changes to improve decoding color accuracy and reduce grid artifacts:

1. The latent projection adapter uses a hidden dimension of 1024.
2. An auxiliary reconstruction loss provides RGB color supervision.
3. The PixelDiT backbone is initially frozen and can be unfrozen in later finetuning.

Each configuration file also includes its 2K-to-4K experiment. See
[PiD teacher inference](inference.md#pid-teacher-inference) to evaluate a teacher
checkpoint.

## PiD Student Distillation

We provide PiD v1.5 student distillation examples for FLUX, FLUX.2, and the
Qwen-Image (Wan 2.1) VAE. Other VAE or vision-encoder families can follow the
same pattern.

```bash
# Flux. 2k resolution distillation.
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_flux/distillation.py
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_student_flux_h1024_d4_res_2048_distill"

# Flux2. 2k resolution distillation.
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_flux2/distillation.py
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_student_flux2_h1024_d4_res_2048_distill"

# QwenImage (wan 2.1) VAE. 2k resolution distillation.
# Configuration in pid/_src/configs/pid_training/experiment_pid_v1pt5_qwenimage/distillation.py
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
      --config=pid/_src/configs/pid_training/config.py \
      -- experiment="pid_v1pt5_student_qwenimage_h1024_d4_res_2048_distill"
```

Each configuration file also includes its 2K-to-4K experiment. See
[PiD student inference](inference.md#pid-student-inference) to evaluate a
distilled checkpoint.

## Tips

- `2kto4k` and distillation experiments are memory intensive. If a run is out of memory, reduce the batch size
  in dataloader or increase `model_parallel.context_parallel_size` in the configuration.
  Context parallelism reduces per-GPU memory use, but also reduces the global batch size for a fixed GPU count.
- multi nodes training is recommended to increase the global batch size. Our codebase is compatible with SLURM
  multi nodes training. We provide a SLURM script in `scripts/multinode_slurm.sh` as an example. Update according
  to your environment and `sbatch scripts/multinode_slurm.sh` to submit the job.

## Related Links

- Hydra configuration system: [中文](hydra_CN.md) | [English](hydra_EN.md)
- WebDataset data format and loading pipeline: [中文](webdataset_CN.md) | [English](webdataset_EN.md)
- Dataloader configuration groups and data sources: [中文](dataloader_CN.md) | [English](dataloader_EN.md)
- Checkpoint layout, resuming, and conversion: [中文](checkpointing_CN.md) | [English](checkpointing_EN.md)
- Inference with training checkpoints: [English](inference.md)
