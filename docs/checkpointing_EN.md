# PiD Training Checkpointing and the DCP Format

## Overview

This project primarily uses PyTorch Distributed Checkpoint (DCP) to save model and training state during training. DCP is well suited to distributed checkpoint saving and training resumption, while a single-file `.pth` checkpoint is generally more convenient for inference or weight initialization in a new training stage.

This document explains:

1. What the DCP format is;
2. The actual checkpoint directory structures used by teacher training and distillation training;
3. How to use `scripts/convert_distcp_to_pt.py` to consolidate DCP model weights into a single `.pth` file.

For an index of released model paths, see [checkpoints.md](checkpoints.md). This document covers only checkpoints produced during training.

## Key Files

| File | Purpose |
|---|---|
| `pid/_ext/imaginaire/checkpointer/dcp.py` | `DistributedCheckpointer`, used for standard teacher/model training |
| `pid/_src/checkpointer/dcp_distill.py` | `DistillationCheckpointer`, used for distillation training with multiple optimizers and schedulers |
| `pid/_src/configs/common/defaults/ckpt_type.py` | Registers the Hydra options `dcp` and `dcp_distill` |
| `scripts/convert_distcp_to_pt.py` | Converts a DCP state-dict directory into a single `.pth` file |

Teacher experiments typically select:

```python
{"override /ckpt_type": "dcp"}
```

Distillation experiments typically select:

```python
{"override /ckpt_type": "dcp_distill"}
```

## What Is the DCP Format?

DCP is a directory-based, sharded checkpoint format provided by `torch.distributed.checkpoint`. Instead of saving a logical state dict as a single `.pt` or `.pth` file, DCP stores it in a directory:

```text
model/
├── .metadata
├── __0_0.distcp
├── __1_0.distcp
└── ...
```

- `.metadata`: Records state-dict keys, tensor shapes, dtypes, chunks, and the locations of the corresponding data in the shard files;
- `__<rank>_<index>.distcp`: Stores the actual tensor and non-tensor payloads;
- The complete directory constitutes one DCP checkpoint. An individual `.distcp` file cannot be used on its own, and `.metadata` should not be edited manually.

The main advantages of DCP are parallel I/O across processes, lower per-process memory pressure during saving and loading, and the ability for the DCP loader to read the required shards according to the current distributed state dict. The tradeoff is that DCP is not a single-file format that can be loaded directly with `torch.load()`, making it less convenient than `.pth` for inspection, transfer, and inference.

This project saves checkpoints with:

```python
DefaultSavePlanner(dedup_save_to_lowest_rank=True)
```

For state replicated identically across DDP ranks, the payload may be written only by the lowest rank, leaving the corresponding `.distcp` files for other ranks at 0 bytes. This occurs in both examples in this document and **does not indicate checkpoint corruption**. In genuinely sharded scenarios such as FSDP or DTensor, data placement may differ, so neither the number nor the size of shard files is fixed.

## Common Checkpoint Hierarchy

Iteration numbers shorter than nine digits are left-padded with zeros:

```text
checkpoints/
├── latest_checkpoint.txt
├── iter_000000025/
├── iter_000005000/
└── ...
```

`latest_checkpoint.txt` is not a symbolic link. It contains the name of the most recently saved checkpoint directory, for example:

```text
iter_000000025
```

When the same job directory is restarted, the checkpointer reads this file first. By default, it restores the model, optimizer, scheduler, and trainer state from the corresponding iteration. Individual components can be excluded with `keys_not_to_resume`.

## Teacher Training Checkpoint

The teacher example examined in this document is located at:

```text
imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/
pid_v1pt5_teacher_flux2_h1024_d4_fix_backbone_res_2048_to_3840_W_RESUME_2026-07-08_08-58-07/checkpoints
```

Its actual directory structure is:

```text
checkpoints/
├── latest_checkpoint.txt                 # iter_000000025
└── iter_000000025/
    ├── model/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    ├── optim/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    ├── scheduler/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    └── trainer/
        ├── .metadata
        ├── __0_0.distcp
        └── __1_0.distcp
```

Directory contents:

| Directory | Contents |
|---|---|
| `model/` | Non-EMA model weights and EMA weights |
| `optim/` | State for a single optimizer, such as the first- and second-moment states of AdamW |
| `scheduler/` | State for a single learning-rate scheduler |
| `trainer/` | Iteration and GradScaler state |

The metadata in this example contains:

- `model/`: 934 keys, including 467 `net.*` keys and 467 `net_ema.*` keys;
- `optim/`: `state.*` and `param_groups.*` entries for a single optimizer;
- `scheduler/`: Fields such as `base_lrs`, `last_epoch`, and `_step_count`;
- `trainer/`: This debug run uses `trainer.grad_scaler_args.enabled=False`, so its metadata contains only the `iteration` key. The value loaded from the shard payload is 25.

The saving code still supplies both `grad_scaler.state_dict()` and `iteration`. With other precisions or configurations, `trainer/` may contain additional fields, so it should not be assumed to contain only the iteration in all cases.

## Distillation Training Checkpoint

The distillation example examined in this document is located at:

```text
imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/
pid_v1pt5_student_flux_h1024_d4_res_2048_distill_W_RESUME_2026-07-08_08-35-44/checkpoints
```

Its actual directory structure is:

```text
checkpoints/
├── latest_checkpoint.txt                 # iter_000000025
└── iter_000000025/
    ├── model/                             # model state
    ├── optim_net/                         # student optimizer
    ├── optim_fake_score/                  # fake-score optimizer
    ├── optim_discriminator/               # discriminator optimizer
    ├── scheduler_net/                     # student scheduler
    ├── scheduler_fake_score/              # fake-score scheduler
    ├── scheduler_discriminator/           # discriminator scheduler
    └── trainer/                           # iteration / GradScaler
```

Each directory above contains its own `.metadata` and shard files ranging from `__0_0.distcp` through `__3_0.distcp`. The optimizer and scheduler states are split into separate DCP directories so that each component can maintain its own parameter mapping.

The directory suffixes come from the keys of the model's `optimizer_dict` and `scheduler_dict`:

```text
optim_<key>/
scheduler_<key>/
```

These names therefore depend on the current configuration and enabled components; they are not permanently fixed to `net`, `fake_score`, and `discriminator`. For example, when the GAN/discriminator is disabled, its corresponding model keys, optimizer, and scheduler directories may be absent.

The `model/.metadata` file in this example contains 1,389 keys:

| Prefix | Count | Meaning |
|---|---:|---|
| `net.*` | 461 | Non-EMA student weights |
| `net_ema.*` | 461 | Student EMA weights |
| `fake_score.*` | 461 | Fake-score network weights |
| `discriminator.*` | 6 | Discriminator weights |

### The Frozen Teacher Is Not Stored in the Distillation DCP

`PidDistillModel.state_dict()` does not save the frozen teacher. The teacher is loaded separately from the configured `pretrained_teacher_path`, so a distillation DCP is not a self-contained package of both teacher and student.

When transferring or restoring a distillation training environment, the DCP directory alone is not sufficient. The teacher checkpoint referenced by the configuration must also remain accessible and compatible.

## Comparing Teacher and Distillation Structures

| Item | Teacher training | Distillation training |
|---|---|---|
| Hydra `ckpt_type` | `dcp` | `dcp_distill` |
| Checkpointer | `DistributedCheckpointer` | `DistillationCheckpointer` |
| Model keys | `net.*`, `net_ema.*` | `net.*`, `net_ema.*`, `fake_score.*`, and optional `discriminator.*` |
| Optimizer | One `optim/` directory | One `optim_<key>/` directory per component |
| Scheduler | One `scheduler/` directory | One `scheduler_<key>/` directory per component |
| Trainer state | `trainer/` | `trainer/` |
| Frozen teacher | Not applicable | Not stored in DCP; must be provided separately |

Each logical directory in the teacher example contains two `.distcp` files, while the distillation example contains four. This difference comes from the distributed world sizes used by the two runs, not from a fixed shard count imposed by either checkpoint format.

## Converting DCP to a Single `.pth` File

Use:

```text
scripts/convert_distcp_to_pt.py
```

When converting model weights, the input must be a DCP directory on the local filesystem and must point exactly to the `model/` directory under a specific iteration:

```text
.../checkpoints/iter_000000025/model
```

Do not pass the `checkpoints/` root or only the `iter_000000025/` directory. The script does not directly accept object-storage URIs such as `s3://...`; download the complete DCP to the local filesystem first.

### Basic Conversion

```bash
mkdir -p /path/to/output_dir
PYTHONPATH=. python scripts/convert_distcp_to_pt.py \
  /path/to/checkpoints/iter_000000025/model \
  /path/to/output_dir
```

The default output is:

```text
/path/to/output_dir/model.pth
```

`model.pth` is the fully consolidated state dict from the supplied `model/` DCP directory. It does not include optimizer, scheduler, or trainer state, because those components are stored in separate directories.

### Exporting EMA Weights for Inference

```bash
mkdir -p /path/to/output_dir
PYTHONPATH=. python scripts/convert_distcp_to_pt.py \
  /path/to/checkpoints/iter_000000025/model \
  /path/to/output_dir \
  --ema
```

`--ema` produces three files:

| Output file | Contents |
|---|---|
| `model.pth` | Complete state dict from `model/` |
| `model_ema_fp32.pth` | Retains only `net_ema.*` and renames it to `net.*`; the script preserves the source dtype, which happens to be fp32 in both examples in this document |
| `model_ema_bf16.pth` | Starting from the EMA-only result above, converts tensors whose original dtype is float32 to bfloat16 while leaving other dtypes and objects unchanged |

For a distillation checkpoint, `--ema` exports the **student EMA**. Neither `fake_score.*` nor `discriminator.*` is included in the EMA-only files, and the frozen teacher was never part of the DCP in the first place.

If the checkpoint does not contain `net_ema.*`, the script first writes `model.pth` and then raises an error during EMA extraction.

### Converting the Teacher Example in This Document

Run the following command from the repository root:

```bash
INPUT="imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/pid_v1pt5_teacher_flux2_h1024_d4_fix_backbone_res_2048_to_3840_W_RESUME_2026-07-08_08-58-07/checkpoints/iter_000000025/model"
OUTPUT="checkpoints/converted/teacher_flux2_iter_000000025"

mkdir -p "$OUTPUT"
PYTHONPATH=. python scripts/convert_distcp_to_pt.py "$INPUT" "$OUTPUT" --ema
```

The distillation example is converted in the same way; only change `INPUT` to its `iter_000000025/model` directory.

## Conversion and Training-Resumption Considerations

- Run the conversion as a single process with regular `python`, not `torchrun`. The conversion consolidates the entire logical state dict in CPU memory.
- The script does not create the output directory; run `mkdir -p` first.
- At startup, the script deletes any existing `model.pth`, `model_ema_fp32.pth`, and `model_ema_bf16.pth` files in the output directory before beginning conversion. Old outputs are therefore already deleted even if `--ema` is not specified or the conversion later fails because of invalid input or an out-of-memory error. Verify the output path before starting.
- The `model/` payloads in the examples in this document are approximately 11 GB for the teacher and 16 GB for distillation. Reserve enough CPU RAM and additional disk space for conversion; peak memory usage may be significantly larger than the file size.
- `--keep-original` exits immediately after writing `model.pth` and does not extract EMA weights. Omitting `--ema` also produces only `model.pth`. If `--keep-original` and `--ema` are specified together, `--keep-original` takes precedence and no EMA-only files are created.
- The script writes files with the `.pth` extension. Both `.pt` and `.pth` are commonly used for single-file PyTorch checkpoints, but this script's outputs should not be searched for under `.pt` filenames.
- A `.pth` file converted from `model/` is suitable for inference or weight initialization in a new training stage, but it does not contain optimizer state, scheduler state, or the trainer iteration. It **cannot replace a complete DCP checkpoint for full-fidelity training resumption**.

### DCP Loading and Resumption Paths

- **Automatic resumption of the same job**: If `checkpoints/latest_checkpoint.txt` exists in the current job directory, the checkpointer prioritizes the local DCP it references and, by default, strictly restores the model, optimizer, scheduler, and trainer state. `load_training_state=False` does not disable this self-resume behavior; only `keys_not_to_resume` can exclude individual components.
- **Explicitly loading an external DCP**: Set `checkpoint.load_path` to the external `iter_xxxxxxxxx/` directory. With `load_training_state=False`, only `model/` is normally loaded. Set it to `True` to also load the optimizer, scheduler, and trainer state. This path does not depend on `latest_checkpoint.txt` in the external directory.
- **Loading a converted `.pth` file**: This initializes model weights only, starts at iteration 0, and cannot restore optimizer state, scheduler state, or the original trainer iteration.

Therefore, to resume the same job completely from the point where it was interrupted, retain the entire `iter_xxxxxxxxx/` directory together with `latest_checkpoint.txt`, and let the project checkpointer load the DCP directly.
