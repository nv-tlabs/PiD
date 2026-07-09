# Repository Layout

The main PiD code lives in `pid/_src`. The files below are the useful starting
points for inference and training.

## Inference

```text
pid/_src/
├── inference/
│   ├── from_ldm.py             # Text/class → LDM → PiD decode.
│   ├── from_clean.py           # Image → VAE/RAE → PiD decode.
│   ├── decoder.py              # Shared PiD loading and decoding.
│   ├── checkpoint_registry.py  # Maps backbones to released checkpoints.
│   ├── pipeline_registry.py    # Diffusers pipeline definitions.
│   └── prompts/                # Example prompt files.
├── inference_internal/
│   ├── pixeldit.py             # Inference for PixelDiT training checkpoints.
│   └── pid_inference.py        # Inference for PiD teacher/student checkpoints.
└── configs/pid/               # Model configs for released checkpoints.
```

## Training

All training jobs start from `scripts/train.py`.

### Trainer

```text
pid/
├── _ext/imaginaire/trainer.py             # PixelDiT and PiD teacher trainer.
├── _ext/imaginaire/checkpointer/dcp.py    # Their DCP checkpointer.
├── _src/trainer/trainer_distillation.py   # PiD student trainer.
└── _src/checkpointer/dcp_distill.py       # PiD student checkpointer.
```

### Config

```text
pid/_src/configs/
├── pid_training/
│   ├── config.py                       # Training config entry point.
│   ├── defaults/                       # Model, data, and callback presets.
│   ├── experiment_pixeldit_finetune/
│   │   └── finetune.py                  # PixelDiT finetuning.
│   └── experiment_pid_v1pt5_*/          # flux, flux2, qwenimage.
│       ├── teacher.py                   # PiD teacher training.
│       └── distillation.py              # PiD student training.
└── common/defaults/                  # Shared optimizer, scheduler, and model presets.
```

### Model

```text
pid/_src/models/
├── pixeldit_model.py        # PixelDiT training and sampling.
├── pid_model.py             # PiD teacher.
└── pid_distill_model.py     # PiD student distillation.
```

### Network

```text
pid/_src/networks/
├── pixeldit_official.py     # PixelDiT backbone.
├── pid_net.py               # PiD network with LQ conditioning.
├── lq_projection_2d.py      # LQ latent projection.
└── discriminators.py        # PiD student discriminator.
```

Training data and logging code live in `pid/_src/datasets`,
`pid/_src/dataprep/fix_batch_generation`, and `pid/_src/callbacks`.

See [training.md](training.md) for commands and dataset preparation.
