# A Concise Guide to the PiD Training Configuration System

## Overview

Rather than using the conventional YAML configuration directory together with `@hydra.main`, this project uses a Python-based training configuration system built on the following components:

- `attrs`: defines the typed root configuration and its default values;
- Hydra `ConfigStore`: registers configuration groups for models, data, optimizers, experiments, and other components;
- Hydra `compose()` and OmegaConf: compose the final configuration according to `defaults` and command-line overrides;
- `LazyCall`: records object constructors and their arguments, and instantiates the objects only when training begins.

The training configuration entry point is `pid/_src/configs/pid_training/config.py`. This guide uses
`pixeldit_text_to_image_finetune_res_2048` as an example to explain the entire process.

## Quick Start

Start a four-GPU training job from the repository root:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- experiment="pixeldit_text_to_image_finetune_res_2048"
```

This command can be divided into three parts:

1. `torchrun ... -m scripts.train` launches distributed training through the specified entry point;
2. `--config=.../config.py` specifies the Python root configuration module;
3. everything after the standalone `--` is interpreted as Hydra overrides, and `experiment=...` selects an experiment configuration.

The standalone `--` **must not be omitted**. The project's configuration wrapper checks for it explicitly to distinguish training-script arguments from Hydra arguments. The quotes around the experiment name are optional when the name contains no spaces.


## Configuration Loading Flow

The configuration flows from the command line to the training objects as follows:

```text
scripts/train.py
  -> Import the module specified by --config and call make_config()
  -> Register base configuration groups and import experiment modules to register experiments
  -> Use hydra.compose() to process defaults and command-line overrides
  -> Use OmegaConf.resolve() to resolve ${...} interpolations
  -> Reconstruct the result as an attrs-based Config object
  -> Use LazyCall to instantiate the model and dataloader during training
```

Relevant implementation files:

| File | Purpose |
|---|---|
| `pid/_src/configs/pid_training/config.py` | Root configuration, default group selections, and the entry point for component and experiment registration |
| `pid/_src/configs/pid_training/defaults/` | PixelDiT/PiD-specific model, data, and callback configurations |
| `pid/_src/configs/common/defaults/` | Shared configurations for optimizers, schedulers, checkpoints, and other components |
| `pid/_src/configs/pid_training/experiment_pixeldit_finetune/finetune.py` | PixelDiT fine-tuning experiment configurations and registration |
| `pid/_ext/imaginaire/utils/config_helper.py` | Hydra composition, overrides, and attrs configuration reconstruction |
| `scripts/train.py` | Argument parsing, configuration loading, and training startup |

## Root Configuration and `defaults`

The `Config` class in the entry point extends the project's shared configuration and additionally declares Hydra's `defaults`:

```python
defaults = [
    "_self_",
    {"data_train": "mock_image"},
    {"data_val": "mock_image"},
    {"optimizer": "adamw"},
    {"scheduler": "lambdalinear"},
    {"model": "ddp_pixeldit"},
    {"callbacks": "basic"},
    {"net": None},
    {"conditioner": None},
    {"ema": "power"},
    {"tokenizer": None},
    {"checkpoint": "local"},
    {"ckpt_type": "dummy"},
    {"experiment": None},
]
```

Each entry represents "configuration group name -> currently selected option name." For example, `optimizer: adamw` selects the configuration named `adamw` from the `optimizer` group. `None` reserves a slot for the group without composing a concrete configuration yet; therefore, the command line can use `experiment=<name>` to select an experiment.

`_self_` represents the current configuration node itself. It appears first in the root configuration, so the base groups that follow it can override the root configuration's default values.

## Configuration Groups, Option Names, and Final Paths

Components are registered through `ConfigStore`. The following table lists the main groups in the current entry point and where they are merged into the final configuration:

| Configuration group | Example option | Location in the final configuration |
|---|---|---|
| `data_train` | `mock_image` | `dataloader_train` |
| `data_val` | `mock_image` | `dataloader_val` |
| `optimizer` | `adamw` | `optimizer` |
| `scheduler` | `lambdalinear` | `scheduler` |
| `model` | `ddp_pixeldit` | `_global_`; it can set both `model` and `trainer` |
| `callbacks` | `basic`, `wandb` | `trainer.callbacks` |
| `net` | `pixeldit_h1536_d14p2` | `model.config.net` |
| `conditioner` | `pixeldit_caption` | `model.config.conditioner` |
| `ema` | `power` | `model.config.ema` |
| `tokenizer` | `flux_vae_tokenizer` | `model.config.tokenizer` |
| `checkpoint` | `local` | `checkpoint` |
| `ckpt_type` | `dcp` | `checkpoint.type` |
| `experiment` | `pixeldit_text_to_image_finetune_res_2048` | `_global_` |

The group name determines which configuration is selected, while `package` determines where that configuration is merged. These do not necessarily have the same name. For example, the PixelDiT network is registered as follows:

```python
cs.store(
    group="net",
    name="pixeldit_h1536_d14p2",
    package="model.config.net",
    node=PIXELDIT_H1536_D14P2,
)
```

Therefore, use `net=...` to select the network, but use its final path to override a network field, for example, `model.config.net.rope_mode=original`.

## How Experiment Configurations Are Composed

The core structure of the 2K PixelDiT fine-tuning experiment is shown below:

```python
PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048 = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
            {"override /model": "ddp_pixeldit"},
            {"override /net": "pixeldit_h1536_d14p2"},
            {"override /conditioner": "pixeldit_caption"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb"]},
            {"override /checkpoint": "local"},
            {"override /tokenizer": None},
            "_self_",
        ],
        job=dict(
            group="pixeldit_finetune",
            name="pixeldit_text_to_image_finetune_res_2048",
        ),
        optimizer=dict(lr=1e-5, weight_decay=0.0),
        # Detailed settings for the model, scheduler, checkpoint, trainer, etc.
    )
)
```

There are three key points:

- the leading `/` makes `/data_train` an absolute configuration-group path resolved from the composition root;
- `override` replaces a group option that already exists in the root `defaults`, rather than adding another group with the same name;
- `_self_` appears last in the experiment, so the model, data, and other component configurations are composed first, and the experiment body then overrides their fields. For example, the default learning rate for `adamw` is `1e-4`, while this experiment changes the final value to `1e-5`.

The experiment is then registered with Hydra:

```python
cs.store(
    group="experiment",
    package="_global_",
    name="pixeldit_text_to_image_finetune_res_2048",
    node=PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048,
)
```

`package="_global_"` merges the experiment contents directly into the root configuration. Therefore, the final fields are `trainer.max_iter`, `model.config.image_size`, and so on, rather than `experiment.trainer.max_iter`.

### Experiment Inheritance

The 2K-to-4K experiment builds on the 2K experiment, then replaces the data group and a small number of fields:

```python
defaults=[
    "/experiment/pixeldit_text_to_image_finetune_res_2048",
    {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840"},
    "_self_",
]
```

`_build_debug_run()` uses the same approach to build on the regular experiment, then changes the iteration count, sampling frequency, and W&B mode to values suitable for debugging. The corresponding experiment name is formed by appending `_debug` to the regular experiment name.

## Command-Line Overrides

The command line can select configuration groups and override fields in the final configuration at the same time. A command-line field override takes precedence over values set by the experiment.

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- \
  experiment=pixeldit_text_to_image_finetune_res_2048 \
  trainer.max_iter=1000 \
  trainer.logging_iter=20 \
  optimizer.lr=2e-5 \
  checkpoint.load_path=/path/to/model.pth \
  job.wandb_mode=disabled
```

Common patterns:

```bash
# Select another registered data configuration; this example changes the per-GPU batch size to 2
data_train=pixeldit_MultiAspect_4K_1M_2bs_2048

# Override a Boolean value
trainer.run_validation=true

# Override a list; quote the entire item when it contains special characters such as [] or commas
'scheduler.warm_up_steps=[1000]'

# Select multiple callback configurations
'callbacks=[basic,wandb]'
```

The final precedence can be understood in simplified form as follows:

```text
Base values from make_config()
  < Base groups selected by the root defaults
  < Group replacements or parent experiments in the experiment defaults
  < The experiment itself (_self_ appears last in the experiment)
  < Command-line field overrides
```

Within a single `defaults` list, entries are still composed from first to last.

## Adding Configurations

### Adding a Component Option

1. Define a configuration node using a regular `dict`, an attrs object, or `LazyCall`;
2. register it with `ConfigStore.instance().store(group=..., name=..., package=..., node=...)`;
3. ensure that the corresponding `register_*()` is called by the entry-point `make_config()`;
4. select the option by its group name in the experiment `defaults` or on the command line.

`LazyCall` only stores `_target_` and the constructor arguments; it does not create the model or dataloader when the configuration is imported.

### Adding an Experiment

1. Define a `LazyDict` in an existing experiment package, and arrange `defaults` and `_self_` appropriately;
2. register it under `group="experiment"`, normally with `package="_global_"`;
3. ensure that the module is imported by the entry point. The current entry point recursively imports the experiment packages it lists. If you create a new experiment package, add the corresponding `import_all_modules_from_package()` call to `make_config()`.

Whether an experiment can be selected depends on its module being imported and executing `cs.store()`, not merely on its Python variable name or filename.

## Common Issues

- **You are told that overrides must be preceded by `--`**: check that there is a standalone `--` between `--config` and `experiment=...`.
- **A group or option cannot be found**: check the `group` and `name` passed to `cs.store()`, and verify that the entry point calls the registration function or imports the experiment module.
- **An override path has no effect**: make sure that you are using the final `package` path. For example, network fields are located under `model.config.net`, not `net`.
- **Experiment values are overridden by component defaults**: check the position of `_self_`. Place `_self_` at the end of the experiment `defaults` when the experiment body should take effect last.
- **Adding a top-level field fails**: the final result is reconstructed as a structured attrs configuration, so arbitrary undeclared top-level fields may be rejected. Prefer overriding existing fields, or extend the `Config` definition first.
- **Hydra multirun is used by mistake**: this project does not expose the usual `-m/--multirun` options provided by `@hydra.main`; in this command, `-m scripts.train` is torchrun's module-launch option, not a Hydra multirun flag.
